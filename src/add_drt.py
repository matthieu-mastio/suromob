import os
import re
import gzip
import random
import argparse

def get_random_links(network_path, num_links):
    """Pick random links from the network that allow 'car' mode.
    
    DRT vehicles must be placed on car-accessible links because MATSim
    filters the network by mode. Links with only 'rail', 'pt', 'walk', etc.
    would cause a 'Start link ... is null' crash at runtime.
    """
    links = []
    try:
        with gzip.open(network_path, 'rt') as f:
            for line in f:
                if '<link id="' in line:
                    # Only keep links that allow 'car' mode
                    modes_match = re.search(r'modes="([^"]+)"', line)
                    if modes_match:
                        modes = modes_match.group(1).replace(",", " ").split()
                        if "car" not in modes:
                            continue
                    match = re.search(r'<link id="([^"]+)"', line)
                    if match:
                        links.append(match.group(1))
        if not links:
            print(f"Warning: No car-accessible links found in {network_path}")
            return []
        return random.sample(links, min(num_links, len(links)))
    except Exception as e:
        print(f"Warning: Could not read network file to pick random links: {e}")
        return []

def create_drt_vehicles(path, network_path="network.xml.gz", capacities=[]):
    num_vehicles = len(capacities)
    if num_vehicles == 0:
        print("No vehicles to create.")
        return

    start_links = get_random_links(network_path, num_vehicles)
    if not start_links:
        start_links = ["25928"] * num_vehicles
    elif len(start_links) < num_vehicles:
        start_links = (start_links * (num_vehicles // len(start_links) + 1))[:num_vehicles]
        
    xml_content = ['<?xml version="1.0" ?>\n<vehicles xmlns="http://matsim.org/files/dtd">']
    for i, capacity in enumerate(capacities, start=1):
        start_link = start_links[i-1]
        xml_content.append(f'    <vehicle id="drt_{i}" start_link="{start_link}" capacity="{capacity}" t_0="0.0" t_1="108000.0"/>')
    xml_content.append('</vehicles>')
    with open(path, 'w') as f:
        f.write('\n'.join(xml_content))

def update_config(config_path, out_config_path, max_wait_time, max_travel_time_alpha, drt_constant):
    with open(config_path, 'r') as f:
        content = f.read()

    # Update DiscreteModeChoice cachedModes
    content = re.sub(
        r'(<param name="cachedModes" value=")([^"]+)(" \/>)',
        r'\g<1>\g<2>, drt\g<3>',
        content
    )

    # Update DiscreteModeChoice availableModes
    content = re.sub(
        r'(<param name="availableModes" value=")([^"]+)(" \/>)',
        r'\g<1>\g<2>, drt\g<3>',
        content
    )
    # Change modeAvailability to Default so drt isn't filtered out by Eqasim IDF code
    content = re.sub(
        r'(<param name="modeAvailability" value=")[^"]+(" />)',
        r'\g<1>Default\g<2>',
        content
    )
    # Change fallbackBehaviour to INITIAL_CHOICE to prevent crashes if an agent has no feasible modes
    content = re.sub(
        r'(<param name="fallbackBehaviour" value=")[^"]+(" />)',
        r'\g<1>INITIAL_CHOICE\g<2>',
        content
    )

    # Add DrtUtilityEstimator and FlatDrtCostModel for drt to Eqasim configuration
    content = re.sub(
        r'(<parameterset type="estimator" >\s*<param name="estimator" value="ZeroUtilityEstimator" />\s*<param name="mode" value="outside" />\s*</parameterset>)',
        r'\n\t\t<parameterset type="cost_model" >\n\t\t\t<param name="mode" value="drt" />\n\t\t\t<param name="model" value="FlatDrtCostModel" />\n\t\t</parameterset>\n\t\t\g<1>\n\t\t<parameterset type="estimator" >\n\t\t\t<param name="estimator" value="DrtUtilityEstimator" />\n\t\t\t<param name="mode" value="drt" />\n\t\t</parameterset>',
        content
    )

    # Add dvrp and multiModeDrt before </config>
    drt_modules = f"""
	<module name="dvrp">
		<parameterset type="travelTimeMatrix">
			<parameterset type="SquareGridZoneSystem">
				<param name="cellSize" value="500"/>
			</parameterset>
		</parameterset>
	</module>
	<module name="multiModeDrt">
		<parameterset type="drt">
			<parameterset type="ExtensiveInsertionSearch"/>
			<param name="mode" value="drt"/>
			<param name="vehiclesFile" value="drt_vehicles.xml"/>
			<param name="operationalScheme" value="door2door"/>
			<param name="stopDuration" value="60.0"/>
			<parameterset type="drtOptimizationConstraints">
				<param name="maxTravelTimeAlpha" value="{max_travel_time_alpha}"/>
				<param name="maxTravelTimeBeta" value="1200.0"/>
				<param name="maxWaitTime" value="{max_wait_time}"/>
				<param name="rejectRequestIfMaxWaitOrTravelTimeViolated" value="true"/>
			</parameterset>
		</parameterset>
	</module>
"""
    content = content.replace('</config>', drt_modules + '</config>')

    # 4. Add DRT scoring parameters inside <module name="scoring" >
    content = content.replace(
        '<param name="simStarttimeInterpretation" value="maxOfStarttimeAndEarliestActivityEnd" />',
        '<param name="simStarttimeInterpretation" value="onlyUseStarttime" />'
    )
    
    scoring_params = f"""
		<parameterset type="activityParams" >
			<param name="activityType" value="outside" />
			<param name="closingTime" value="undefined" />
			<param name="earliestEndTime" value="undefined" />
			<param name="latestStartTime" value="undefined" />
			<param name="minimalDuration" value="undefined" />
			<param name="openingTime" value="undefined" />
			<param name="priority" value="1.0" />
			<param name="scoringThisActivityAtAll" value="false" />
			<param name="typicalDuration" value="undefined" />
			<param name="typicalDurationScoreComputation" value="relative" />
		</parameterset>
		<parameterset type="activityParams" >
			<param name="activityType" value="outside interaction" />
			<param name="closingTime" value="undefined" />
			<param name="earliestEndTime" value="undefined" />
			<param name="latestStartTime" value="undefined" />
			<param name="minimalDuration" value="undefined" />
			<param name="openingTime" value="undefined" />
			<param name="priority" value="1.0" />
			<param name="scoringThisActivityAtAll" value="false" />
			<param name="typicalDuration" value="undefined" />
			<param name="typicalDurationScoreComputation" value="relative" />
		</parameterset>
		<parameterset type="activityParams" >
			<param name="activityType" value="drt interaction" />
			<param name="scoringThisActivityAtAll" value="false" />
		</parameterset>
		<parameterset type="modeParams" >
			<param name="constant" value="{drt_constant}" />
			<param name="marginalUtilityOfDistance_util_m" value="0.0" />
			<param name="marginalUtilityOfTraveling_util_hr" value="-6.0" />
			<param name="mode" value="drt" />
			<param name="monetaryDistanceRate" value="0.0" />
		</parameterset>
"""
    idx = content.find('<module name="subtourModeChoice"')
    if idx != -1:
        r_idx = content.rfind('</module>', 0, idx)
        if r_idx != -1:
            content = content[:r_idx] + scoring_params + content[r_idx:]
            
    with open(out_config_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure DRT for Eqasim simulation")
    parser.add_argument('--base-dir', type=str, default='.', help="Base directory of the scenario")
    parser.add_argument('--nb-4', type=int, default=30, help="Number of 4-seat shuttles")
    parser.add_argument('--nb-6', type=int, default=0, help="Number of 6-seat shuttles")
    parser.add_argument('--nb-15', type=int, default=0, help="Number of 15-seat shuttles")
    parser.add_argument('--nb-20', type=int, default=0, help="Number of 20-seat shuttles")
    parser.add_argument('--max-wait-time', type=float, default=1200.0, help="maxWaitTime for DRT (in seconds)")
    parser.add_argument('--max-travel-time-alpha', type=float, default=1.5, help="maxTravelTimeAlpha multiplier")
    parser.add_argument('--drt-constant', type=float, default=-0.5, help="Mode constant (ASC) for DRT in scoring")

    args = parser.parse_args()

    # Build the capacity list based on arguments
    capacities = [4] * args.nb_4 + [6] * args.nb_6 + [15] * args.nb_15 + [20] * args.nb_20

    base_dir = args.base_dir
    network_path = os.path.join(base_dir, 'network.xml.gz')
    vehicles_path = os.path.join(base_dir, 'drt_vehicles.xml')
    config_in_path = os.path.join(base_dir, 'config.xml')
    config_out_path = os.path.join(base_dir, 'config_drt.xml')

    create_drt_vehicles(vehicles_path, network_path=network_path, capacities=capacities)
    update_config(
        config_in_path,
        config_out_path,
        max_wait_time=args.max_wait_time,
        max_travel_time_alpha=args.max_travel_time_alpha,
        drt_constant=args.drt_constant
    )
    print(f"Successfully generated DRT setup in {base_dir}")
    print(f"Total vehicles: {len(capacities)}")
    print(f"maxWaitTime: {args.max_wait_time}, maxTravelTimeAlpha: {args.max_travel_time_alpha}, constant: {args.drt_constant}")
