package org.eqasim.core.simulation.modes.drt.mode_choice.predictors;

import org.eqasim.core.simulation.mode_choice.cost.CostModel;
import org.eqasim.core.simulation.mode_choice.utilities.predictors.PredictorUtils;
import org.eqasim.core.simulation.modes.drt.mode_choice.variables.DrtVariables;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.contrib.drt.routing.DrtRoute;
import org.matsim.contribs.discrete_mode_choice.model.DiscreteModeChoiceTrip;
import org.matsim.core.router.TripStructureUtils;

import java.util.List;
import java.util.Map;

public class DefaultDrtPredictor implements DrtPredictor {
    private final Map<String, CostModel> costModels;
    private final DrtFeedbackService feedbackService;

    public DefaultDrtPredictor(Map<String, CostModel> costModels, DrtFeedbackService feedbackService) {
        this.costModels = costModels;
        this.feedbackService = feedbackService;
    }

    @Override
    public DrtVariables predictVariables(Person person, DiscreteModeChoiceTrip trip, List<? extends PlanElement> elements) {
        double travelTime_min = 0.0;
        double accessEgressTime_min = 0.0;
        double cost_MU = 0.0;
        double waitingTime_min = 0.0;
        double rejectionProbability = 0.0;

        for (Leg leg : TripStructureUtils.getLegs(elements)) {
            if (costModels.containsKey(leg.getMode())) {
                DrtRoute route = (DrtRoute) leg.getRoute();

                // Travel time estimate: use direct ride time with typical 20% pooling detour (avoid SLA maxTravelTime buffer)
                double directRideTime = route.getDirectRideTime();
                if (directRideTime > 0.0) {
                    travelTime_min = (directRideTime * 1.2) / 60.0;
                } else {
                    travelTime_min = route.getMaxTravelTime() / 60.0;
                }

                // Waiting time from feedback (or realistic 5.0 min fallback if no history)
                double fallbackWaitTime_min = Math.min(5.0, route.getConstraints().maxWaitDuration() / 60.0);
                waitingTime_min = feedbackService.getAverageWaitingTimeMin(leg.getMode(), fallbackWaitTime_min);

                // Rejection probability from feedback
                rejectionProbability = feedbackService.getRejectionProbability(leg.getMode());

                cost_MU = costModels.get(leg.getMode()).calculateCost_MU(person, trip, elements);

            } else if (leg.getMode().equals(TransportMode.walk)) {
                accessEgressTime_min += leg.getTravelTime().seconds() / 60.0;
            } else {
                throw new IllegalStateException("Encountered unknown mode in DrtPredictor: " + leg.getMode());
            }
        }

        double euclideanDistance_km = PredictorUtils.calculateEuclideanDistance_km(trip);

        return new DrtVariables(travelTime_min, cost_MU, euclideanDistance_km, waitingTime_min, accessEgressTime_min, rejectionProbability);
    }
}
