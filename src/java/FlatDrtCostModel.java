package org.eqasim.ile_de_france.mode_choice.costs;

import java.util.List;
import org.eqasim.core.simulation.mode_choice.cost.AbstractCostModel;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.contribs.discrete_mode_choice.model.DiscreteModeChoiceTrip;

public class FlatDrtCostModel extends AbstractCostModel {
    private final double flatFare;

    public FlatDrtCostModel() {
        super("drt");
        // Tarif unique fixé à 2.00 euros pour le DRT
        this.flatFare = 2.0;
    }

    @Override
    public double calculateCost_MU(Person person, DiscreteModeChoiceTrip trip, List<? extends PlanElement> elements) {
        // Retourne toujours le même tarif, peu importe la distance
        return flatFare; 
    }
}
