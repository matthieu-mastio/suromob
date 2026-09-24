package org.eqasim.core.simulation.modes.drt.mode_choice.predictors;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.events.Event;
import org.matsim.core.controler.events.AfterMobsimEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.AfterMobsimListener;
import org.matsim.core.controler.listener.StartupListener;
import org.matsim.core.events.handler.BasicEventHandler;

import com.google.inject.Singleton;

@Singleton
public class DrtFeedbackService implements BasicEventHandler, AfterMobsimListener, StartupListener {
    private static final Logger log = LogManager.getLogger(DrtFeedbackService.class);

    // Method of Successive Averages (MSA) exponent: gamma_n = 1.0 / (n ^ MSA_EXPONENT)
    // MSA_EXPONENT = 0.75 guarantees Robbins-Monro convergence while retaining adaptability
    private static final double MSA_EXPONENT = 0.75;

    private static class ModeStats {
        final AtomicInteger servedCount = new AtomicInteger(0);
        final AtomicInteger rejectedCount = new AtomicInteger(0);
        final DoubleAdder totalWaitTimeSeconds = new DoubleAdder();

        int updateCount = 0;
        double smoothedRejectionRate = 0.0;
        double smoothedWaitTimeMin = -1.0;
        boolean hasHistory = false;
    }

    private final Map<String, ModeStats> modeStatsMap = new ConcurrentHashMap<>();

    private ModeStats getOrCreateStats(String mode) {
        return modeStatsMap.computeIfAbsent(mode, k -> new ModeStats());
    }

    @Override
    public void notifyStartup(StartupEvent event) {
        modeStatsMap.clear();
    }

    @Override
    public void handleEvent(Event event) {
        String eventType = event.getEventType();
        if ("PassengerRequest scheduled".equals(eventType)) {
            String mode = event.getAttributes().get("mode");
            if (mode == null) mode = "drt";
            ModeStats stats = getOrCreateStats(mode);
            stats.servedCount.incrementAndGet();

            String pickupTimeStr = event.getAttributes().get("pickupTime");
            if (pickupTimeStr != null) {
                try {
                    double pickupTime = Double.parseDouble(pickupTimeStr);
                    double waitTime = Math.max(0.0, pickupTime - event.getTime());
                    stats.totalWaitTimeSeconds.add(waitTime);
                } catch (NumberFormatException ignored) {}
            }
        } else if ("PassengerRequest rejected".equals(eventType)) {
            String mode = event.getAttributes().get("mode");
            if (mode == null) mode = "drt";
            ModeStats stats = getOrCreateStats(mode);
            stats.rejectedCount.incrementAndGet();
        }
    }

    @Override
    public void notifyAfterMobsim(AfterMobsimEvent event) {
        int iteration = event.getIteration();
        for (Map.Entry<String, ModeStats> entry : modeStatsMap.entrySet()) {
            String mode = entry.getKey();
            ModeStats stats = entry.getValue();

            int served = stats.servedCount.getAndSet(0);
            int rejected = stats.rejectedCount.getAndSet(0);
            double totalWait = stats.totalWaitTimeSeconds.sumThenReset();
            int total = served + rejected;

            if (total > 0) {
                stats.updateCount++;
                double gamma = 1.0 / Math.pow(stats.updateCount, MSA_EXPONENT);

                double rawRejectionRate = (double) rejected / total;
                double rawWaitTimeMin = (served > 0) ? (totalWait / served) / 60.0 : 15.0;

                if (!stats.hasHistory) {
                    stats.smoothedRejectionRate = rawRejectionRate;
                    stats.smoothedWaitTimeMin = rawWaitTimeMin;
                    stats.hasHistory = true;
                } else {
                    stats.smoothedRejectionRate = (1.0 - gamma) * stats.smoothedRejectionRate
                            + gamma * rawRejectionRate;
                    stats.smoothedWaitTimeMin = (1.0 - gamma) * stats.smoothedWaitTimeMin
                            + gamma * rawWaitTimeMin;
                }

                log.info(String.format(
                        "[DrtFeedbackService-MSA] Iteration %d (update #%d, gamma=%.4f), Mode '%s': Served=%d, Rejected=%d, RejRate=%.2f%% (MSA=%.2f%%), Wait=%.1f min (MSA=%.1f min)",
                        iteration, stats.updateCount, gamma, mode, served, rejected, rawRejectionRate * 100.0,
                        stats.smoothedRejectionRate * 100.0, rawWaitTimeMin, stats.smoothedWaitTimeMin));
            } else if (stats.hasHistory) {
                // Fleet was idle this iteration (0 requests).
                // Update MSA towards 0% rejection and fast wait time (vehicles standing by)
                stats.updateCount++;
                double gamma = 1.0 / Math.pow(stats.updateCount, MSA_EXPONENT);
                double idleRejectionRate = 0.0;
                double idleWaitTimeMin = 3.0;

                stats.smoothedRejectionRate = (1.0 - gamma) * stats.smoothedRejectionRate
                        + gamma * idleRejectionRate;
                stats.smoothedWaitTimeMin = (1.0 - gamma) * stats.smoothedWaitTimeMin
                        + gamma * idleWaitTimeMin;

                log.info(String.format(
                        "[DrtFeedbackService-MSA] Iteration %d (update #%d, gamma=%.4f), Mode '%s': IDLE (0 requests) -> RejRate MSA=%.2f%%, Wait MSA=%.1f min",
                        iteration, stats.updateCount, gamma, mode, stats.smoothedRejectionRate * 100.0, stats.smoothedWaitTimeMin));
            }
        }
    }

    public double getRejectionProbability(String mode) {
        ModeStats stats = modeStatsMap.get(mode);
        if (stats == null || !stats.hasHistory) {
            return 0.0;
        }
        return stats.smoothedRejectionRate;
    }

    public double getAverageWaitingTimeMin(String mode, double fallbackWaitTimeMin) {
        ModeStats stats = modeStatsMap.get(mode);
        if (stats == null || !stats.hasHistory || stats.smoothedWaitTimeMin < 0) {
            return fallbackWaitTimeMin;
        }
        return stats.smoothedWaitTimeMin;
    }

    @Override
    public void reset(int iteration) {
        // Event handler reset per iteration
    }
}
