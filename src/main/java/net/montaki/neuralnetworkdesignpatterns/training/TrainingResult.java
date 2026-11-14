package net.montaki.neuralnetworkdesignpatterns.training;

import java.util.List;

public record TrainingResult(int epochs, double initialLoss, double finalLoss, List<Double> lossHistory) {
    public TrainingResult {
        lossHistory = List.copyOf(lossHistory);
    }
}
