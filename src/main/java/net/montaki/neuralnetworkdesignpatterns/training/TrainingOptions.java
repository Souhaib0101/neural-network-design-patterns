package net.montaki.neuralnetworkdesignpatterns.training;

public record TrainingOptions(int epochs, double learningRate) {
    public TrainingOptions {
        if (epochs <= 0 || !Double.isFinite(learningRate) || learningRate <= 0) {
            throw new IllegalArgumentException("Nombre d'epoques et taux d'apprentissage strictement positifs requis");
        }
    }
}
