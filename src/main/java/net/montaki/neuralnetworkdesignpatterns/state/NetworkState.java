package net.montaki.neuralnetworkdesignpatterns.state;

public interface NetworkState {
    NetworkStatus status();

    default void requireConfiguration() {
        throw new IllegalStateException("Configuration interdite dans l'etat " + status());
    }

    default void requireTraining() {
        throw new IllegalStateException("Entrainement interdit dans l'etat " + status());
    }

    default void requirePrediction() {
        throw new IllegalStateException("Prediction interdite dans l'etat " + status());
    }
}
