package net.montaki.neuralnetworkdesignpatterns.state;

public final class TrainingState implements NetworkState {
    @Override
    public NetworkStatus status() { return NetworkStatus.TRAINING; }
}
