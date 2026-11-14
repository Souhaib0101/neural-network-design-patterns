package net.montaki.neuralnetworkdesignpatterns.state;

public final class TrainedState implements NetworkState {
    @Override
    public NetworkStatus status() { return NetworkStatus.TRAINED; }

    @Override
    public void requireTraining() { }

    @Override
    public void requirePrediction() { }
}
