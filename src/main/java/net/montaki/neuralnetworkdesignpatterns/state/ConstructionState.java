package net.montaki.neuralnetworkdesignpatterns.state;

public final class ConstructionState implements NetworkState {
    @Override
    public NetworkStatus status() { return NetworkStatus.CONSTRUCTION; }

    @Override
    public void requireConfiguration() { }

    @Override
    public void requireTraining() { }
}
