package net.montaki.neuralnetworkdesignpatterns.observer;

@FunctionalInterface
public interface NetworkObserver {
    void onStateChanged(StateChange change);
}
