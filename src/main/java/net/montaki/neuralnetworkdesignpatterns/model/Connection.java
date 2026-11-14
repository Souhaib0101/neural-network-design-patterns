package net.montaki.neuralnetworkdesignpatterns.model;

import java.util.Objects;

public final class Connection {
    private final int index;
    private final Neuron source;
    private final Neuron destination;
    private double weight;

    Connection(int index, Neuron source, Neuron destination, double weight) {
        this.index = index;
        this.source = Objects.requireNonNull(source);
        this.destination = Objects.requireNonNull(destination);
        setWeight(weight);
    }

    public int getIndex() { return index; }
    public Neuron getSource() { return source; }
    public Neuron getDestination() { return destination; }
    public double getWeight() { return weight; }

    void setWeight(double weight) {
        if (!Double.isFinite(weight) || weight < 0 || weight > 1) {
            throw new IllegalArgumentException("Un poids doit appartenir a [0,1]");
        }
        this.weight = weight;
    }
}
