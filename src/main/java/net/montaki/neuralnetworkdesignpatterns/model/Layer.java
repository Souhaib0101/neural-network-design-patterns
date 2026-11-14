package net.montaki.neuralnetworkdesignpatterns.model;

import java.util.List;

public abstract sealed class Layer permits InputLayer, HiddenLayer, OutputLayer {
    private final List<Neuron> neurons;

    protected Layer(List<Neuron> neurons) {
        if (neurons.isEmpty()) {
            throw new IllegalArgumentException("Une couche doit contenir au moins un neurone");
        }
        this.neurons = List.copyOf(neurons);
    }

    public List<Neuron> getNeurons() { return neurons; }
    public int size() { return neurons.size(); }
}
