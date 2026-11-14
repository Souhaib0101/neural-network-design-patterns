package net.montaki.neuralnetworkdesignpatterns.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import net.montaki.neuralnetworkdesignpatterns.activation.ActivationFunction;

public final class Neuron {
    private final String id;
    private final int index;
    private ActivationFunction activation;
    private final List<Connection> incoming = new ArrayList<>();
    private final List<Connection> outgoing = new ArrayList<>();

    Neuron(String id, int index, ActivationFunction activation) {
        this.id = id;
        this.index = index;
        this.activation = Objects.requireNonNull(activation);
    }

    public String getId() { return id; }
    public int getIndex() { return index; }
    public ActivationFunction getActivation() { return activation; }
    public List<Connection> getIncoming() { return List.copyOf(incoming); }
    public List<Connection> getOutgoing() { return List.copyOf(outgoing); }

    void setActivation(ActivationFunction activation) { this.activation = Objects.requireNonNull(activation); }
    void addIncoming(Connection connection) { incoming.add(connection); }
    void addOutgoing(Connection connection) { outgoing.add(connection); }
}
