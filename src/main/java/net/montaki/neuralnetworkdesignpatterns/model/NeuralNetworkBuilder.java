package net.montaki.neuralnetworkdesignpatterns.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import net.montaki.neuralnetworkdesignpatterns.activation.ActivationFunction;
import net.montaki.neuralnetworkdesignpatterns.activation.Identity;
import net.montaki.neuralnetworkdesignpatterns.security.RequiresRole;
import net.montaki.neuralnetworkdesignpatterns.security.Role;

/** Builder : construction ordonnee, validation de la topologie et initialisation reproductible. */
public final class NeuralNetworkBuilder {
    private final List<LayerSpec> specifications = new ArrayList<>();
    private boolean outputDefined;
    private long seed = 42;

    public NeuralNetworkBuilder inputLayer(int size) {
        if (!specifications.isEmpty()) {
            throw new IllegalStateException("La couche d'entree doit etre unique et placee en premier");
        }
        add(size, new Identity());
        return this;
    }

    public NeuralNetworkBuilder hiddenLayer(int size, ActivationFunction activation) {
        requireOpenTopology();
        add(size, activation);
        return this;
    }

    public NeuralNetworkBuilder outputLayer(int size, ActivationFunction activation) {
        requireOpenTopology();
        add(size, activation);
        outputDefined = true;
        return this;
    }

    public NeuralNetworkBuilder seed(long seed) {
        this.seed = seed;
        return this;
    }

    @RequiresRole({Role.ADMIN})
    public NeuralNetwork build() {
        if (!outputDefined) {
            throw new IllegalStateException("Une couche d'entree et une couche de sortie sont requises");
        }
        List<Layer> layers = new ArrayList<>();
        for (int l = 0; l < specifications.size(); l++) {
            LayerSpec spec = specifications.get(l);
            List<Neuron> neurons = new ArrayList<>();
            for (int n = 0; n < spec.size(); n++) {
                neurons.add(new Neuron("L" + l + "N" + n, n, spec.activation()));
            }
            layers.add(l == 0 ? new InputLayer(neurons) : l == specifications.size() - 1
                    ? new OutputLayer(neurons) : new HiddenLayer(neurons));
        }
        List<Connection> connections = new ArrayList<>();
        Random random = new Random(seed);
        for (int l = 1; l < layers.size(); l++) {
            for (Neuron destination : layers.get(l).getNeurons()) {
                for (Neuron source : layers.get(l - 1).getNeurons()) {
                    Connection connection = new Connection(connections.size(), source, destination, random.nextDouble());
                    connections.add(connection);
                    source.addOutgoing(connection);
                    destination.addIncoming(connection);
                }
            }
        }
        return new NeuralNetwork(layers, connections);
    }

    private void add(int size, ActivationFunction activation) {
        if (size <= 0) {
            throw new IllegalArgumentException("Taille de couche strictement positive requise");
        }
        specifications.add(new LayerSpec(size, Objects.requireNonNull(activation)));
    }

    private void requireOpenTopology() {
        if (specifications.isEmpty() || outputDefined) {
            throw new IllegalStateException("Definir l'entree, puis les couches cachees, puis la sortie");
        }
    }

    private record LayerSpec(int size, ActivationFunction activation) { }
}
