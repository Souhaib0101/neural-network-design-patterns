package net.montaki.neuralnetworkdesignpatterns.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;
import net.montaki.neuralnetworkdesignpatterns.activation.ActivationFunction;
import net.montaki.neuralnetworkdesignpatterns.observer.NetworkObserver;
import net.montaki.neuralnetworkdesignpatterns.observer.StateChange;
import net.montaki.neuralnetworkdesignpatterns.security.RequiresRole;
import net.montaki.neuralnetworkdesignpatterns.security.Role;
import net.montaki.neuralnetworkdesignpatterns.state.ConstructionState;
import net.montaki.neuralnetworkdesignpatterns.state.NetworkState;
import net.montaki.neuralnetworkdesignpatterns.state.NetworkStatus;
import net.montaki.neuralnetworkdesignpatterns.state.TrainedState;
import net.montaki.neuralnetworkdesignpatterns.state.TrainingState;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingData;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingOptions;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class NeuralNetwork {
    private static final Logger LOGGER = LoggerFactory.getLogger(NeuralNetwork.class);
    private final List<Layer> layers;
    private final List<Connection> connections;
    private final List<NetworkObserver> observers = new CopyOnWriteArrayList<>();
    private final Object stateLock = new Object();
    private volatile NetworkState state = new ConstructionState();

    NeuralNetwork(List<Layer> layers, List<Connection> connections) {
        this.layers = List.copyOf(layers);
        this.connections = List.copyOf(connections);
    }

    public List<Layer> getLayers() { return layers; }
    public List<Connection> getConnections() { return connections; }
    public NetworkStatus getStatus() { return state.status(); }

    public void addObserver(NetworkObserver observer) { observers.add(Objects.requireNonNull(observer)); }
    public void removeObserver(NetworkObserver observer) { observers.remove(observer); }

    @RequiresRole({Role.ADMIN})
    public void setWeight(int connectionIndex, double weight) {
        synchronized (stateLock) {
            state.requireConfiguration();
            connections.get(connectionIndex).setWeight(weight);
        }
    }

    @RequiresRole({Role.ADMIN})
    public void setActivation(int layerIndex, int neuronIndex, ActivationFunction activation) {
        synchronized (stateLock) {
            state.requireConfiguration();
            if (layerIndex == 0) {
                throw new IllegalArgumentException("La couche d'entree transmet directement les donnees");
            }
            layers.get(layerIndex).getNeurons().get(neuronIndex).setActivation(activation);
        }
    }

    @RequiresRole({Role.ADMIN, Role.TRAINER, Role.VIEWER})
    public double[] predict(double[] input) {
        synchronized (stateLock) {
            state.requirePrediction();
            validateVector(input, layers.getFirst().size());
            return forward(input).outputs()[layers.size() - 1].clone();
        }
    }

    @RequiresRole({Role.ADMIN, Role.TRAINER})
    public TrainingResult fit(TrainingData data, TrainingOptions options) {
        return train(data, options);
    }

    @RequiresRole({Role.ADMIN, Role.TRAINER})
    public TrainingResult train(TrainingData data, TrainingOptions options) {
        Objects.requireNonNull(data);
        Objects.requireNonNull(options);
        double[][] inputs = data.inputs();
        double[][] targets = data.targets();
        for (int i = 0; i < inputs.length; i++) {
            validateVector(inputs[i], layers.getFirst().size());
            validateVector(targets[i], layers.getLast().size());
        }
        NetworkState previous;
        double[] snapshot;
        StateChange start;
        synchronized (stateLock) {
            state.requireTraining();
            previous = state;
            snapshot = connections.stream().mapToDouble(Connection::getWeight).toArray();
            start = changeState(new TrainingState(), "Entrainement demarre");
        }
        notifyObservers(start);
        try {
            double initialLoss = loss(inputs, targets);
            List<Double> history = new ArrayList<>();
            for (int epoch = 0; epoch < options.epochs(); epoch++) {
                double[] gradients = new double[connections.size()];
                for (int sample = 0; sample < inputs.length; sample++) {
                    accumulateGradients(forward(inputs[sample]), targets[sample], gradients, inputs.length);
                }
                for (Connection connection : connections) {
                    double candidate = connection.getWeight() - options.learningRate() * gradients[connection.getIndex()];
                    requireFinite(candidate);
                    // Descente de gradient projetee : les poids restent dans [0,1].
                    connection.setWeight(Math.max(0.0, Math.min(1.0, candidate)));
                }
                history.add(loss(inputs, targets));
            }
            TrainingResult result = new TrainingResult(options.epochs(), initialLoss, history.getLast(), history);
            StateChange finish;
            synchronized (stateLock) {
                finish = changeState(new TrainedState(), "Entrainement termine");
            }
            notifyObservers(finish);
            return result;
        } catch (RuntimeException | Error failure) {
            StateChange rollback;
            synchronized (stateLock) {
                for (int i = 0; i < snapshot.length; i++) {
                    connections.get(i).setWeight(snapshot[i]);
                }
                rollback = changeState(previous, "Echec : poids et etat restaures");
            }
            notifyObservers(rollback);
            throw failure;
        }
    }

    private ForwardPass forward(double[] input) {
        double[][] sums = new double[layers.size()][];
        double[][] outputs = new double[layers.size()][];
        outputs[0] = input.clone();
        sums[0] = input.clone();
        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            sums[l] = new double[layer.size()];
            outputs[l] = new double[layer.size()];
            for (Neuron neuron : layer.getNeurons()) {
                for (Connection connection : neuron.getIncoming()) {
                    sums[l][neuron.getIndex()] += connection.getWeight()
                            * outputs[l - 1][connection.getSource().getIndex()];
                }
                requireFinite(sums[l][neuron.getIndex()]);
            }
            for (Neuron neuron : layer.getNeurons()) {
                double value = neuron.getActivation().apply(sums[l], neuron.getIndex());
                requireFinite(value);
                outputs[l][neuron.getIndex()] = value;
            }
        }
        return new ForwardPass(sums, outputs);
    }

    private void accumulateGradients(ForwardPass pass, double[] target, double[] gradients, int sampleCount) {
        int last = layers.size() - 1;
        double[] upstream = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            upstream[i] = (pass.outputs()[last][i] - target[i]) / sampleCount;
        }
        for (int l = last; l >= 1; l--) {
            Layer layer = layers.get(l);
            double[] delta = new double[layer.size()];
            // Jacobienne complete : inclut les termes croises de Softmax.
            for (int j = 0; j < layer.size(); j++) {
                for (int i = 0; i < layer.size(); i++) {
                    delta[j] += upstream[i] * layer.getNeurons().get(i).getActivation()
                            .derivative(pass.sums()[l], i, j);
                }
            }
            double[] previousUpstream = new double[layers.get(l - 1).size()];
            for (Neuron neuron : layer.getNeurons()) {
                for (Connection connection : neuron.getIncoming()) {
                    int source = connection.getSource().getIndex();
                    gradients[connection.getIndex()] += delta[neuron.getIndex()] * pass.outputs()[l - 1][source];
                    previousUpstream[source] += connection.getWeight() * delta[neuron.getIndex()];
                }
            }
            upstream = previousUpstream;
        }
    }

    private double loss(double[][] inputs, double[][] targets) {
        double total = 0.0;
        for (int sample = 0; sample < inputs.length; sample++) {
            double[] output = forward(inputs[sample]).outputs()[layers.size() - 1];
            for (int i = 0; i < output.length; i++) {
                double error = output[i] - targets[sample][i];
                total += error * error / (2.0 * inputs.length);
            }
        }
        requireFinite(total);
        return total;
    }

    private void validateVector(double[] vector, int expectedSize) {
        if (vector == null || vector.length != expectedSize
                || Arrays.stream(vector).anyMatch(value -> !Double.isFinite(value))) {
            throw new IllegalArgumentException("Vecteur fini de dimension " + expectedSize + " requis");
        }
    }

    private void requireFinite(double value) {
        if (!Double.isFinite(value)) {
            throw new ArithmeticException("Depassement numerique pendant le calcul");
        }
    }

    /** Appelee sous stateLock, notification effectuee apres liberation du verrou. */
    private StateChange changeState(NetworkState next, String reason) {
        NetworkStatus previous = state.status();
        state = next;
        return new StateChange(previous, next.status(), Instant.now(), reason);
    }

    private void notifyObservers(StateChange change) {
        for (NetworkObserver observer : observers) {
            try {
                observer.onStateChanged(change);
            } catch (RuntimeException failure) {
                LOGGER.warn("Observateur ignore apres echec : {}", failure.getClass().getSimpleName());
            }
        }
    }

    private record ForwardPass(double[][] sums, double[][] outputs) { }
}
