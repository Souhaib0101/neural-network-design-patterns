package net.montaki.neuralnetworkdesignpatterns.model;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import net.montaki.neuralnetworkdesignpatterns.activation.ActivationFunction;
import net.montaki.neuralnetworkdesignpatterns.activation.Identity;
import net.montaki.neuralnetworkdesignpatterns.activation.Sigmoid;
import net.montaki.neuralnetworkdesignpatterns.activation.Softmax;
import net.montaki.neuralnetworkdesignpatterns.observer.Dashboard;
import net.montaki.neuralnetworkdesignpatterns.security.SecurityContext;
import net.montaki.neuralnetworkdesignpatterns.state.NetworkStatus;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingData;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingOptions;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class NeuralNetworkTest {
    @BeforeEach
    void authenticate() { SecurityContext.authenticate("admin", "1234"); }

    @AfterEach
    void clear() { SecurityContext.clear(); }

    @Test
    void builderCreatesTheFullyConnectedThreeFiveTwoTopology() {
        NeuralNetwork network = new NeuralNetworkBuilder().inputLayer(3)
                .hiddenLayer(5, new Sigmoid()).outputLayer(2, new Softmax()).build();
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.CONSTRUCTION);
        assertThat(network.getLayers()).hasSize(3);
        assertThat(network.getLayers().get(0)).isInstanceOf(InputLayer.class);
        assertThat(network.getLayers().get(1)).isInstanceOf(HiddenLayer.class);
        assertThat(network.getLayers().get(2)).isInstanceOf(OutputLayer.class);
        assertThat(network.getConnections()).hasSize(25).allSatisfy(connection -> {
            assertThat(connection.getWeight()).isBetween(0.0, 1.0);
            assertThat(connection.getSource().getOutgoing()).contains(connection);
            assertThat(connection.getDestination().getIncoming()).contains(connection);
        });
        assertThat(network.getLayers().get(1).getNeurons().get(0).getIncoming()).hasSize(3);
        assertThat(network.getLayers().get(1).getNeurons().get(0).getOutgoing()).hasSize(2);
    }

    @Test
    void builderRejectsInvalidTopologyAndIsReproducible() {
        assertThatThrownBy(() -> new NeuralNetworkBuilder().build()).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> new NeuralNetworkBuilder().inputLayer(0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new NeuralNetworkBuilder().hiddenLayer(2, new Identity()))
                .isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> new NeuralNetworkBuilder().inputLayer(1).outputLayer(1, new Identity())
                .hiddenLayer(2, new Identity())).isInstanceOf(IllegalStateException.class);
        NeuralNetwork first = linear();
        NeuralNetwork second = linear();
        assertThat(first.getConnections().getFirst().getWeight()).isEqualTo(second.getConnections().getFirst().getWeight());
        assertThat(first.getConnections().getFirst()).isNotSameAs(second.getConnections().getFirst());
    }

    @Test
    void stateRejectsPredictionBeforeTrainingAndConfigurationAfterwards() {
        NeuralNetwork network = linear();
        assertThatThrownBy(() -> network.predict(new double[]{1})).isInstanceOf(IllegalStateException.class);
        network.setActivation(1, 0, new Identity());
        network.fit(sample(), new TrainingOptions(1, 0.1));
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.TRAINED);
        assertThatThrownBy(() -> network.setWeight(0, 0.3)).isInstanceOf(IllegalStateException.class);
        assertThatThrownBy(() -> network.setActivation(1, 0, new Sigmoid())).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void oneTrainingStepMatchesTheAnalyticalGradient() {
        NeuralNetwork network = linear();
        network.setWeight(0, 0.2);
        TrainingResult result = network.train(sample(), new TrainingOptions(1, 0.5));
        assertThat(network.getConnections().getFirst().getWeight()).isCloseTo(0.5, within(1e-12));
        assertThat(network.predict(new double[]{1})[0]).isCloseTo(0.5, within(1e-12));
        assertThat(result.initialLoss()).isCloseTo(0.18, within(1e-12));
        assertThat(result.finalLoss()).isCloseTo(0.045, within(1e-12));
    }

    @Test
    void realTrainingLearnsAnAchievableLinearFunction() {
        NeuralNetwork network = linear();
        network.setWeight(0, 0.1);
        TrainingResult result = network.fit(new TrainingData(
                new double[][]{{0}, {0.25}, {0.5}, {0.75}, {1}},
                new double[][]{{0}, {0.2}, {0.4}, {0.6}, {0.8}}), new TrainingOptions(300, 0.3));
        assertThat(result.finalLoss()).isLessThan(result.initialLoss() / 1000);
        assertThat(result.lossHistory()).hasSize(300);
        assertThat(network.predict(new double[]{0.5})[0]).isCloseTo(0.4, within(1e-8));
    }

    @Test
    void backpropagationAlsoUpdatesTheHiddenLayer() {
        NeuralNetwork network = new NeuralNetworkBuilder().inputLayer(1).hiddenLayer(1, new Identity())
                .outputLayer(1, new Identity()).build();
        network.setWeight(0, 0.4);
        network.setWeight(1, 0.6);
        network.train(sample(), new TrainingOptions(1, 0.1));
        assertThat(network.getConnections().get(0).getWeight()).isCloseTo(0.4336, within(1e-12));
        assertThat(network.getConnections().get(1).getWeight()).isCloseTo(0.6224, within(1e-12));
        assertThat(network.predict(new double[]{1})[0]).isCloseTo(0.4336 * 0.6224, within(1e-12));
    }

    @Test
    void softmaxBackpropagationMatchesFiniteDifferences() {
        NeuralNetwork network = new NeuralNetworkBuilder().inputLayer(1).outputLayer(2, new Softmax()).build();
        network.setWeight(0, 0.2);
        network.setWeight(1, 0.6);
        TrainingResult result = network.train(new TrainingData(new double[][]{{1.3}}, new double[][]{{0.1, 0.9}}),
                new TrainingOptions(1, 0.01));
        for (int i = 0; i < 2; i++) {
            double[] plus = {0.2, 0.6};
            double[] minus = plus.clone();
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            double gradient = (softmaxLoss(plus) - softmaxLoss(minus)) / 2e-6;
            assertThat(network.getConnections().get(i).getWeight())
                    .isCloseTo((i == 0 ? 0.2 : 0.6) - 0.01 * gradient, within(1e-9));
        }
        assertThat(result.finalLoss()).isLessThan(result.initialLoss());
        assertThat(java.util.Arrays.stream(network.predict(new double[]{1.3})).sum()).isCloseTo(1, within(1e-12));
    }

    @Test
    void weightsStayWithinTheRequiredBounds() {
        NeuralNetwork network = linear();
        assertThatThrownBy(() -> network.setWeight(0, -0.1)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> network.setWeight(0, Double.NaN)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> network.setWeight(0, 1.1)).isInstanceOf(IllegalArgumentException.class);
        network.train(new TrainingData(new double[][]{{1}}, new double[][]{{10}}), new TrainingOptions(3, 10));
        assertThat(network.getConnections().getFirst().getWeight()).isEqualTo(1);
    }

    @Test
    void invalidDimensionsNeverStartTrainingAndInvalidPredictionIsRejected() {
        NeuralNetwork network = linear();
        Dashboard dashboard = new Dashboard();
        network.addObserver(dashboard);
        assertThatThrownBy(() -> network.train(new TrainingData(new double[][]{{1, 2}}, new double[][]{{1}}),
                new TrainingOptions(1, 0.1))).isInstanceOf(IllegalArgumentException.class);
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.CONSTRUCTION);
        assertThat(dashboard.getHistory()).isEmpty();
        network.train(sample(), new TrainingOptions(1, 0.1));
        assertThatThrownBy(() -> network.predict(new double[]{Double.NaN})).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> network.predict(new double[]{1, 2})).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void observersSeeAllTransitionsAndOneBrokenObserverDoesNotBlockOthers() {
        NeuralNetwork network = linear();
        Dashboard dashboard = new Dashboard();
        network.addObserver(change -> { throw new IllegalStateException("Tableau de bord indisponible"); });
        network.addObserver(dashboard);
        network.fit(sample(), new TrainingOptions(1, 0.1));
        network.train(sample(), new TrainingOptions(1, 0.1));
        assertThat(dashboard.getHistory()).extracting(change -> change.current())
                .containsExactly(NetworkStatus.TRAINING, NetworkStatus.TRAINED, NetworkStatus.TRAINING, NetworkStatus.TRAINED);
        assertThat(dashboard.getCompletedTrainings()).isEqualTo(2);
        network.removeObserver(dashboard);
        network.train(sample(), new TrainingOptions(1, 0.1));
        assertThat(dashboard.getHistory()).hasSize(4);
    }

    @Test
    void failedTrainingRestoresWeightsAndConstructionStateAfterAnUpdate() {
        NeuralNetwork network = linear();
        network.setWeight(0, 0.2);
        AtomicInteger calls = new AtomicInteger();
        network.setActivation(1, 0, new ActivationFunction() {
            @Override
            public double apply(double[] sums, int index) {
                if (calls.incrementAndGet() == 3) { throw new IllegalStateException("Echec simule apres mise a jour"); }
                return sums[index];
            }

            @Override
            public double derivative(double[] sums, int i, int j) { return i == j ? 1 : 0; }
        });
        Dashboard dashboard = new Dashboard();
        network.addObserver(dashboard);
        assertThatThrownBy(() -> network.train(sample(), new TrainingOptions(2, 0.5)))
                .isInstanceOf(IllegalStateException.class).hasMessageContaining("simule");
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.CONSTRUCTION);
        assertThat(network.getConnections().getFirst().getWeight()).isEqualTo(0.2);
        assertThat(dashboard.getHistory()).extracting(change -> change.current())
                .containsExactly(NetworkStatus.TRAINING, NetworkStatus.CONSTRUCTION);
    }

    @Test
    void failedRetrainingRestoresTrainedState() {
        NeuralNetwork network = linear();
        network.train(sample(), new TrainingOptions(1, 0.1));
        double weight = network.getConnections().getFirst().getWeight();
        assertThatThrownBy(() -> network.train(new TrainingData(new double[][]{{1}},
                new double[][]{{Double.MAX_VALUE}}), new TrainingOptions(1, 0.1)))
                .isInstanceOf(ArithmeticException.class);
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.TRAINED);
        assertThat(network.getConnections().getFirst().getWeight()).isEqualTo(weight);
        assertThat(network.predict(new double[]{1})[0]).isEqualTo(weight);
    }

    @Test
    void concurrentOperationsAreRejectedWhileTraining() throws Exception {
        NeuralNetwork network = linear();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        network.addObserver(change -> {
            if (change.current() == NetworkStatus.TRAINING) {
                entered.countDown();
                try {
                    if (!release.await(5, TimeUnit.SECONDS)) { throw new IllegalStateException("Timeout test"); }
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(failure);
                }
            }
        });
        try (var executor = Executors.newSingleThreadExecutor()) {
            var task = executor.submit(() -> {
                SecurityContext.authenticate("trainer", "1234");
                try { return network.train(sample(), new TrainingOptions(1, 0.1)); }
                finally { SecurityContext.clear(); }
            });
            try {
                assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();
                assertThatThrownBy(() -> network.predict(new double[]{1})).isInstanceOf(IllegalStateException.class);
                assertThatThrownBy(() -> network.train(sample(), new TrainingOptions(1, 0.1)))
                        .isInstanceOf(IllegalStateException.class);
                assertThatThrownBy(() -> network.setWeight(0, 0.4)).isInstanceOf(IllegalStateException.class);
            } finally {
                release.countDown();
            }
            assertThat(task.get(5, TimeUnit.SECONDS).epochs()).isEqualTo(1);
        }
    }

    private NeuralNetwork linear() {
        return new NeuralNetworkBuilder().inputLayer(1).outputLayer(1, new Identity()).seed(42).build();
    }

    private TrainingData sample() {
        return new TrainingData(new double[][]{{1}}, new double[][]{{0.8}});
    }

    private double softmaxLoss(double[] weights) {
        double a = Math.exp(weights[0] * 1.3);
        double b = Math.exp(weights[1] * 1.3);
        return (Math.pow(a / (a + b) - 0.1, 2) + Math.pow(b / (a + b) - 0.9, 2)) / 2;
    }
}
