package net.montaki.neuralnetworkdesignpatterns.activation;

import net.montaki.neuralnetworkdesignpatterns.activation.legacy.LegacyTanh;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class ActivationTest {
    @Test
    void sigmoidIsStableAtExtremeValues() {
        Sigmoid sigmoid = new Sigmoid();
        assertThat(sigmoid.apply(new double[]{0}, 0)).isEqualTo(0.5);
        assertThat(sigmoid.apply(new double[]{1000}, 0)).isEqualTo(1);
        assertThat(sigmoid.apply(new double[]{-1000}, 0)).isEqualTo(0);
        assertThat(sigmoid.derivative(new double[]{0}, 0, 0)).isEqualTo(0.25);
    }

    @Test
    void identityIsPointwise() {
        Identity identity = new Identity();
        assertThat(identity.apply(new double[]{2, -3}, 1)).isEqualTo(-3);
        assertThat(identity.derivative(new double[]{2, -3}, 1, 1)).isEqualTo(1);
        assertThat(identity.derivative(new double[]{2, -3}, 1, 0)).isZero();
    }

    @Test
    void softmaxNormalizesTheWholeLayerWithoutOverflow() {
        Softmax softmax = new Softmax();
        double[] sums = {1000, 1001, 1002};
        double total = 0;
        for (int i = 0; i < sums.length; i++) {
            double probability = softmax.apply(sums, i);
            assertThat(probability).isBetween(0.0, 1.0);
            total += probability;
        }
        assertThat(total).isCloseTo(1, within(1e-12));
        assertThat(softmax.apply(sums, 2)).isGreaterThan(softmax.apply(sums, 0));
    }

    @Test
    void fullSoftmaxJacobianMatchesNumericalDifferentiation() {
        Softmax softmax = new Softmax();
        double[] sums = {-0.3, 0.2, 0.8};
        for (int i = 0; i < sums.length; i++) {
            for (int j = 0; j < sums.length; j++) {
                double[] plus = sums.clone();
                double[] minus = sums.clone();
                plus[j] += 1e-6;
                minus[j] -= 1e-6;
                double numerical = (softmax.apply(plus, i) - softmax.apply(minus, i)) / 2e-6;
                assertThat(softmax.derivative(sums, i, j)).isCloseTo(numerical, within(1e-9));
            }
        }
    }

    @Test
    void adapterPreservesTheLegacyFunctionAndItsDerivative() {
        LegacyTanh legacy = new LegacyTanh();
        ActivationFunction adapter = new LegacyActivationAdapter(legacy);
        assertThat(adapter.apply(new double[]{0.7, 2}, 0)).isEqualTo(legacy.evaluate(0.7));
        assertThat(adapter.derivative(new double[]{0.7, 2}, 0, 0)).isEqualTo(legacy.slope(0.7));
        assertThat(adapter.derivative(new double[]{0.7, 2}, 0, 1)).isZero();
    }
}
