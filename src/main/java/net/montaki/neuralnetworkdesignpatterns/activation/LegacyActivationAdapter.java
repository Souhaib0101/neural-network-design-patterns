package net.montaki.neuralnetworkdesignpatterns.activation;

import java.util.Objects;
import net.montaki.neuralnetworkdesignpatterns.activation.legacy.LegacyActivation;

public final class LegacyActivationAdapter implements ActivationFunction {
    private final LegacyActivation legacy;

    public LegacyActivationAdapter(LegacyActivation legacy) {
        this.legacy = Objects.requireNonNull(legacy);
    }

    @Override
    public double apply(double[] sums, int outputIndex) {
        return legacy.evaluate(sums[outputIndex]);
    }

    @Override
    public double derivative(double[] sums, int outputIndex, int inputIndex) {
        return outputIndex == inputIndex ? legacy.slope(sums[outputIndex]) : 0.0;
    }
}
