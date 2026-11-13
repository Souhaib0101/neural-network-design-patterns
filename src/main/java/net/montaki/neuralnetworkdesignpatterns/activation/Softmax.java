package net.montaki.neuralnetworkdesignpatterns.activation;

import java.util.Arrays;

public final class Softmax implements ActivationFunction {
    @Override
    public double apply(double[] sums, int outputIndex) {
        double maximum = Arrays.stream(sums).max().orElseThrow();
        double denominator = 0.0;
        for (double sum : sums) {
            denominator += Math.exp(sum - maximum);
        }
        return Math.exp(sums[outputIndex] - maximum) / denominator;
    }

    @Override
    public double derivative(double[] sums, int outputIndex, int inputIndex) {
        double yi = apply(sums, outputIndex);
        double yj = apply(sums, inputIndex);
        return yi * ((outputIndex == inputIndex ? 1.0 : 0.0) - yj);
    }
}
