package net.montaki.neuralnetworkdesignpatterns.activation;

public final class Sigmoid implements ActivationFunction {
    @Override
    public double apply(double[] sums, int outputIndex) {
        double x = sums[outputIndex];
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        double exponential = Math.exp(x);
        return exponential / (1.0 + exponential);
    }

    @Override
    public double derivative(double[] sums, int outputIndex, int inputIndex) {
        double output = apply(sums, outputIndex);
        return outputIndex == inputIndex ? output * (1.0 - output) : 0.0;
    }
}
