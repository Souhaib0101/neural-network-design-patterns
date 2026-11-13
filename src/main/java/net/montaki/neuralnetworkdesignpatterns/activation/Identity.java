package net.montaki.neuralnetworkdesignpatterns.activation;

public final class Identity implements ActivationFunction {
    @Override
    public double apply(double[] sums, int outputIndex) {
        return sums[outputIndex];
    }

    @Override
    public double derivative(double[] sums, int outputIndex, int inputIndex) {
        return outputIndex == inputIndex ? 1.0 : 0.0;
    }
}
