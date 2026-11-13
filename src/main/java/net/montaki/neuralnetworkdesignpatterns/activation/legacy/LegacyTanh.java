package net.montaki.neuralnetworkdesignpatterns.activation.legacy;

public final class LegacyTanh implements LegacyActivation {
    @Override
    public double evaluate(double value) {
        return Math.tanh(value);
    }

    @Override
    public double slope(double value) {
        double output = Math.tanh(value);
        return 1.0 - output * output;
    }
}
