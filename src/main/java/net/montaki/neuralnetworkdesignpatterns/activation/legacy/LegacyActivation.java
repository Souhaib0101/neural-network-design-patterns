package net.montaki.neuralnetworkdesignpatterns.activation.legacy;

/** Ancien contrat, volontairement different de ActivationFunction. */
public interface LegacyActivation {
    double evaluate(double value);

    double slope(double value);
}
