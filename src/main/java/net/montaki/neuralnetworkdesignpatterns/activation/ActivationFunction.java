package net.montaki.neuralnetworkdesignpatterns.activation;

/** Strategy : le contexte vectoriel permet aussi une vraie Softmax. */
public interface ActivationFunction {
    double apply(double[] sums, int outputIndex);

    /** Element (outputIndex,inputIndex) de la jacobienne de l'activation. */
    double derivative(double[] sums, int outputIndex, int inputIndex);
}
