package net.montaki.neuralnetworkdesignpatterns.training;

import java.util.Arrays;

public final class TrainingData {
    private final double[][] inputs;
    private final double[][] targets;

    public TrainingData(double[][] inputs, double[][] targets) {
        this.inputs = copyAndValidate(inputs);
        this.targets = copyAndValidate(targets);
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Le nombre d'entrees et de cibles doit correspondre");
        }
    }

    private static double[][] copyAndValidate(double[][] values) {
        if (values == null || values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException("Les donnees ne peuvent pas etre vides");
        }
        int width = values[0].length;
        return Arrays.stream(values).map(row -> {
            if (row == null || row.length != width || Arrays.stream(row).anyMatch(value -> !Double.isFinite(value))) {
                throw new IllegalArgumentException("Les lignes doivent etre finies et de meme dimension");
            }
            return row.clone();
        }).toArray(double[][]::new);
    }

    public double[][] inputs() { return copyAndValidate(inputs); }

    public double[][] targets() { return copyAndValidate(targets); }
}
