package net.montaki.neuralnetworkdesignpatterns.training;

import org.junit.jupiter.api.Test;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TrainingDataTest {
    @Test
    void dataOwnsDefensiveCopies() {
        double[][] input = {{1}};
        double[][] target = {{0.8}};
        TrainingData data = new TrainingData(input, target);
        input[0][0] = 99;
        target[0][0] = 99;
        data.inputs()[0][0] = 77;
        assertThat(data.inputs()[0][0]).isEqualTo(1);
        assertThat(data.targets()[0][0]).isEqualTo(0.8);
    }

    @Test
    void dataRejectsEmptyRaggedNonFiniteAndMismatchedRows() {
        assertThatThrownBy(() -> new TrainingData(new double[0][], new double[0][]))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new TrainingData(new double[][]{{1}, {1, 2}}, new double[][]{{1}, {2}}))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new TrainingData(new double[][]{{Double.NaN}}, new double[][]{{1}}))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new TrainingData(new double[][]{{1}}, new double[][]{{1}, {2}}))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void optionsRejectInvalidTrainingParameters() {
        assertThatThrownBy(() -> new TrainingOptions(0, 0.1)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new TrainingOptions(1, -0.1)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new TrainingOptions(1, Double.POSITIVE_INFINITY)).isInstanceOf(IllegalArgumentException.class);
    }
}
