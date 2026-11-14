package net.montaki.neuralnetworkdesignpatterns.demo;

import java.util.Arrays;
import net.montaki.neuralnetworkdesignpatterns.activation.LegacyActivationAdapter;
import net.montaki.neuralnetworkdesignpatterns.activation.Sigmoid;
import net.montaki.neuralnetworkdesignpatterns.activation.Softmax;
import net.montaki.neuralnetworkdesignpatterns.activation.legacy.LegacyTanh;
import net.montaki.neuralnetworkdesignpatterns.model.NeuralNetwork;
import net.montaki.neuralnetworkdesignpatterns.model.NeuralNetworkBuilder;
import net.montaki.neuralnetworkdesignpatterns.observer.Dashboard;
import net.montaki.neuralnetworkdesignpatterns.security.SecurityContext;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingData;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingOptions;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "demo.enabled", havingValue = "true", matchIfMissing = true)
public class DemoRunner implements CommandLineRunner {
    private final String username;
    private final String password;
    private final int epochs;

    public DemoRunner(@Value("${demo.username:admin}") String username,
                      @Value("${demo.password:1234}") String password,
                      @Value("${demo.epochs:40}") int epochs) {
        this.username = username;
        this.password = password;
        this.epochs = epochs;
    }

    @Override
    public void run(String... args) {
        try {
            SecurityContext.authenticate(username, password);
            NeuralNetwork network = new NeuralNetworkBuilder().inputLayer(3)
                    .hiddenLayer(5, new Sigmoid()).outputLayer(2, new Softmax()).seed(42).build();
            Dashboard dashboard = new Dashboard();
            network.addObserver(dashboard);
            network.setActivation(1, 0, new LegacyActivationAdapter(new LegacyTanh()));
            System.out.println("Topologie : 3 -> 5 -> 2 ; connexions = " + network.getConnections().size());
            try {
                network.predict(new double[]{0.2, 0.4, 0.6});
            } catch (IllegalStateException expected) {
                System.out.println("Avant entrainement : " + expected.getMessage());
            }
            TrainingData data = new TrainingData(
                    new double[][]{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}},
                    new double[][]{{0.2, 0.8}, {0.3, 0.7}, {0.7, 0.3}, {0.6, 0.4}});
            TrainingResult result = network.fit(data, new TrainingOptions(epochs, 0.3));
            System.out.printf(java.util.Locale.ROOT, "Perte initiale=%.6f ; finale=%.6f%n",
                    result.initialLoss(), result.finalLoss());
            System.out.println("Prediction : " + Arrays.toString(network.predict(new double[]{0.2, 0.4, 0.6})));
            dashboard.getHistory().forEach(change -> System.out.println(change.previous() + " -> " + change.current()));
            System.out.println("Entrainements termines : " + dashboard.getCompletedTrainings());
            SecurityContext.authenticate("viewer", "1234");
            System.out.println("Prediction VIEWER : " + Arrays.toString(network.predict(new double[]{1, 0, 0})));
            try {
                network.train(data, new TrainingOptions(1, 0.1));
            } catch (SecurityException expected) {
                System.out.println("Securite : " + expected.getMessage());
            }
        } catch (SecurityException failure) {
            System.out.println("Securite : " + failure.getMessage());
        } finally {
            SecurityContext.clear();
        }
    }
}
