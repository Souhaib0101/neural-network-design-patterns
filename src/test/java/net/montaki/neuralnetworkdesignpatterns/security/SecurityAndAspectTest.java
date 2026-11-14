package net.montaki.neuralnetworkdesignpatterns.security;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import java.util.concurrent.Executors;
import net.montaki.neuralnetworkdesignpatterns.activation.Identity;
import net.montaki.neuralnetworkdesignpatterns.aspects.LoggingAspect;
import net.montaki.neuralnetworkdesignpatterns.model.NeuralNetwork;
import net.montaki.neuralnetworkdesignpatterns.model.NeuralNetworkBuilder;
import net.montaki.neuralnetworkdesignpatterns.state.NetworkStatus;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingData;
import net.montaki.neuralnetworkdesignpatterns.training.TrainingOptions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SecurityAndAspectTest {
    @AfterEach
    void clear() { SecurityContext.clear(); }

    @Test
    void anonymousCannotConstructANetworkEvenWithNew() {
        assertThatThrownBy(this::network).isInstanceOf(SecurityException.class)
                .hasMessageContaining("Authentification requise");
    }

    @Test
    void viewerCanPredictButCannotTrainOrConfigure() {
        SecurityContext.authenticate("admin", "1234");
        NeuralNetwork network = network();
        network.train(data(), new TrainingOptions(1, 0.1));
        SecurityContext.authenticate("viewer", "1234");
        assertThat(network.predict(new double[]{1})).hasSize(1);
        assertThatThrownBy(() -> network.train(data(), new TrainingOptions(1, 0.1))).isInstanceOf(SecurityException.class);
        assertThatThrownBy(() -> network.fit(data(), new TrainingOptions(1, 0.1))).isInstanceOf(SecurityException.class);
        assertThatThrownBy(() -> network.setWeight(0, 0.5)).isInstanceOf(SecurityException.class);
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.TRAINED);
    }

    @Test
    void trainerCanTrainAnExistingNetwork() {
        SecurityContext.authenticate("admin", "1234");
        NeuralNetwork network = network();
        SecurityContext.authenticate("trainer", "1234");
        network.fit(data(), new TrainingOptions(1, 0.1));
        assertThat(network.getStatus()).isEqualTo(NetworkStatus.TRAINED);
        assertThatThrownBy(this::network).isInstanceOf(SecurityException.class);
    }

    @Test
    void failedAuthenticationRemovesThePreviousIdentity() {
        SecurityContext.authenticate("admin", "1234");
        assertThatThrownBy(() -> SecurityContext.authenticate("admin", "incorrect")).isInstanceOf(SecurityException.class);
        assertThat(SecurityContext.isAuthenticated()).isFalse();
        assertThat(SecurityContext.hasRole(Role.ADMIN)).isFalse();
        assertThatThrownBy(() -> SecurityContext.authenticate("inconnu", "1234")).isInstanceOf(SecurityException.class);
    }

    @Test
    void identityDoesNotLeakToAnotherThread() throws Exception {
        SecurityContext.authenticate("admin", "1234");
        try (var executor = Executors.newSingleThreadExecutor()) {
            assertThat(executor.submit(SecurityContext::isAuthenticated).get()).isFalse();
        }
    }

    @Test
    void compileTimeAspectLogsPrivateMethodsAndPreservesTheResult() {
        SecurityContext.authenticate("admin", "1234");
        Logger logger = (Logger) LoggerFactory.getLogger(LoggingAspect.class);
        Level original = logger.getLevel();
        ListAppender<ILoggingEvent> appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
        logger.setLevel(Level.DEBUG);
        try {
            NeuralNetwork network = network();
            network.setWeight(0, 0.2);
            network.train(data(), new TrainingOptions(1, 0.5));
            assertThat(network.predict(new double[]{1})[0]).isEqualTo(0.5);
            assertThat(appender.list).extracting(ILoggingEvent::getFormattedMessage)
                    .anyMatch(message -> message.contains("AVANT NeuralNetwork.forward"))
                    .anyMatch(message -> message.contains("APRES NeuralNetwork.predict") && message.contains("duree="))
                    .noneMatch(message -> message.contains("SecurityContext.authenticate") || message.contains("password="));
            assertThatThrownBy(() -> network.predict(new double[]{1, 2})).isInstanceOf(IllegalArgumentException.class);
            assertThat(appender.list).extracting(ILoggingEvent::getFormattedMessage)
                    .anyMatch(message -> message.contains("ECHEC NeuralNetwork.predict"));
        } finally {
            logger.detachAppender(appender);
            logger.setLevel(original);
            appender.stop();
        }
    }

    private NeuralNetwork network() {
        return new NeuralNetworkBuilder().inputLayer(1).outputLayer(1, new Identity()).build();
    }

    private TrainingData data() { return new TrainingData(new double[][]{{1}}, new double[][]{{0.8}}); }
}
