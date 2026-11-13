package net.montaki.neuralnetworkdesignpatterns;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(properties = {"demo.enabled=false", "logging.file.name=target/test.log",
        "logging.level.net.montaki.neuralnetworkdesignpatterns.aspects.LoggingAspect=OFF"})
class NeuralNetworkDesignPatternsApplicationTests {

    @Test
    void contextLoads() {
    }

}
