package net.montaki.neuralnetworkdesignpatterns.observer;

import java.time.Instant;
import net.montaki.neuralnetworkdesignpatterns.state.NetworkStatus;

public record StateChange(NetworkStatus previous, NetworkStatus current, Instant at, String reason) {
}
