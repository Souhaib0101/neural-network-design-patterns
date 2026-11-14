package net.montaki.neuralnetworkdesignpatterns.observer;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public final class Dashboard implements NetworkObserver {
    private final List<StateChange> history = new CopyOnWriteArrayList<>();

    @Override
    public void onStateChanged(StateChange change) {
        history.add(change);
    }

    public List<StateChange> getHistory() {
        return List.copyOf(history);
    }

    public long getCompletedTrainings() {
        return history.stream().filter(change -> change.current()
                == net.montaki.neuralnetworkdesignpatterns.state.NetworkStatus.TRAINED
                && "Entrainement termine".equals(change.reason())).count();
    }
}
