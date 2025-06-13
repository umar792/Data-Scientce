import random
from collections import defaultdict, Counter

class MarkovCoinPredictor:
    """
    A simple order-k Markov predictor for sequences of coin flips (1/2).
    It watches the last k flips as the “state” and records how often each next flip follows that state.
    When asked to predict, it looks at the most recent k flips, finds which next value (1 or 2)
    occurred more often in that context, and outputs that as the prediction.
    """

    def __init__(self, k=3):
        """
        k = order of the Markov model (how many past flips we condition on)
        """
        self.k = k
        # transitions[(state_tuple)][next_flip] = count
        self.transitions = defaultdict(Counter)
        self._trained = False

    def train(self, history):
        """
        Build the transition counts from a list of past flips.
        history: list of integers (each 1 or 2)
        After you call train(), you can call predict().
        """
        if len(history) < self.k + 1:
            raise ValueError(f"Need at least k+1 (={self.k+1}) flips to build transitions.")

        for i in range(len(history) - self.k):
            state = tuple(history[i : i + self.k])      # last k flips as a tuple
            nxt = history[i + self.k]                   # the flip that followed
            self.transitions[state][nxt] += 1

        self._trained = True

    def predict(self, recent_history):
        """
        Given the most recent flips (a list of length >= k), return 1 or 2 as the prediction.
        If the exact state was never seen in training, fall back to the overall most common next flip.
        recent_history: list of ints (each 1/2), length >= k
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train(history) first.")

        if len(recent_history) < self.k:
            raise ValueError(f"Need at least k (={self.k}) recent flips to predict.")

        state = tuple(recent_history[-self.k :])
        counter = self.transitions.get(state, None)

        if counter is None or sum(counter.values()) == 0:
            # unseen state → fallback: look up overall distribution of next flips
            all_counts = Counter()
            for subcounter in self.transitions.values():
                all_counts.update(subcounter)
            if not all_counts:
                # no data at all?
                return random.choice([1, 2])
            # pick whichever of 1 or 2 is more common overall
            return 1 if all_counts[1] >= all_counts[2] else 2

        # if we have data for this state, pick whichever next‐flip had higher count
        if counter[1] > counter[2]:
            return 1
        elif counter[2] > counter[1]:
            return 2
        else:
            # tie → choose uniformly at random between 1 and 2
            return random.choice([1, 2])


if __name__ == "__main__":
    # Example usage:

    # 1. Suppose your teacher gave you this observed sequence of flips so far:
    history = [2,2,1,1,1,2,2,1,1,1,1,2,2,2,1,2,1,1,2,2,2,1,2,2,2,2,1,2,1,1,1,1,1,2,2,2,1,2,2,1,2,2,1,2,1,2,2,1,2,2,2,1,2,1,2,2,2,1,1,2,2,2,1,2,
               2,1,1,1,2,1,2,1,1,2,2,1,1,2,2,2,1,1,1,2,2,2,1,2,1,1,1,2,1,1,2,1,1,2,1,2,1,2,2,2,1,1,2,2,1,1,1,1,2,2,1,1,2,1,2,2,2,2,2,1,1,1,]
    # 2. Create a predictor of order-3 (you can adjust k to 2,3,4,...)
    predictor = MarkovCoinPredictor(k=3)

    # 3. Train it on the history
    predictor.train(history)

    # 4. Ask for a prediction, given the most recent 3 flips:
    next_flip_pred = predictor.predict(history)
    print(f"Based on last {predictor.k} flips, I predict next flip = {next_flip_pred!r}")

    # 5. If you want to test “online”, you can simulate:
    #    (a) generate a “true” random flip each time,
    #    (b) see whether you predicted correctly,
    #    (c) append the true flip to history, retrain (or update), repeat.
    #
    # Here’s a short simulation showing how it might perform if the generator
    # actually had a small built-in bias (e.g. P(1)=0.6, P(2)=0.4).
    print("\n--- Simulation on a biased coin (P(1)=0.6, P(2)=0.4) ---")
    biased_history = history.copy()
    hits = 0
    trials = 50
    for _ in range(trials):
        pred = predictor.predict(biased_history)
        true_flip = 1 if random.random() < 0.6 else 2
        if pred == true_flip:
            hits += 1
        biased_history.append(true_flip)
        # update model incrementally (you could also retrain fully each time):
        if len(biased_history) >= predictor.k + 1:
            state = tuple(biased_history[-(predictor.k+1) : -1])
            predictor.transitions[state][true_flip] += 1

    print(f"Out of {trials} trials, correct predictions: {hits} ({hits/trials:.2%})")
