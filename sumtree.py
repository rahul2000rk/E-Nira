import numpy as np

class SumTree:
    """
    A SumTree data structure for prioritized experience replay.
    It allows for efficient sampling of experiences based on their priorities
    and updating priorities.
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaves
        self.data = np.zeros(capacity, dtype=object) # Stores actual experiences
        self.n_entries = 0 # Current number of experiences in memory

    def _propagate(self, idx, change):
        """Propagates change up the tree from leaf to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrieves a leaf node's index from the tree based on a value `s`."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        """Returns the sum of all priorities (value of the root node)."""
        return self.tree[0]

    def add(self, priority, data):
        """Adds a new experience and its priority to the tree."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # Wrap around

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """Updates the priority of an existing experience at `idx`."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """
        Retrieves an experience and its leaf index based on a sampled value `s`
        (where `s` is between 0 and total_priority).
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])