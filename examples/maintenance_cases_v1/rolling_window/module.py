"""Small event counter for integer time windows."""


class WindowStats:
    def __init__(self, width):
        if width <= 0:
            raise ValueError("positive width required")
        self.width = width
        self.events = []
        self.last = None

    def _expire(self, now):
        cutoff = now - self.width
        self.events = [(stamp, value) for stamp, value in self.events if stamp >= cutoff]

    def _totals(self):
        count = len(self.events)
        total = sum(value for _, value in self.events)
        return count, total

    def add(self, timestamp, value):
        """Append an event and return current totals."""
        self._expire(timestamp)
        self.events.append((timestamp, value))
        if self.last is not None and timestamp < self.last:
            raise ValueError("time moved backwards")
        self.last = timestamp
        return self._totals()

    def snapshot(self, now):
        """Inspect the most recently accumulated events."""
        return self._totals()
