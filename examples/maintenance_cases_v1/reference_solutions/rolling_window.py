from collections import deque


class WindowStats:
    def __init__(self, width):
        if type(width) is not int or width <= 0:
            raise ValueError("invalid width")
        self.width = width
        self.events = deque()
        self.last = None
        self.total = 0

    def _advance(self, now):
        if type(now) is not int or (self.last is not None and now < self.last):
            raise ValueError("invalid time")
        while self.events and self.events[0][0] <= now - self.width:
            self.total -= self.events.popleft()[1]
        self.last = now

    def add(self, timestamp, value):
        if type(value) is not int:
            raise ValueError("invalid value")
        self._advance(timestamp)
        self.events.append((timestamp, value))
        self.total += value
        return len(self.events), self.total

    def snapshot(self, now):
        self._advance(now)
        return len(self.events), self.total
