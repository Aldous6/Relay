"""Cache guarded by generations of asynchronous refresh work."""


class GenerationCache:
    def __init__(self):
        self.values = {}
        self.generations = {}

    def begin(self, key):
        """Start a new refresh for a key."""
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation
        return generation

    def commit(self, key, token, value):
        """Publish a completed refresh if its generation is current."""
        if self.generations.get(key) != token:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)

    def invalidate(self, key):
        """Remove a cached key and its generation."""
        self.values.pop(key, None)
        self.generations.pop(key, None)

    def clear(self):
        """Invalidate all currently stored values."""
        for key in list(self.values):
            self.invalidate(key)
