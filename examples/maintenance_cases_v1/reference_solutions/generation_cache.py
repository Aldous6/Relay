class GenerationCache:
    def __init__(self):
        self.values = {}
        self.pending = {}
        self.next_token = 0

    def begin(self, key):
        self.next_token += 1
        self.pending[key] = self.next_token
        return self.next_token

    def commit(self, key, token, value):
        if key not in self.pending or self.pending[key] != token:
            return False
        del self.pending[key]
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)

    def invalidate(self, key):
        self.values.pop(key, None)
        self.pending.pop(key, None)

    def clear(self):
        self.values.clear()
        self.pending.clear()
