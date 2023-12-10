class Dice(np.ndarray):
    rng = np.random.default_rng()

    def __new__(cls, n=10_000, s=6, k=5, p=None, dtype=np.uint8):
        if p is None:
            p = np.ones(s) / s
        else:
            s = np.size(p)

        arr = cls.rng.multinomial(k, p, n).astype(dtype).view(cls)
        arr.n = n
        arr.s = s
        arr.k = k
        arr.p = p
        return arr

    def __array_finalize__(self, arr):
        if arr is not None:
            self.n = getattr(arr, 'n', None)
            self.s = getattr(arr, 's', None)
            self.k = getattr(arr, 'k', None)
            self.p = getattr(arr, 'p', None)

    def __floordiv__(self, other):
        return self - other + self.rng.multinomial(other.sum(axis=1).astype(np.int64), self.p)

    def __ifloordiv__(self, other):
        self = self - other + self.rng.multinomial(other.sum(axis=1).astype(np.int64), self.p)
        return self

    def roll(self, *pols, rerolls=2):
        # note: self is a local variable
        pol_cycle = it.cycle(pols)
        for _ in range(rerolls):
            self //= next(pol_cycle)(self)
        return self

