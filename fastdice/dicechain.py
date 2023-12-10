class DiceChain:
    def __init__(self, *pols, k=5, s=6, p=None, dtype=np.uint8):
        if p is None:
            p = np.ones(s) / s

        self.states = get_arrs(k, s, dtype)
        self.init_probs = get_probs(self.states, p)
        self.trans_mats = tuple(get_tm(self.states, pol, p) for pol in pols)

    def _roll_reg(self, mat, probs, rerolls):
        return probs @ np.linalg.matrix_power(mat, rerolls)

    def _roll_irr(self, *mats, probs, rerolls):
        mat_cycle = it.cycle(mats)
        mat = np.eye(self.states.shape[0])
        for _ in range(rerolls):
            mat = mat @ next(mat_cycle)
        return probs @ mat

    def roll(self, *mats, probs=None, rerolls=2):
        mats = self.trans_mats if not mats else mats
        probs = self.init_probs if probs is None else probs

        if len(mats) == 1:
            return self._roll_reg(mats[0], probs=probs, rerolls=rerolls)
        else:
            return self._roll_irr(mats, probs=probs, rerolls=rerolls)

    def random_walk(self, *mats, n=10_000, probs=None, rerolls=2):
        mats = self.trans_mats if not mats else mats
        probs = self.init_probs if probs is None else probs

        mat_cycle = it.cycle(mats)

        indices = np.random.choice(self.states.size, n, probs, replace=True)
        for _ in range(rerolls):
            indices = (next(mat_cycle).cumsum(axis=1) > np.random.uniform(self.states.size)[:, None]).argmax(axis=1)

        return np.bincount(indices) / self.states.size

    def score(self, *rfs, probs=None):
        probs = self.init_probs if probs is None else probs
        return np.vstack([rf(self.states) * probs for rf in rfs])