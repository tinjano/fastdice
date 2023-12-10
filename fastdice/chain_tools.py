def get_arrs(k=5, s=6, dtype=np.uint8):
    arrs = np.zeros((1, s), dtype=dtype)
    base_arrs = np.eye(s, dtype=dtype)

    for _ in range(k):
        add_arrs = np.tile(base_arrs, (arrs.shape[0], 1))
        arrs = np.repeat(arrs, s, axis=0) + add_arrs
        arrs = np.unique(arrs, axis=0)

    return arrs


def get_probs(arrs, p=None):
    k = arrs[0].sum()

    if p is None:
        p = np.ones(arrs.shape[1]) / arrs.shape[1]

    return multinomial.pmf(arrs, n=k, p=p)


def get_tm(arrs, pol, p=None):
    if p is None:
        p = np.ones(arrs.shape[1]) / arrs.shape[1]

    subarrs = pol(arrs)

    return multinomial.pmf(
        (subarrs - arrs)[:, None, :] + arrs,
        n=subarrs.sum(axis=1, keepdims=True),
        p=p
    )