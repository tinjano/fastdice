def get_arrs(k=5, s=6, dtype=np.uint8):
    arrs = np.zeros((1, s), dtype=dtype)
    base_arrs = np.eye(s, dtype=dtype)

    for _ in range(k):
        add_arrs = np.tile(base_arrs, (arrs.shape[0], 1))
        arrs = np.repeat(arrs, s, axis=0) + add_arrs
        arrs = np.unique(arrs, axis=0)

    return arrs


def get_subarrs(k=5, s=6, dtype=np.uint8):
    return np.vstack(
        [get_arrs(sum_, s) for sum_ in range(k + 1)]
    )


def get_tensor(k=5, s=6, subarrs=None, p=None, dtype=np.uint8):
    if p is None:
        p = np.ones(s) / s

    arrs = get_arrs(k, s, dtype)

    if subarrs is None:
        subarrs = get_subarrs(k, s, dtype)

    return multinomial.pmf(
        ((subarrs[:, None, :] - arrs)[:, None, :, :] + arrs[:, None, :]).transpose((0, 2, 1, 3)),
        n=subarrs.sum(axis=1).reshape(subarrs.shape[0], 1, 1),
        p=p
    )


def solve(states, tensor, rf, rerolls=2):
    pols = []
    rewards = rf(states)

    for _ in range(rerolls):
        tensor_eval = (tensor * rewards).sum(axis=2)

        max_indices = tensor_eval.argmax(axis=0)
        pols.append(max_indices)
        rewards = tensor_eval[max_indices, np.arange(tensor_eval.shape[1])]

    return pols[::-1]
