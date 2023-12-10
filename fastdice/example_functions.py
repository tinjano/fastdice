# Dice
def pol_keepmax(arr):
    subarr = arr.copy()
    subarr[np.arange(arr.n), subarr.argmax(axis=1)] = 0
    return subarr

def pol_genstraight(arr):
    if (arr.k > arr.s):
        raise ValueError(f'The number of dice must be at most the number of faces. Passed were {k=} and {s=}.')

    subarr = arr > 0
    cum_sum = subarr.cumsum(axis=1)
    k_sum = cum_sum - np.hstack((np.zeros((arr.n, arr.k)), cum_sum))[:, :-arr.k]
    heads = k_sum.argmax(axis=1)[:, None]

    col_indices = np.tile(np.arange(arr.s), arr.n).reshape(arr.n, arr.s)
    mask = (col_indices > heads) | (heads - col_indices >= arr.k)
    subarr[mask] = 0

    return arr - subarr

def rf_y01(arr):
    return np.any(arr == arr.k, axis=1)


def rf_s01(arr):
    if (arr.k > arr.s):
        raise ValueError(f'The number of dice must be at most the number of faces. Passed were {k=} and {s=}.')

    subarr = np.where(arr > 1, 1, arr)
    cum_sum = subarr.cumsum(axis=1)
    k_sum = cum_sum - np.hstack((np.zeros((arr.n, arr.k)), cum_sum))[:, :-arr.k]

    return np.any(k_sum == arr.k, axis=1)

# Chain, MDP
def pol_keepmax_mc(arr):
    subarr = arr[:, ::-1].copy()  # we reverse to keep larger numbers in case of ties
    subarr[np.arange(arr.shape[0]), subarr.argmax(axis=1)] = 0
    return subarr[:, ::-1]

rf_y01 = lambda arrs: np.any(arrs==5, axis=1)
rf_p01 = lambda arrs: np.any(arrs==4, axis=1)
rf_t01 = lambda arrs: np.any(arrs==3, axis=1)

def pol_poison_t(arrs, t):
    indices = np.tile(np.arange(7), (arrs.shape[0], 1)) + 1
    return arrs * ((indices < t) & (indices != 7))

def rf_poison(arrs):
    indices = np.tile(np.arange(7), (arrs.shape[0], 1)) + 1
    return np.sqrt((indices**arrs).prod(axis=1) * (arrs[:, 6] == 0))


def rf_sfpy(arrs):
    rewards = np.zeros((arrs.shape[0],))

    mask = arrs == 5
    rewards += ((np.argmax(mask, axis=1) + 1) * 5 + 50) * np.any(mask, axis=1)

    mask = arrs == 4
    rewards += ((np.argmax(mask, axis=1) + 1) * 4 + 40) * np.any(mask, axis=1)

    mask3, mask2 = arrs == 3, arrs == 2
    rewards += ((np.argmax(mask3, axis=1) + 1) * 3 + (np.argmax(mask2, axis=1) + 1) * 2 + 30) * (
                np.any(mask3, axis=1) & np.any(mask2, axis=1))

    mask = arrs < 2
    rewards += 35 * (np.all(mask, axis=1) & arrs[:, 0]) + 40 * (np.all(mask, axis=1) & arrs[:, 5])

    return rewards