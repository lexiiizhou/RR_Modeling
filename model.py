from parameters import *


def u_blind_accept(r, p, param_b, R):
    """
    r: restaurant index (1, 2, 3, 4)
    R: list of restaurant values
    p: reward probability
    param_b: [alpha, delta]
    """
    alpha = param_b[0]
    delta = param_b[1]
    prob = perceived_probability(alpha, [p])[0]
    return np.nan_to_num(prob * R[r-1] * delta**(wait_time(p)))
# u_blind_reject = 0


def u_accept(r, p, param, R):
    """
    r: restaurant index (1, 2, 3, 4)
    R: list of restaurant values
    p: reward probability
    param: [alpha, delta, gamma]
    """
    restaurant_iter = {1: 2,
                       2: 3,
                       3: 4,
                       4: 1}

    alpha = param[0]
    delta = param[1]
    gamma = param[2]
    r_next = restaurant_iter[r]  # r_next is the index of the next restaurant
    prob = perceived_probability(alpha, [p])[0]
    prob_conj = gamma + 0.5
    wait_current = wait_time(p)
    wait_next = wait_current + wait_time(prob_conj)
    return np.nan_to_num(prob * R[r - 1] * (delta ** wait_current) + (prob_conj) * R[r_next - 1] * (delta ** wait_next))


def u_reject(r, p, param, R):
    """
    r: restaurant index (1, 2, 3, 4)
    R: list of restaurant values
    p: reward probability
    param: [alpha, delta, gamma]
    """
    restaurant_iter = {1: 2,
                       2: 3,
                       3: 4,
                       4: 1}
    alpha = param[0]
    delta = param[1]
    gamma = param[2]
    r_next = restaurant_iter[r] # r_next is the index of the next restaurant
    prob_conj = gamma + 0.5
    wait = wait_time(prob_conj)
    return np.nan_to_num((prob_conj)*R[r-1]*(delta**wait))


def softmax(V, beta):
    """
    V: n dimensional real vector representating the value of n different options
    """
    p = [1 / np.sum(np.exp(beta * (V - V[a]))) for a in range(len(V))]

    if (np.sum(p) < 0) or (np.sum(p) > 1.000001) or (np.any(np.isnan(p)) or (not np.allclose(np.sum(p), 1))):
        print(p)
        print(beta)
        print(V)
        raise ValueError('p is not a probability')

    return (p)
