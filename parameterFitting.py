import copy
from scipy.optimize import minimize
from model import *


def llh(param, data, R=None, minimize=True, savedata=False):
    df = copy.deepcopy(data)

    restaurant_iter = {1: 2,
                       2: 3,
                       3: 4,
                       4: 1}

    if R is None:
        R = restaurant_val(data)

    def select_utility(param, r, p):
        if len(param) == 4:
            u_a = u_accept(r, p, param[:-1], R)
            u_r = u_reject(r, p, param[:-1], R)
        elif len(param) == 3:
            u_a = u_blind_accept(r, p, param[:-1], R)
            u_r = 0.0
        return [u_r, u_a]

    # model parameters
    if len(param) == 4:
        alpha = param[0]
        delta = param[1]
        gamma = param[2]
        beta = param[3]
    elif len(param) == 3:
        alpha = param[0]
        delta = param[1]
        beta = param[2]

    llh = 0

    # Isolate r, p, and choice. Don't care about session in this case
    partial_data = data[['tone_prob', 'restaurant', 'accept']]
    Data = np.ones(5)

    for i in range(len(data)):
        tone = float(partial_data.loc[i][0])  # data: tone probability
        r = int(partial_data.loc[i][1])  # data: restaurant
        a = int(partial_data.loc[i][2])  # data: choice

        u = np.array(select_utility(param, r, tone))
        # compute softmax probabilities
        p = softmax(u, beta)

        # updata log likelihood
        if p[a] < 1e-5:
            p[a] = 1 - 0.999
        llh += np.log(p[a])
        """a=1 means accept, a=0 means reject"""
        Data = np.vstack((Data, np.array([a, u, p, p[a], llh])))

    if minimize == True:
        return -llh

    Data = Data[1:]

    df['a'] = Data[:, 0]
    df['u'] = Data[:, 1]
    df['p'] = Data[:, 2]
    df['likelihood'] = Data[:, 3]
    df['llh'] = Data[:, 4]
    lh = np.exp(llh / data.shape[0])
    if savedata == False:
        return -llh, lh, param
    if minimize == True:
        return -llh
    return -llh, lh, param, df


def optimize(fname,
             R,
             bounds,
             Data,
             niter,
             toplot=False,
             ):
    outcomes = np.full([niter, len(bounds) + 1], np.nan)
    optimcurve = np.full(niter, np.nan)
    for i in range(niter):
        # random starting point based on maximum bounds
        params0 = np.array([bound[1] * np.random.rand() for bound in bounds])

        # compute the function value at the starting point
        llh0 = fname(params0, Data, R)

        # run the optimizer with constraints
        result = minimize(fun=fname, x0=params0, args=(Data, R), bounds=bounds)
        x = result.x
        bestllh = fname(x, Data, R)
        outcomes[i, :] = [bestllh] + [xi for xi in x]
        optimcurve[i] = min(outcomes[:(i + 1), 0])

    # find the global minimum out of all outcomes
    i = np.argwhere(outcomes[:, 0] == np.min(outcomes[:, 0]))
    bestparameters = outcomes[i[0], 1:].flatten()
    bestllh = -1 * outcomes[i[0], 0].flatten()[0]

    # plot the best llh found by the optimizer as a function of iteration number.
    if toplot:
        plt.figure()
        plt.plot(range(niter), np.round(optimcurve, 6), 'o-')
        plt.xlabel('iteration')
        plt.ylabel('best minimum llh')

    return (bestparameters, bestllh)


def llh_reparameterized(param, data, R=None):
    P = [0, 0.2, 0.8, 1.0]
    df = copy.deepcopy(data)

    restaurant_iter = {1: 2,
                       2: 3,
                       3: 4,
                       4: 1}

    if R is None:
        R = restaurant_val(data)

    def select_utility(param, r, p):
        if len(param) == 4:
            u_a = u_accept(r, p, param[:-1], R)
            u_r = u_reject(r, p, param[:-1], R)
        elif len(param) == 3:
            u_a = u_blind_accept(r, p, param[:-1], R)
            u_r = 0.0
        return [u_r, u_a]

    # model parameters
    if len(param) == 4:
        alpha = param[0]
        delta = param[1]
        gamma = param[2]
        beta = param[3]
    elif len(param) == 3:
        alpha = param[0]
        delta = param[1]
        beta = param[2]

    llh = 0

    # Isolate r, p, and choice. Don't care about session in this case
    partial_data = data[['p', 'r', 'choice']]
    total_sess = data['session'].unique()

    for i in total_sess:
        r = 1  # index of the first restaurant
        p = np.random.choice(P)
        U = np.array(select_utility(param, r, p))
        trials = len(data[data['session'] == i])
        data = data[data['session'] == i]

        for t in range(trials):
            # compute softmax probabilities
            p = softmax(U, beta)
            a = data['choice'][t]

            # updata log likelihood
            if p[a] < 1e-5:
                p[a] = 1 - 0.999
            llh += np.log(p[a])
            """a=1 means accept, a=0 means reject"""

            # Update to the next restaurant with new probability
            r = restaurant_iter[r]
            p = np.random.choice(P)
            U = np.array(select_utility(param, r, p))

    lh = np.exp(llh / data.shape[0])
    return -llh
