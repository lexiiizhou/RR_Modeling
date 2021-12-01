from main import *
from model import *
from parameters import *
from ipywidgets import *
import random


def simulate_utility(params, rp_combo, R, ax=None):
    def simulate_utility(param):
        def select_utility(param, r, p):
            if len(param) == 4:
                return [u_accept(r, p, param[:-1], R), u_reject(r, p, param[:-1], R)]
            elif len(param) == 3:
                return [u_blind_accept(r, p, param[:-1], R), 0]

        u_as = []
        u_rs = []
        for i in rp_combo:
            r = i[0]
            p = i[1]
            u = select_utility(param, r, p)
            u_as.append(u[0])
            u_rs.append(u[1])
        return np.array(u_as), np.array(u_rs)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    rs = np.array(rp_combo).astype('int').T.tolist()[0]
    rvs = []
    for i in range(len(rs)):
        rvs.append(R[int(rs[i] - 1)])
    ps = np.array(rp_combo).T.tolist()[1]
    ax.set_xlabel('restaurant value', labelpad=20)
    ax.set_ylabel('prob', labelpad=20)
    ax.set_zlabel('utility', labelpad=20)
    ax.set_zlim(0, 60)
    ax.legend()
    R[0] = round(R[0], 3)
    R[1] = round(R[1], 3)
    R[2] = round(R[2], 3)
    R[3] = round(R[3], 3)

    uas, urs = simulate_utility(params)
    ax.scatter(rvs, ps, uas, label='accept')
    ax.scatter(rvs, ps, urs, label='reject')
    ax.set_title('RV: ' + str(R))
    ax.legend()

    return ax


def simulate_utility_interactive(param, rp_combo, R):
    def simulate_utility(param):
        def select_utility(param, r, p):
            if len(param) == 3:
                return [u_accept(r, p, param, R), u_reject(r, p, param, R)]
            elif len(param) == 2:
                return [u_blind_accept(r, p, param, R), 0.0]

        u_as = []
        u_rs = []
        for i in rp_combo:
            r = i[0]
            p = i[1]
            u = select_utility(param, r, p)
            u_as.append(u[0])
            u_rs.append(u[1])
        return np.array(u_as), np.array(u_rs)

    def update1(alpha=15, delta=0.5):
        param = [alpha, delta]
        print(param)
        uas, urs = simulate_utility(param)
        ax.cla()
        ax.scatter(rvs, ps, uas, label='accept')
        ax.scatter(rvs, ps, urs, label='reject')
        ax.set_title('Restauarnt Vals: ' + str(R))
        ax.set_zlim(0, 30)
        ax.legend()

    def update2(alpha=15, delta=0.5, gamma=0):
        param = [alpha, delta, gamma]
        uas, urs = simulate_utility(param)
        ax.cla()
        ax.scatter(rvs, ps, uas, label='accept')
        ax.scatter(rvs, ps, urs, label='reject')
        ax.set_title('Restauarnt Vals: ' + str(R))
        ax.set_zlim(0, 30)
        ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    rs = np.array(rp_combo).astype('int').T.tolist()[0]
    rvs = []
    for i in range(len(rs)):
        rvs.append(R[int(rs[i] - 1)])
    ps = np.array(rp_combo).T.tolist()[1]
    ax.set_xlabel('restaurant value', labelpad=20)
    ax.set_ylabel('prob', labelpad=20)
    ax.set_zlabel('utility', labelpad=20)
    ax.set_zlim(0, 30)
    ax.legend()
    R[0] = round(R[0], 3)
    R[1] = round(R[1], 3)
    R[2] = round(R[2], 3)
    R[3] = round(R[3], 3)
    if len(param) == 4:
        widgets.interact(update2, alpha=param[0], delta=param[1], gamma=param[2])
    elif len(param) == 3:
        widgets.interact(update1, alpha=param[0], delta=param[1])
    # fig.savefig('/Users/lexizhou/Desktop/figures/utility given r, p')


def simulate(param, n, trials, R=None, Random=False):
    """simulate choice for n sessions of #trials w/ fixed beta=5"""
    R_passedin = R

    def select_utility(param, r, p):
        if len(param) == 4:
            return [u_accept(r, p, param[:-1], R), u_reject(r, p, param[:-1], R)]
        elif len(param) == 3:
            return [u_blind_accept(r, p, param[:-1], R), 0.0]

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

    # task parameters
    """randomly generate restaurant values"""
    if R is None:
        R = np.array([random.random(), random.random(), random.random(), random.random()])
        R = (R / sum(R)) * 100
    P = [0.0, 0.2, 0.8, 1.0]
    Data = np.ones(7)

    # loop over sessions
    for b in range(n):

        # initialize values
        r = 1  # index of the first restaurant
        p = np.random.choice(P)
        U = np.array(select_utility(param, r, p))
        # loop over trials
        for t in range(trials):

            # compute softmax probabilities
            choice_p = softmax(U, beta)
            if Random == True:
                choice_p = [0.5, 0.5]

            # pick action
            a = np.random.choice([1, 0], p=choice_p)

            # store data
            Data = np.vstack((Data, np.array([b, t, r, p, a, U, choice_p])))

            # go to the next restaurant and generate new prob
            r = restaurant_iter[r]
            p = np.random.choice(P)
            U = np.array(select_utility(param, r, p))

    if R_passedin is None:
        return Data[1:], R  # remove first row
    else:
        return Data[1:]


"""only for model 2, since model one will have no accept trials"""


def plot_sim_interactive(param, sessions, trials):
    R = np.array([random.random(), random.random(), random.random(), random.random()])
    R = (R / sum(R)) * 100

    def simulate_choice(param):
        simulated_data = simulate(param, sessions, trials, R)
        simulated_data = pd.DataFrame(simulated_data, columns=['session', 'trial', 'r', 'p', 'choice', 'U', 'choice_p'])

        all_accepts = []
        for i in range(4):
            data = simulated_data[simulated_data['r'] == (i + 1)]
            accepts = data[data['choice'] == 1]

            if len(accepts) == 0:
                return R, 0, 0, 0, 0

            zero = len(accepts[accepts['p'] == 0.0]) / len(accepts)
            twenty = len(accepts[accepts['p'] == 0.2]) / len(accepts)
            eighty = len(accepts[accepts['p'] == 0.8]) / len(accepts)
            hundred = len(accepts[accepts['p'] == 1.0]) / len(accepts)
            accept_perc = [zero, twenty, eighty, hundred]
            all_accepts.append(accept_perc)

        all_accepts = np.array(all_accepts).T.tolist()
        zeros = np.array(all_accepts[0])
        twentys = np.array(all_accepts[1])
        eightys = np.array(all_accepts[2])
        hundreds = np.array(all_accepts[3])
        return R, zeros, twentys, eightys, hundreds

    def update1(alpha=15, delta=0.5, beta=5):
        param = [alpha, delta, beta]
        R, zeros, twentys, eightys, hundreds = simulate_choice(param)

        x_pos = ['R1: ' + str(round(R[0], 2)), 'R2: ' + str(round(R[1], 2)), 'R3: ' + str(round(R[2], 2)),
                 'R4: ' + str(round(R[3], 2))]
        ax1.cla()
        ax1.bar(x_pos, zeros, color='lightsalmon', label='0.0')
        ax1.bar(x_pos, twentys, bottom=zeros, color='powderblue', label='0.2')
        ax1.bar(x_pos, eightys, bottom=zeros + twentys, color='yellowgreen', label='0.8')
        ax1.bar(x_pos, hundreds, bottom=zeros + twentys + eightys, color='khaki', label='1.0')
        ax1.set_xlabel('Restaurants')
        ax1.set_ylabel('accept percentage')
        ax1.set_ylim(0, 1)
        ax1.set_title('model 1: ' +
                      'alpha: ' + str(round(param[0], 4)) +
                      ' | ' + 'delta: ' + str(round(param[1], 4)) +
                      ' | ' + 'beta: ' + str(round(param[2], 4)))
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        ax2.cla()
        simulate_utility(param, rp_combo, R, ax2)

    def update2(alpha=15, delta=0.5, gamma=0, beta=5):
        param = [alpha, delta, gamma, beta]
        R, zeros, twentys, eightys, hundreds = simulate_choice(param)

        x_pos = ['R1: ' + str(round(R[0], 2)), 'R2: ' + str(round(R[1], 2)), 'R3: ' + str(round(R[2], 2)),
                 'R4: ' + str(round(R[3], 2))]
        ax1.cla()
        ax1.bar(x_pos, zeros, color='lightsalmon', label='0.0')
        ax1.bar(x_pos, twentys, bottom=zeros, color='powderblue', label='0.2')
        ax1.bar(x_pos, eightys, bottom=zeros + twentys, color='yellowgreen', label='0.8')
        ax1.bar(x_pos, hundreds, bottom=zeros + twentys + eightys, color='khaki', label='1.0')
        ax1.set_xlabel('Restaurants')
        ax1.set_ylabel('accept percentage')
        ax1.set_ylim(0, 1)
        ax1.set_title('model 2: ' +
                      'alpha: ' + str(round(param[0], 4)) +
                      ' | ' + 'delta: ' + str(round(param[1], 4)) +
                      ' | ' + 'gamma: ' + str(round(param[2], 4)) +
                      ' | ' + 'beta: ' + str(round(param[3], 4)))
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        ax2.cla()
        simulate_utility(param, rp_combo, R, ax2)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plt.subplots_adjust(wspace=0.5)
    if len(param) == 4:
        widgets.interact(update2, alpha=param[0], delta=param[1], gamma=param[2], beta=param[3])
    elif len(param) == 3:
        widgets.interact(update1, alpha=param[0], delta=param[1], beta=param[2])


def plot_sim(param, sessions, trials, Random=False, R=None):
    simulated_data = None
    if Random == True:
        simulated_data, R = simulate(param, sessions, trials, Random=True)
    if R is not None:
        simulated_data = simulate(param, sessions, trials, R=R)
    else:
        simulated_data, R = simulate(param, sessions, trials)
    simulated_data = pd.DataFrame(simulated_data, columns=['session', 'trial', 'r', 'p', 'choice', 'U', 'choice_p'])

    all_accepts = []
    for i in range(4):
        data = simulated_data[simulated_data['r'] == (i + 1)]
        accepts = data[data['choice'] == 1]

        if len(accepts) == 0:
            return 'no accepts', simulated_data, param

        zero = len(accepts[accepts['p'] == 0.0]) / len(accepts)
        twenty = len(accepts[accepts['p'] == 0.2]) / len(accepts)
        eighty = len(accepts[accepts['p'] == 0.8]) / len(accepts)
        hundred = len(accepts[accepts['p'] == 1.0]) / len(accepts)
        accept_perc = [zero, twenty, eighty, hundred]
        all_accepts.append(accept_perc)

    all_accepts = np.array(all_accepts).T.tolist()
    zeros = np.array(all_accepts[0])
    twentys = np.array(all_accepts[1])
    eightys = np.array(all_accepts[2])
    hundreds = np.array(all_accepts[3])
    x_pos = ['R1: ' + str(round(R[0], 2)), 'R2: ' + str(round(R[1], 2)), 'R3: ' + str(round(R[2], 2)),
             'R4: ' + str(round(R[3], 2))]

    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x_pos, zeros, color='lightsalmon', label='0.0')
    ax1.bar(x_pos, twentys, bottom=zeros, color='powderblue', label='0.2')
    ax1.bar(x_pos, eightys, bottom=zeros + twentys, color='yellowgreen', label='0.8')
    ax1.bar(x_pos, hundreds, bottom=zeros + twentys + eightys, color='khaki', label='1.0')
    ax1.set_xlabel('probabilities')
    ax1.set_ylabel('accept percentage')
    ax1.set_ylim(0, 1)
    ax1.set_title('model 2: ' +
                  'alpha: ' + str(round(param[0], 4)) +
                  ' | ' + 'delta: ' + str(round(param[1], 4)) +
                  ' | ' + 'gamma: ' + str(round(param[2], 4)))
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    simulate_utility(param, rp_combo, R, ax2)
    return fig, simulated_data, param


def fakedata_wparam(sessions, trials, alphas, deltas, gammas):
    params = []
    for a in alphas:
        for d in deltas:
            for g in gammas:
                params.append([a, d, g])
    fakedata=[]
    for i in params:
        data, R = simulate(i, sessions, restaurant_iter, trials)
        fakedata.append(data)
    return fakedata


def simulate_wparam(sessions, trials, alphas, deltas, gammas):
    params = []
    for a in alphas:
        for d in deltas:
            for g in gammas:
                params.append([a, d, g])
    simulated_data = []
    for i in params:
        fig, fakedata, pa = plot_sim(i, sessions, trials)
        if fig != 'no accepts':
            fig.savefig('/Users/lexizhou/Desktop/figures/model1sim/'+str(i)+'.png')
        simulated_data.append(fakedata)
    return fig, simulated_data, params
