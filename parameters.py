from plot_data import *


"""if to differentiate high vs low probability"""


def d_perceived_probability(alpha_low, alpha_high, P):
    PL = [p for p in P if p <= 0.5]
    PH = [p for p in P if p > 0.5]
    perceived = np.array(
        [np.exp(alpha_low * (p - 0.5)) - 1 for p in PL] + [1 - np.exp(-alpha_high * (p - 0.5)) for p in PH])
    return (perceived / 2) + 0.5


def d_plot_actual_vs_perceived(alpha_low, alpha_high):
    P = np.arange(0, 1, 0.02).tolist()
    P_actual = np.array([0, 0.2, 0.8, 1])

    perceived = d_perceived_probability(alpha_low, alpha_high, P)

    fig = plt.figure(figsize=(4, 4))
    plt.plot(P, perceived, label='perceived')
    plt.plot(P_actual, P_actual, marker="o", label='actual')
    plt.plot(P_actual, d_perceived_probability(alpha_low, alpha_high, P_actual), ls="", marker="o")
    plt.title('alpha_low:' + str(alpha_low) + '  alpha_high:' + str(alpha_high))
    plt.grid()
    plt.xlabel('True Probability')
    plt.ylabel('Perceived')
    plt.legend()


def perceived_probability_old(alpha, P):
    return [1/(1+np.exp(-alpha*(p-0.5))) for p in P]


def perceived_probability(alpha, P):
    return [p**alpha/((p**alpha + (1-p)**alpha)**(1/alpha)) for p in P]


def plot_actual_vs_perceived(perceived_probability):
    P = np.arange(0, 1.01, 0.02).tolist()
    P_actual = np.array([0.0, 0.2, 0.8, 1.0])
    alpha = np.arange(0, 1, 0.1)
    alpha = [round(a, 2) for a in alpha]

    fig = plt.figure()
    plt.plot(P_actual, P_actual, marker="D", ls=':', label='actual')
    for a in alpha:
        perceived = perceived_probability(a, P)
        plt.plot(P, perceived, label='alpha:' + str(a))
        plt.plot(P_actual, perceived_probability(a, P_actual), ls="", marker="o", color='#FF796C')

    plt.title('Probability Value')
    plt.grid()
    plt.xlabel('True Probability')
    plt.ylabel('Perceived')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    # fig.savefig('/Users/lexizhou/Desktop/figures/probability_value')

plot_actual_vs_perceived(perceived_probability)


"""The protocal was modified such that wait time scales with reward probability"""
def wait_time(p):
    P = [0, 0.2, 0.8, 1]
    t = [7, 5, 3, 1]
    poly = np.polyfit(P, t, 3)
    a, b, c, d = poly[0], poly[1], poly[2], poly[3]
    return round(a*p**3 + b*p**2 + c*p + d, 2)

"""
model probability as being continuous rather than discrete(i.e. 0, 0.2, 0.8, 1), 
hence wait time becomes continuous
"""

p = np.arange(0, 1, 0.02)
t = np.array([wait_time(i) for i in p])

fig = plt.figure()
plt.plot(p, t)
plt.title('wait time determined by probability')
plt.xlabel('probability')
plt.ylabel('wait time')
# fig.savefig('/Users/lexizhou/Desktop/figures/wait_time')


"""delta(time discounting)"""
delta = np.arange(0.1, 1, 0.1)

"""
reminder:
p = np.arange(0, 1, 0.02)
t = np.array([wait_time(i) for i in p])
"""

fig = plt.figure()
for d in delta:
    time_discounting = []
    for p_i in p:
        time_discounting.append(d**(wait_time(p_i)))
    plt.plot(t, time_discounting, label='delta: '+str(round(d, 4)))
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.title('time discounting param delta')
plt.xlabel('wait time as determined by probability')
plt.ylabel('discounting factor')
# fig.savefig('/Users/lexizhou/Desktop/figures/delta')

# Higher delta means less sensitive to wait time discount


"""probability conjecture"""
n = 10000

uniform_prob = []
for i in range(n):
    uniform_prob.append(np.random.choice([0, 0.2, 0.8, 1]))

fig = plt.figure()
count, bins, ignored = plt.hist(uniform_prob, 8, facecolor='lightsalmon')
plt.plot([0, 0.2, 0.8, 1], list(count[:2]) + list(count[-2:]), color = 'black')
plt.xlabel('p')
plt.ylabel('Count')
plt.title("Underlying Distribution for Reward Probability(Uniform)")
plt.axis([-0.1, 1.1, 0, 5000]) # x_start, x_end, y_start, y_end
plt.grid(True)

"""gamma is similiar to a summary statistics of this underlying distribution"""
gamma = np.arange(-0.5, 0.6, 0.25)
for g in gamma:
    plt.vlines(x=g+0.5, ymin=2000, ymax=2800, ls='-', lw=2)
    plt.text(g+0.45, 3000, 'g: '+str(round(g, 4)), fontsize=12)
plt.text(0.43, 4100, 'Neural', fontsize = 14, color = 'blue')
plt.text(0.8, 4100, 'Optimistic', fontsize = 14, color = 'green')
plt.text(0.0, 4100, 'Pessimistic', fontsize=14, color = 'red')
# fig.savefig('/Users/lexizhou/Desktop/figures/gamma')


def restaurant_val(mouse_df):
    """
    mouse_df: dataframe for a specific mouse
    """
    pellets = mouse_df[mouse_df['collection'] != 'nan']
    R1 = len(pellets[pellets['restaurant'] == 1])*100/len(pellets)
    R2 = len(pellets[pellets['restaurant'] == 2])*100/len(pellets)
    R3 = len(pellets[pellets['restaurant'] == 3])*100/len(pellets)
    R4 = len(pellets[pellets['restaurant'] == 4])*100/len(pellets)
    values = [R1, R2, R3, R4]
    return values
