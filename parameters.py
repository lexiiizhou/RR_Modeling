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


# prospect theory
def perceived_probability(alpha, P):
    return [p**alpha/((p**alpha + (1-p)**alpha)**(1/alpha)) for p in P]


"""The protocal was modified such that wait time scales with reward probability"""
def wait_time(p):
    P = [0, 0.2, 0.8, 1]
    t = [7, 5, 3, 1]
    poly = np.polyfit(P, t, 3)
    a, b, c, d = poly[0], poly[1], poly[2], poly[3]
    return round(a*p**3 + b*p**2 + c*p + d, 2)


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
