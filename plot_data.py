import matplotlib.pyplot as plt
import numpy as np
from utils import *

fig_save = '/home/wholebrain/Desktop/modeling_figures'


def plot_pellets(mouse, ID):
    """
    m_accept: dataframe of accept trials
    """
    pellet_perc = np.array(pellets_by_sess(mouse)[0]).T.tolist()
    sessions = mouse['session_day'].unique()

    R1 = np.array(pellet_perc[0])
    R2 = np.array(pellet_perc[1])
    R3 = np.array(pellet_perc[2])
    R4 = np.array(pellet_perc[3])

    fig, ax = plt.subplots()
    ax.bar(sessions, R4, color='lightsalmon', label='Chocolate')
    ax.bar(sessions, R3, bottom=R4, color='powderblue', label='Grape')
    ax.bar(sessions, R2, bottom=R3 +R4, color='yellowgreen', label='Plain')
    ax.bar(sessions, R1, bottom=R2 + R3 + R4, color='khaki', label='Banana')
    ax.set_xlabel('session (day)')
    ax.set_ylabel('%pellets')
    ax.set_ylim(0, 1)
    ax.set_title(ID +': %pellets accepted across sessions')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.savefig(fig_save + '/flavor preference')
    return ax


def plot_prob(mouse, ID, R, fig=False):
    """
    m_accept: dataframe of accept trials
    """
    all_accepts = []
    for i in range(4):
        data = mouse[mouse['restaurant'] == ( i +1)]
        accepts = data[data['accept'] == 1]

        if len(accepts) == 0:
            return 'no accepts', 'nothing', 'here'

        zero = len(accepts[accepts['tone_prob'] == 0] ) /len(accepts)
        twenty = len(accepts[accepts['tone_prob'] == 0.2] ) /len(accepts)
        eighty = len(accepts[accepts['tone_prob'] == 0.8] ) /len(accepts)
        hundred= len(accepts[accepts['tone_prob'] == 1] ) /len(accepts)
        accept_perc = [zero, twenty, eighty, hundred]
        all_accepts.append(accept_perc)

    all_accepts = np.array(all_accepts).T.tolist()
    zeros = np.array(all_accepts[0])
    twentys = np.array(all_accepts[1])
    eightys = np.array(all_accepts[2])
    hundreds = np.array(all_accepts[3])
    x_pos = ['R1:  ' +str(round(R[0] ,2)), 'R2:  ' +str(round(R[1] ,2)), 'R3:  ' +str(round(R[2] ,2)), 'R4:  ' +str(round(R[3] ,2))]

    fig, ax = plt.subplots()
    ax.bar(x_pos, zeros, color='lightsalmon', label='0.0')
    ax.bar(x_pos, twentys, bottom=zeros, color='powderblue', label='0.2')
    ax.bar(x_pos, eightys, bottom=zeros +twentys, color='yellowgreen', label='0.8')
    ax.bar(x_pos, hundreds, bottom=zeros +twentys +eightys, color='khaki', label='1.0')
    ax.set_xlabel('Restaurant')
    ax.set_ylabel('accept percentage')
    ax.set_ylim(0, 1)
    ax.set_title(ID + ' :%pellets accepted from all sessions')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    return ax


