import os
import pandas as pd
import re


def list_files(dir, type):
    """
    List all files of a certain type in the given dir
    """
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith(type):
                r.append(os.path.join(root, name))
    return r


def stack_data(mouse_files):
    """
    mouse_files = fully_trained_m1/2
    output: pandas dataframe with all trials across sessions stacked
    """
    mouse = pd.read_csv(mouse_files[0])
    session_day = re.findall(r'\d+', mouse_files[0].split('_')[-4])
    mouse['session_day'] = session_day[0]
    for file in mouse_files[1:]:
        session_day = re.findall(r'\d+', file.split('_')[-4])
        df = pd.read_csv(file)
        df['session_day'] = session_day[0]
        mouse = mouse.append(df, ignore_index=True)
    total_sessions = len(mouse_files)
    total_trials = len(mouse)
    mouse['tone_prob'] = mouse['tone_prob'].map(lambda prob: prob / 100)
    mouse.fillna("nan", inplace=True)
    return mouse[['trial_index', 'tone_prob',
                  'restaurant', 'accept',
                  'collection', 'quit', 'lapIndex',
                  'blockIndex', 'session_day']], total_sessions, total_trials


def pellets_by_sess(mouse):
    pellet_percentages = []
    total_sess = mouse['session_day'].unique()
    nocountsession = []
    for i in total_sess:
        # total number of pellets taken
        session_df = mouse[mouse['session_day'] == i]
        pellets = session_df[session_df['collection'] != 'nan']
        R1 = len(pellets[pellets['restaurant'] == 1])/len(pellets)
        R2 = len(pellets[pellets['restaurant'] == 2])/len(pellets)
        R3 = len(pellets[pellets['restaurant'] == 3])/len(pellets)
        R4 = len(pellets[pellets['restaurant'] == 4])/len(pellets)
        single_sess = [R1, R2, R3, R4]
        if 0.0 in single_sess:
            nocountsession.append(i)
        pellet_percentages.append(single_sess)
    return [pellet_percentages, nocountsession]


def throw_nocount(mouse, nocountsession):
    """throw away sessions where the restaurant failed to count"""
    for sess in nocountsession:
        mouse = mouse[mouse['session_day'] != sess]
    return mouse


def satiated_trials(mouse, npellets=None):
    mouse = mouse.copy().reset_index()

    def halfpoint(session_day):
        session = mouse[mouse['session_day'] == session_day]
        totalpellets = sum(session['collected'])
        return round(totalpellets / 2)

    mouse['collected'] = mouse['collection'] != 'nan'
    mouse['pelletcounts'] = -1
    mouse['passhalfpoint'] = 0
    current_session = mouse['session_day'][0]
    current_halfpoint = halfpoint(current_session)
    sessionstart = 0
    row_index = 0
    while row_index < len(mouse):
        if mouse['session_day'][row_index] != current_session:
            mouse['pelletcounts'][row_index] = 0
            sessionstart = row_index
            current_session = mouse['session_day'][row_index]
            current_halfpoint = halfpoint(current_session)
        pelletssofar = sum(mouse[sessionstart:row_index]['collected'])
        mouse['pelletcounts'][row_index] = pelletssofar
        if pelletssofar >= current_halfpoint:
            mouse['passhalfpoint'][row_index] = 1
        else:
            mouse['passhalfpoint'][row_index] = 0
        row_index += 1
    return mouse[mouse['passhalfpoint'] == 1], mouse


def committed_trials(mouse):
    mouse = mouse.copy().reset_index()
    mouse['accept'] = (mouse['accept'] == 1) & (mouse['quit'] == 'nan')
    mouse['accept'] = mouse['accept'].astype(int)
    return mouse

