import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.stats import norm

#return the info from excell file, and the path of the directory name 
def catch_info_xls(la_file):
    dirname, basename = os.path.split(la_file) #basename we don't care it's just to split the path to just get the folder (dirname)
    read_file = pd.read_excel(la_file ,header=None)
    
    return read_file, dirname

#return the info from the json with all informations about the frenquencies and amplitude file
#from the the same folder as the directory name get by the catch_info_xls function
def catch_info_json(directory_name):
    for fichier in os.listdir(directory_name):

        if fichier.endswith(".json"):

            chemin_json = os.path.join(directory_name, fichier)

            with open(chemin_json, "r") as read_file:
                info_json = json.load(read_file)

            return info_json

#return a list of the position of different trial in the excel file (the lines where is the string "New trial")
#so at the end, list_index_newtrial is a list with all the position of each "New trial" in the excel file
def search_nb_newtrial(data, nb_trial):
    list_index_newtrial = []

    for i in range (nb_trial): #pick the number of trial
        my_index = data[data[4] == 'New trial'].index[i]
        list_index_newtrial.append(my_index)

    return list_index_newtrial

#return index of all the end trial in the columm E of the excell file (the lines where is the string "The Trial ended")
#so at the end, list_index_endtrial is a list with all the position of each "The Trial ended" in the excel file
def search_nb_endtrial(data, nb_trial):
    list_index_endtrial = []

    for i in range (nb_trial):
        my_index = data[data[4] == 'The trial ended'].index[i]#c'est à la colonne 4+1
        list_index_endtrial.append(my_index)

    return list_index_endtrial

#here we want for each trial, the numbers of lines between "New trial" and "The trial ended" in order to navigate and obtain information on events
# at the end, liste_diff_newEndTrial return a liste of numbers of lines for each event
def diff_new_end_trial(where_newtrial, where_endtrial):
    liste_diff_newEndTrial = []

    for icount_liste in range (len(where_newtrial)):
        diff = where_endtrial[icount_liste] - where_newtrial[icount_liste]
        liste_diff_newEndTrial.append(diff)
    
    return liste_diff_newEndTrial

###Jusque la, cest le meme code que dans main_function###

'''
Objectif de cette partie : 
- recuperer le type de trial (Go-NoGo)
- recuperer l'outcome (Hit, Miss, Correct Rejection, False Alarm)
- recuperer le iti2 (intervalle de temps entre la fin de la parturbation (si elle a lieu) et le stimulus (si Go))
'''
def genotype_souris(nom_souris):
    gen = "WT"
    if "SIS" in nom_souris:
        gen = "SIS"
    return gen

def trial_type(json, trial_number):
    return json[trial_number]["trial_type"]#return the trial type for a specific trial number

def touch(data, start, end):
    """Returns wether it was a touch trial or not"""
    touch = "NoTouch"
    for i in range(start, end):
        event = str(data.iloc[i][4])
        if "SocialTouch" in event:
            touch = "Touch"
    return touch

def reaction(data, start, end):
    """returns what the set up does"""
    reaction = "NaN"
    for i in range (start, end):
        event = str(data.iloc[i][4])#on se place sur la colonne 5
        if "Reward" in event:
            reaction = "Reward"
        elif "Timeout" in event :
           reaction = "Timeout"
    return reaction

def mouse_reaction(data, start, end):
    """"returns what the mouse does for each trial"""
    mouse = "No Lick"
    for i in range (start, end):
        event = str(data.iloc[i][5])#on se place sur la colonne 6
        if "Port1In" in event or "Port1Out"in event:
            mouse = "Lick"
    return mouse

def outcome(trial_type, mouse_reaction,setup):
    """returns the behavioural interpretation"""
    outcome = "Pourquoi ca apparait???"
    if trial_type == "Go" or trial_type == "Go-Touch":
        if mouse_reaction == "Lick" and setup == "Reward":
            outcome = "Hit"
        elif mouse_reaction == "Lick" and setup == "Timeout":
            outcome = "Go-Timeout"
        else:
            outcome = "Miss"
    elif trial_type == "Nogo" or trial_type == "Nogo-Touch":
        if mouse_reaction == "No Lick":
            outcome = "Correct Rejection"
        else :
            outcome = "False Alarm"
    return outcome

def performance(outcome):
    perf = 0
    if outcome == "Hit" or outcome == "Correct Rejection":
        perf = 1
    return perf

def ITI2(data, start, end):

    iti2 = 0

    for i in range(start, end):

        if str(data.iloc[i][4]) == "ITI2":

            debut_iti2 = data.iloc[i][2]
            fin_iti2 = data.iloc[i + 1][2]

            iti2 = fin_iti2 - debut_iti2

    return iti2

def choice(mouse_behaviour):
    choix = 0
    if mouse_behaviour == "Lick":
        choix = 1
    return choix

def licking_time(data, start, end):
    temps_lechage = 0
    for i in range(start, end):
        event = str(data.iloc[i][4])
        if event == "SocialTouch":
            next_event = str(data.iloc[i + 1][5])  # colonne F

            if next_event in ["Port1In", "Port1Out"]:

                temps_lechage = data.iloc[i + 1][2] - data.iloc[i][2]

            break
    return temps_lechage

def previous_trial(col):
    """informations du trial precedent"""
    col["Prev_trial_type"] = col["Trial_type"].shift(1)
    col["Prev_SetUp"] = col["Set up"].shift(1)
    col["Prev_Outcome"] = col["Outcome"].shift(1)
    col["Prev_ITI2"] = col["ITI2"].shift(1)
    col["Prev_Perf"] = col["Performance"].shift(1)
    col["Prev_Lick"] = col["Choice"].shift(1)
    return col

def trial_table(data_excel, data_json, lst_newtrial, lst_endtrial):
    """on va renvoyer un DataFrame de tous les trials"""
    all_trials= []
    for i in range(len(data_json)):
        trial_typ = trial_type(data_json,i)
        react = reaction(data_excel, lst_newtrial[i], lst_endtrial[i])
        mouse = mouse_reaction(data_excel, lst_newtrial[i], lst_endtrial[i])
        outcom = outcome(trial_typ,mouse,react)
        iti2 = ITI2(data_excel, lst_newtrial[i], lst_endtrial[i])
        touc = touch(data_excel,lst_newtrial[i],lst_endtrial[i])
        perf = performance(outcom)
        choix = choice(mouse)
        lick_time = licking_time(data_excel,lst_newtrial[i],lst_endtrial[i])

        all_trials.append({
            "Trial": i + 1,
            "Trial_type": trial_typ,
            "Set up": react,
            "Mouse beahviour" : mouse,
            "Outcome" : outcom,
            "ITI2": iti2,
            "Touch" : touc,
            "Performance" : perf,
            "Choice": choix,
            "Licking_time":lick_time
        })
    return pd.DataFrame(all_trials)