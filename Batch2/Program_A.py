from Tableaux_A import *
from Coefficients_A import *
from Coefficients_ITI2 import *
import os
import statsmodels.api as sm
from scipy.stats import pearsonr



racine = r"C:\Users\Asus\Stage 1A\SocialAnxiety_Batch2_testing"#chemin local

resultats = []

all_trials = []

for souris in os.listdir(racine):

    chemin_souris = os.path.join(racine, souris)

    if not os.path.isdir(chemin_souris):
        continue

    print(f"\nSouris : {souris}\n")

    for session in os.listdir(chemin_souris):

        chemin_session = os.path.join(chemin_souris, session)

        if not os.path.isdir(chemin_session):
            continue

        print(f"Session : {session}")#pour deboggage

        for fichier in os.listdir(chemin_session):

            if not fichier.endswith(".xls"):
                continue

            ma_path = os.path.join(chemin_session, fichier)

            print(f"Fichier : {fichier}\n")#pour deboggage

        #ma_path = r"C:\Users\Asus\Stage 1A\GoNoGo data ilona\training day1\20260128-103838.xls"

        my_data_xls, my_dirname = catch_info_xls(ma_path)

        my_data_json = catch_info_json(my_dirname)

        number_trial = len(my_data_json)

        liste_newtrial = search_nb_newtrial(my_data_xls, number_trial)

        liste_endtrial = search_nb_endtrial(my_data_xls, number_trial)

        df = trial_table(
        my_data_xls,
            my_data_json,
            liste_newtrial,
            liste_endtrial
        )
        df["Genotype"] = genotype_souris(souris)
        df["Session"] = session
        df["Mouse"] = souris
        df = previous_trial(df)
        all_trials.append(df)
        performance_touch = (df.groupby("Touch")["Performance"].mean())

        resultats.append({
            "Mouse": souris,
            "Session": session,
            "NoTouch": performance_touch.get("NoTouch"),
            "Touch": performance_touch.get("Touch")
        })

        #On cree les colonnes pour chaque type de trial possible
        df_go = df[df["Trial_type"] == "Go"]

        df_go_touch = df[df["Trial_type"] == "Go-Touch"]

        df_nogo = df[df["Trial_type"] == "Nogo"]

        df_nogo_touch = df[df["Trial_type"] == "Nogo-Touch"]

        
        #print(df.head())
        #on créé un tableau excel avec tout ça
        output_file = os.path.join(my_dirname,f"Trial_table_{session}.xlsx")
        df.to_excel(output_file, index=False)
        output_analysis = os.path.join(my_dirname,f"Analysis_by_trial_type_{session}.xlsx")
        colonnes_a_garder = [
            "Trial",
            "Trial_type",
            "Outcome",
            "ITI2",
            "Choice",
            "Performance",
            "Licking_time",
            "Prev_trial_type",
            "Prev_Outcome",
            "Prev_ITI2",
            "Prev_Perf",
            "Prev_Lick"
        ]

        df_go_export = df_go[colonnes_a_garder]

        df_go_touch_export = df_go_touch[colonnes_a_garder]

        df_nogo_export = df_nogo[colonnes_a_garder]

        df_nogo_touch_export = df_nogo_touch[colonnes_a_garder]

        output_coeffs_trials = os.path.join(
            racine,
            "Coefficients_Trials.xlsx"
        )
        with pd.ExcelWriter(output_analysis) as writer:

            df_go_export.to_excel(
                writer,
                sheet_name="Go",
                index=False
            )

            df_go_touch_export.to_excel(
                writer,
                sheet_name="GoTouch",
                index=False
            )

            df_nogo_export.to_excel(
                writer,
                sheet_name="NoGo",
                index=False
            )

            df_nogo_touch_export.to_excel(
                writer,
                sheet_name="NoGoTouch",
                index=False
            )
        
        


        

df_resultats = pd.DataFrame(resultats)

fichier_resume = os.path.join(
    racine,
    "Performance_Touch_vs_NoTouch.xlsx"
)

df_resultats.to_excel(
    fichier_resume,
    index=False
)

# Fusion de toutes les sessions

df_total = pd.concat(all_trials, ignore_index=True)
df_go_total = df_total[df_total["Trial_type"] == "Go"]
df_go_touch_total = df_total[df_total["Trial_type"] == "Go-Touch"]
df_nogo_total = df_total[df_total["Trial_type"] == "Nogo"]
df_nogo_touch_total = df_total[df_total["Trial_type"] == "Nogo-Touch"]


# WT
df_go_wt = df_go_total[df_go_total["Genotype"] == "WT"]
df_go_touch_wt = df_go_touch_total[df_go_touch_total["Genotype"] == "WT"]
df_nogo_wt = df_nogo_total[df_nogo_total["Genotype"] == "WT"]
df_nogo_touch_wt = df_nogo_touch_total[df_nogo_touch_total["Genotype"] == "WT"]

# SIS
df_go_sis = df_go_total[df_go_total["Genotype"] == "SIS"]
df_go_touch_sis = df_go_touch_total[df_go_touch_total["Genotype"] == "SIS"]
df_nogo_sis = df_nogo_total[df_nogo_total["Genotype"] == "SIS"]
df_nogo_touch_sis = df_nogo_touch_total[df_nogo_touch_total["Genotype"] == "SIS"]

# coeff_go_perf, model_go_perf = logistic_previous_outcome_trial_perf(df_go_total)

# coeff_go_touch_perf, model_go_touch_perf = logistic_previous_outcome_trial_perf(df_go_touch_total)

# coeff_nogo_perf, model_nogo_perf = logistic_previous_outcome_trial_perf(df_nogo_total)

# coeff_nogo_touch_perf, model_nogo_touch_perf = logistic_previous_outcome_trial_perf(df_nogo_touch_total)

# # output_coeffs_trials = os.path.join(
# #     racine,
# #     "Coefficients_Trials.xlsx"
# # )

# with pd.ExcelWriter(output_coeffs_trials) as writer:

#     coeff_go.to_excel(
#         writer,
#         sheet_name="Go",
#         index=False
#     )

#     coeff_go_touch.to_excel(
#         writer,
#         sheet_name="GoTouch",
#         index=False
#     )

#     coeff_nogo.to_excel(
#         writer,
#         sheet_name="NoGo",
#         index=False
#     )

#     coeff_nogo_touch.to_excel(
#         writer,
#         sheet_name="NoGoTouch",
#         index=False
#     )

    
output_wt_perf = os.path.join(
racine,
"Coefficients_WT_trials_perf.xlsx" )

#Performance : 0=echec, 1=success 
# WT
coeff_go_wt_perf, model_go_wt_perf = logistic_previous_outcome_trial_perf(df_go_wt)
coeff_go_touch_wt_perf, model_go_touch_wt_perf = logistic_previous_outcome_trial_perf(df_go_touch_wt)
coeff_nogo_wt_perf, model_nogo_wt_perf = logistic_previous_outcome_trial_perf(df_nogo_wt)
coeff_nogo_touch_wt_perf, model_nogo_touch_wt_perf = logistic_previous_outcome_trial_perf(df_nogo_touch_wt)

# SIS
coeff_go_sis_perf, model_go_sis_perf = logistic_previous_outcome_trial_perf(df_go_sis)
coeff_go_touch_sis_perf, model_go_touch_sis_perf = logistic_previous_outcome_trial_perf(df_go_touch_sis)
coeff_nogo_sis_perf, model_nogo_sis_perf = logistic_previous_outcome_trial_perf(df_nogo_sis)
coeff_nogo_touch_sis_perf, model_nogo_touch_sis_perf = logistic_previous_outcome_trial_perf(df_nogo_touch_sis)

#WT
with pd.ExcelWriter(output_wt_perf) as writer:

    coeff_go_wt_perf.to_excel(
        writer,
        sheet_name="Go",
        index=False
    )

    coeff_go_touch_wt_perf.to_excel(
        writer,
        sheet_name="GoTouch",
        index=False
    )

    coeff_nogo_wt_perf.to_excel(
        writer,
        sheet_name="NoGo",
        index=False
    )

    coeff_nogo_touch_wt_perf.to_excel(
        writer,
        sheet_name="NoGoTouch",
        index=False
    )

    output_sis_perf = os.path.join(
    racine,
    "Coefficients_SIS_trials_perf.xlsx"
)
    
#SIS
with pd.ExcelWriter(output_sis_perf) as writer:

    coeff_go_sis_perf.to_excel(
        writer,
        sheet_name="Go",
        index=False
    )

    coeff_go_touch_sis_perf.to_excel(
        writer,
        sheet_name="GoTouch",
        index=False
    )

    coeff_nogo_sis_perf.to_excel(
        writer,
        sheet_name="NoGo",
        index=False
    )

    coeff_nogo_touch_sis_perf.to_excel(
        writer,
        sheet_name="NoGoTouch",
        index=False
    )

#Choice : lick / nolick

# WT
coeff_go_wt_lick, model_go_wt_lick = logistic_previous_lick_choice(df_go_wt)
coeff_go_touch_wt_lick, model_go_touch_wt_lick = logistic_previous_lick_choice(df_go_touch_wt)
coeff_nogo_wt_lick, model_nogo_wt_lick = logistic_previous_lick_choice(df_nogo_wt)
coeff_nogo_touch_wt_lick, model_nogo_touch_wt_lick = logistic_previous_lick_choice(df_nogo_touch_wt)


# SIS
coeff_go_sis_lick, model_go_sis_lick = logistic_previous_lick_choice(df_go_sis)
coeff_go_touch_sis_lick, model_go_touch_sis_lick = logistic_previous_lick_choice(df_go_touch_sis)
coeff_nogo_sis_lick, model_nogo_sis_lick = logistic_previous_lick_choice(df_nogo_sis)
coeff_nogo_touch_sis_lick, model_nogo_touch_sis_lick = logistic_previous_lick_choice(df_nogo_touch_sis)

# # WT
# coeff_go_wt_choice, model_go_wt_choice = logistic_previous_outcome_trial_choice(df_go_wt)
# coeff_go_touch_wt_choice, model_go_touch_wt_choice = logistic_previous_outcome_trial_choice(df_go_touch_wt)
# coeff_nogo_wt_choice, model_nogo_wt_choice = logistic_previous_outcome_trial_choice(df_nogo_wt)
# coeff_nogo_touch_wt_choice, model_nogo_touch_wt_choice = logistic_previous_outcome_trial_choice(df_nogo_touch_wt)

# # SIS
# coeff_go_sis_choice, model_go_sis_choice = logistic_previous_outcome_trial_choice(df_go_sis)
# coeff_go_touch_sis_choice, model_go_touch_sis_choice = logistic_previous_outcome_trial_choice(df_go_touch_sis)
# coeff_nogo_sis_choice, model_nogo_sis_choice = logistic_previous_outcome_trial_choice(df_nogo_sis)
# coeff_nogo_touch_sis_choice, model_nogo_touch_sis_choice = logistic_previous_outcome_trial_choice(df_nogo_touch_sis)

#WT
output_wt_choice = os.path.join(
    racine,
    "Coefficients_WT_trials_choice.xlsx"
    )
with pd.ExcelWriter(output_wt_choice) as writer:

    coeff_go_wt_lick.to_excel(
        writer,
        sheet_name="Go",
        index=False
    )

    coeff_go_touch_wt_lick.to_excel(
        writer,
        sheet_name="GoTouch",
        index=False
    )

    coeff_nogo_wt_lick.to_excel(
        writer,
        sheet_name="NoGo",
        index=False
    )

    coeff_nogo_touch_wt_lick.to_excel(
        writer,
        sheet_name="NoGoTouch",
        index=False
    )

    output_sis_choice = os.path.join(
    racine,
    "Coefficients_SIS_trials_choice.xlsx"
)
    
#SIS
with pd.ExcelWriter(output_sis_choice) as writer:

    coeff_go_sis_lick.to_excel(
        writer,
        sheet_name="Go",
        index=False
    )

    coeff_go_touch_sis_lick.to_excel(
        writer,
        sheet_name="GoTouch",
        index=False
    )

    coeff_nogo_sis_lick.to_excel(
        writer,
        sheet_name="NoGo",
        index=False
    )

    coeff_nogo_touch_sis_lick.to_excel(
        writer,
        sheet_name="NoGoTouch",
        index=False
    )

#ITI2 
coeffi_go, modeli_go = logistic_previous_outcome_ITI2(df_go_total)
coeffi_go_touch, modeli_go_touch = logistic_previous_outcome_ITI2(df_go_touch_total)
coeffi_nogo, modeli_nogo = logistic_previous_outcome_ITI2(df_nogo_total)
coeffi_nogo_touch, modeli_nogo_touch = logistic_previous_outcome_ITI2(df_nogo_touch_total)

output_coeffs_iti2= os.path.join(
    racine,
    "Coefficients_ITI2.xlsx"
    )
with pd.ExcelWriter(output_coeffs_iti2) as writer:

    coeffi_go.to_excel(
        writer,
        sheet_name="Go",
        index=False
    )

    coeffi_go_touch.to_excel(
        writer,
        sheet_name="GoTouch",
        index=False
    )

    coeffi_nogo.to_excel(
        writer,
        sheet_name="NoGo",
        index=False
    )

    coeffi_nogo_touch.to_excel(
        writer,
        sheet_name="NoGoTouch",
        index=False
    )


#plotter le marginal effect du ITI2 avec les hit et miss seulement pour les trials Go-touch sans timeout
df_go_touch_hitmiss = df_go_touch_total[
    (df_go_touch_total["Set up"] != "Timeout") &
    (df_go_touch_total["Outcome"].isin(["Hit", "Miss"]))
].copy()

df_go_touch_hitmiss["Hit"] = (
    df_go_touch_hitmiss["Outcome"] == "Hit"
).astype(int)

# Séparation WT / SIS

df_go_touch_hitmiss_wt = df_go_touch_hitmiss[
    df_go_touch_hitmiss["Genotype"] == "WT"
]

df_go_touch_hitmiss_sis = df_go_touch_hitmiss[
    df_go_touch_hitmiss["Genotype"] == "SIS"
]

#corrélation iti2 Perf

import statsmodels.formula.api as smf
#toutes souris
model_iti2_gotouch = smf.logit(
    "Hit ~ ITI2",
    data=df_go_touch_hitmiss
).fit(disp=False)

df_corr = df_go_total.dropna(
    subset=["ITI2", "Performance"]
)

r_go, p_go = pearsonr(
    df_corr["ITI2"],
    df_corr["Performance"]
)

#print(r_go, p_go)

print("\n Corrélation Pearson ITI2 go")
print(f"r = {r_go:.3f}")
print(f"p = {p_go:.5f}")
print("\n")

# import matplotlib.pyplot as plt


# df_cr = df_nogo_touch_total[
#     df_nogo_touch_total["Outcome"] == "Correct Rejection"
# ]

# plt.plot(
#     df_cr["Trial"],
#     df_cr["ITI2"]
# )

# plt.xlabel("ITI2 (s)")
# plt.ylabel("Correct Rejection Rate")
# plt.title(
#     "Relation entre l'ITI2 et le taux de Correct Rejection\n"
#     "Trials Nogo-Touch - Toutes les souris et sessions confondues"
# )

# plt.show()

print("\nTables saved.\n")#pour deboggage

#pour le mixed model pour chaque souris

dfs_iti2, models_iti2_souris = build_all_iti2_models(df_total)








#Analyse statistique
# print("\nTableau de contingence : Outcome en fonction de previous Outcome\n")
# print(pd.crosstab(df["Prev_Outcome"], df["Outcome"]))
# hit_rates = (df.groupby("Prev_Outcome")["Outcome"].apply(lambda x: (x == "Hit").mean()))
# print("\nHit Rates\n")
# print(hit_rates)
# performance_touch = (df.groupby("Touch")["Performance"].mean())
# print("\nPerformance\n")
# print(performance_touch)
# print("\n")

from scipy.stats import pearsonr

# Première séance uniquement
df_first_session = df_go_touch_hitmiss[
    df_go_touch_hitmiss["Session"] == "session 1"
].copy()


# Hit rate moyen et ITI2 moyen par souris
df_mouse_first = (
    df_first_session
    .groupby("Mouse")
    .agg(
        HitRate=("Hit", "mean"),
        ITI2=("ITI2", "mean"),
        Genotype=("Genotype", "first")
    )
    .reset_index()
)

print(df_mouse_first)

# Toutes souris
r_all, p_all = pearsonr(
    df_mouse_first["ITI2"],
    df_mouse_first["HitRate"]
)

print("\nPremière séance - Toutes souris")
print(f"r = {r_all:.3f}")
print(f"p = {p_all:.5f}")


# WT
df_mouse_first_wt = df_mouse_first[
    df_mouse_first["Genotype"] == "WT"
]

r_wt, p_wt = pearsonr(
    df_mouse_first_wt["ITI2"],
    df_mouse_first_wt["HitRate"]
)
print("\nPremière séance - WT")
print(f"r = {r_wt:.3f}")
print(f"p = {p_wt:.5f}")


# SIS
df_mouse_first_sis = df_mouse_first[
    df_mouse_first["Genotype"] == "SIS"
]

r_sis, p_sis = pearsonr(
    df_mouse_first_sis["ITI2"],
    df_mouse_first_sis["HitRate"] )

print("\nPremière séance - SIS")
print(f"r = {r_sis:.3f}")
print(f"p = {p_sis:.5f}")

#Logostic mixed model ITI2
df_go_touch_hitmiss["Hit"] = (
    df_go_touch_hitmiss["Outcome"] == "Hit"
).astype(int)

coeffs_iti2_genotype, model_iti2_genotype = (
    logistic_iti2_genotype(
        df_go_touch_hitmiss
    )
)

output_glm = os.path.join(
    racine,
    "ITI2_Genotype_MixedModel.xlsx"
)

coeffs_iti2_genotype.to_excel(
    output_glm,
    index=True
)

#model
df_mouse = (
    df_go_touch_hitmiss
    .groupby("Mouse")
    .agg(
        ITI2=("ITI2", "mean"),
        HitRate=("Hit", "mean"),
        Genotype=("Genotype", "first")
    )
    .reset_index()
)

r, p = pearsonr(
    df_mouse["ITI2"],
    df_mouse["HitRate"]
)
print("model")
print(f"Pearson r = {r:.3f}")
print(f"p = {p:.4f}")
print("\n")