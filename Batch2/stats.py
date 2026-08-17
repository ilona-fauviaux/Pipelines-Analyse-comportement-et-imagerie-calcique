from Program_A import *
from Barplots_trials import *
from scipy.integrate import trapezoid
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


#### Statistiques des coefficients du GLM principal
print("\n===============PERFORMANCE================\n")
print("\n GLM WT-GO \n")

glm_main = model_go_wt_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-GO_touch \n")

glm_main = model_go_touch_wt_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-NOGO \n")

glm_main = model_nogo_wt_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-NOGO_touch \n")

glm_main = model_nogo_touch_wt_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )
print("\n GLM SIS-GO \n")

glm_main = model_go_sis_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-GO_touch \n")

glm_main = model_go_touch_sis_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-NOGO \n")

glm_main = model_nogo_sis_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-NOGO_touch \n")

glm_main = model_nogo_touch_sis_perf

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} |\n "
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n===============LICK================\n")
print("\n GLM WT-GO \n")

glm_main = model_go_wt_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-GO_touch \n")

glm_main = model_go_touch_wt_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-NOGO \n")

glm_main = model_nogo_wt_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM WT-NOGO_touch \n")

glm_main = model_nogo_touch_wt_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )
print("\n GLM SIS-GO \n")

glm_main = model_go_sis_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-GO_touch \n")

glm_main = model_go_touch_sis_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-NOGO \n")

glm_main = model_nogo_sis_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

print("\n GLM SIS-NOGO_touch \n")

glm_main = model_nogo_touch_sis_lick

for var in glm_main.params.index:
    beta = glm_main.params[var]
    se = glm_main.bse[var]
    z = glm_main.tvalues[var]
    p = glm_main.pvalues[var]

    ci_low, ci_high = glm_main.conf_int().loc[var]

    print(
        f"{var:20s} | \n"
        f"β = {beta:8.4f} | \n"
        f"SE = {se:8.4f} | \n"
        f"z = {z:8.3f} | \n"
        f"p = {p:.4g} | \n"
        f"95% CI [{ci_low:.4f}, {ci_high:.4f}]\n"
    )

# def resume_modele_iti2(souris,trial_type,model,df_souris):
#     iti2_range = np.linspace(
#         df_souris["ITI2"].min(),
#         df_souris["ITI2"].max(),
#         100
#     )

#     X_pred = sm.add_constant(
#         pd.DataFrame({"ITI2": iti2_range}),
#         has_constant="add"
#     )

#     pred = model.predict(X_pred)

#     beta0 = model.params["const"]
#     beta1 = model.params["ITI2"]

#     p_milieu = pred[len(pred)//2]

#     return {
#         "Mouse": souris,
#         "Trial_type": trial_type,

#         "n_trials": len(df_souris),

#         "ITI2_min": df_souris["ITI2"].min(),
#         "ITI2_max": df_souris["ITI2"].max(),
#         "ITI2_mean": df_souris["ITI2"].mean(),
#         "ITI2_std": df_souris["ITI2"].std(),

#         "Success_rate": df_souris["Success"].mean(),

#         "beta0": beta0,
#         "beta1": beta1,

#         "OddsRatio": np.exp(beta1),

#         "p_value":
#             model.pvalues["ITI2"],

#         "Prob_min":
#             pred.min(),

#         "Prob_max":
#             pred.max(),

#         "Delta_prob":
#             pred.max() - pred.min(),

#         "Pente_max":
#             beta1 * 0.25,

#         "Pente_milieu":
#             beta1 * p_milieu * (1-p_milieu),

#         "AUC_courbe":
#             trapezoid(pred, iti2_range),

#         "Pseudo_R2":
#             getattr(model, "prsquared", np.nan)
#     }

# def export_resume_iti2():
    
#     lignes = []

#     for trial_type, models in models_iti2_souris.items():

#         df_trial = dfs_iti2[trial_type]

#         for souris, model in models.items():

#             df_souris = df_trial[
#                 df_trial["Mouse"] == souris
#             ]

#             lignes.append(
#                 resume_modele_iti2(
#                     souris,
#                     trial_type,
#                     model,
#                     df_souris
#                 )
#             )

#     df_resume = pd.DataFrame(lignes)

#     df_resume.to_excel(
#         os.path.join(
#             DOSSIER_ITI2_MIXED,
#             "Resume_Mixed_Models_ITI2.xlsx"
#         ),
#         index=False
#     )

#     return df_resume

# export_resume_iti2()

##################################Licking_time#######################################

def moyenne_slopes_par_trial():

    for trial_type, models in models_iti2_souris.items():

        slopes = []

        df_trial = dfs_iti2[trial_type]

        for souris, model in models.items():

            df_souris = df_trial[
                df_trial["Mouse"] == souris
            ]

            iti2_range = np.linspace(
                df_souris["ITI2"].min(),
                df_souris["ITI2"].max(),
                100
            )

            X_pred = sm.add_constant(
                pd.DataFrame({"ITI2": iti2_range}),
                has_constant="add"
            )

            pred = model.predict(X_pred)

            beta1 = model.params["ITI2"]

            p_milieu = pred[len(pred)//2]

            slope_bleue = (
                beta1
                * p_milieu
                * (1 - p_milieu)
            )

            slopes.append(slope_bleue)

        if len(slopes) > 0:
            print(f"\n-----------------------{trial_type}----------------------\n")
            print(f"Nb souris = {len(slopes)}\n")
            print(f"Slope moyenne = {np.mean(slopes):.4f}\n")
            print(f"Slope médiane = {np.median(slopes):.4f}\n")
            print(f"Slope SD = {np.std(slopes):.4f}\n")
            print(f"Slope min = {np.min(slopes):.4f}\n")
            print(f"Slope max = {np.max(slopes):.4f}\n")

moyenne_slopes_par_trial()

def stats_slopes_par_genotype():

    for trial_type, models in models_iti2_souris.items():

        slopes_wt = []
        slopes_sis = []

        df_trial = dfs_iti2[trial_type]

        for souris, model in models.items():

            df_souris = df_trial[
                df_trial["Mouse"] == souris
            ]

            iti2_range = np.linspace(
                df_souris["ITI2"].min(),
                df_souris["ITI2"].max(),
                100
            )

            X_pred = sm.add_constant(
                pd.DataFrame({"ITI2": iti2_range}),
                has_constant="add"
            )

            pred = model.predict(X_pred)

            beta1 = model.params["ITI2"]

            p_milieu = pred[len(pred)//2]

            slope = beta1 * p_milieu * (1 - p_milieu)

            genotype = df_souris["Genotype"].iloc[0]

            if genotype == "WT":
                slopes_wt.append(slope)

            elif genotype == "SIS":
                slopes_sis.append(slope)

        print(f"\n-----------------------{trial_type}----------------------------\n")

        if len(slopes_wt) > 0:

            print("\n-------------WT---------------\n")
            print(f"Nb souris = {len(slopes_wt)}")
            print(f"Slope moyenne = {np.mean(slopes_wt):.4f}")
            print(f"Slope médiane = {np.median(slopes_wt):.4f}")
            print(f"Slope SD = {np.std(slopes_wt):.4f}")
            print(f"Slope min = {np.min(slopes_wt):.4f}")
            print(f"Slope max = {np.max(slopes_wt):.4f}")

        if len(slopes_sis) > 0:

            print("\n---------------SIS---------------\n")
            print(f"Nb souris = {len(slopes_sis)}")
            print(f"Slope moyenne = {np.mean(slopes_sis):.4f}")
            print(f"Slope médiane = {np.median(slopes_sis):.4f}")
            print(f"Slope SD = {np.std(slopes_sis):.4f}")
            print(f"Slope min = {np.min(slopes_sis):.4f}")
            print(f"Slope max = {np.max(slopes_sis):.4f}")

stats_slopes_par_genotype()

def calcul_moyennes_par_souris_1(df_total, session):

    df_test = df_total[
        (df_total["Session"] == session)
        &
        (df_total["Trial_type"].isin(["Go-Touch","Nogo-Touch"]))
        &
        (df_total["Licking_time"] > 0)
    ].copy()

    df_moy = (
        df_test
        .groupby(["Mouse", "Genotype"], as_index=False)
        ["Licking_time"]
        .mean()
    )

    return df_moy

def comparaison_WT_SIS(df_moy):

    wt = df_moy.loc[
        df_moy["Genotype"] == "WT",
        "Licking_time"
    ]

    sis = df_moy.loc[
        df_moy["Genotype"] == "SIS",
        "Licking_time"
    ]

    if len(wt) < 3 or len(sis) < 3:

        print("Pas assez de souris pour faire un test.")

        return {
            "test": "Impossible",
            "p_value": None
        }

    shapiro_wt = shapiro(wt)
    shapiro_sis = shapiro(sis)

    print("\n----- Test de normalité -----")

    print(
        f"WT : p = {shapiro_wt.pvalue:.4f}"
    )

    if shapiro_wt.pvalue > 0.05:
        print("WT suit une loi normale")
    else:
        print("WT ne suit PAS une loi normale")

    print(
        f"SIS : p = {shapiro_sis.pvalue:.4f}"
    )

    if shapiro_sis.pvalue > 0.05:
        print("SIS suit une loi normale")
    else:
        print("SIS ne suit PAS une loi normale")

    if (
        shapiro_wt.pvalue > 0.05
        and
        shapiro_sis.pvalue > 0.05
    ):

        print("\n=> Utilisation d'un t-test")

        stat, p = ttest_ind(
            wt,
            sis,
            equal_var=False
        )

        test = "t-test"

    else:

        print(
            "\n=> Utilisation d'un test de Mann-Whitney (Wilcoxon rang-sum)"
        )

        stat, p = mannwhitneyu(
            wt,
            sis,
            alternative="two-sided"
        )

        test = "Wilcoxon"

    print(
        f"p-value = {p:.4f}"
    )

    diff = ""
    if p < 0.05:
        diff = "significatif"
        print("Difference significative entre WT et SIS")
    else:
        diff = "pas significatif"
        print("Pas de difference significative entre WT et SIS")

    return {
        "test": test,
        "stat": stat,
        "p_value": p,
        "analyse" : diff
    }

def analyse_WT_SIS_par_sessions_1(df_total):

    sessions = sorted(df_total["Session"].dropna().unique())

    results = []

    for session in sessions:

        print("\n----------------------------------")
        print(f"SESSION : {session}")
        print("--------------------------------\n")

        # 1) moyennes par souris
        df_moy = calcul_moyennes_par_souris_1(df_total, session)

        # 2) comparaison WT vs SIS
        res = comparaison_WT_SIS(df_moy)

        # 3) ajouter session au résultat
        res["session"] = session

        results.append(res)

    df_resultats = pd.DataFrame(results)

    df_resultats = df_resultats[
        ["session", "test", "p_value", "analyse"]
    ]

    return df_resultats



#Différenciation des Go-Touch et Nogo-Touch
def calcul_moyennes_par_souris_et_trial(df_total, session):

    df_test = df_total[
        (df_total["Session"] == session)
        &
        (df_total["Trial_type"].isin(["Go-Touch","Nogo-Touch"]))
        &
        (df_total["Licking_time"] > 0)
    ].copy()

    df_moy = (
        df_test
        .groupby(["Mouse", "Genotype", "Trial_type"], as_index=False)
        ["Licking_time"]
        .mean()
    )

    return df_moy

def analyse_WT_SIS_par_sessions_et_trial(df_total):

    sessions = sorted(df_total["Session"].dropna().unique())
    trial_types = ["Go-Touch", "Nogo-Touch"]

    results = []

    for session in sessions:

        print("\n----------------------------------")
        print(f"SESSION : {session}")
        print("------------------------------------\n")

        df_moy = calcul_moyennes_par_souris_et_trial(df_total, session)

        for trial in trial_types:

            print("\n----------------------------------")
            print(f"TRIAL TYPE : {trial}")
            print("----------------------------------\n")

            df_trial = df_moy[df_moy["Trial_type"] == trial]

            res = comparaison_WT_SIS(df_trial)

            res["session"] = session
            res["trial_type"] = trial

            results.append(res)

    df_resultats = pd.DataFrame(results)

    df_resultats = df_resultats[
        ["session", "trial_type", "test", "p_value", "analyse"]
    ]

    return df_resultats

resultats = analyse_WT_SIS_par_sessions_et_trial(df_total)
resultats_1 = analyse_WT_SIS_par_sessions_1(df_total)
print("\n--------------- Moyennes !! ---------------------\n")
print(resultats)
print("\n")
print(resultats_1)

# def comparaison_WT_SIS_toutes_sessions(df_total):
 
#     sessions = sorted(
#         df_total["Session"].dropna().unique()
#     )

#     resultats = []

#     for session in sessions:

#         print(f"\n------------------Session : {session}-----------------------------\n")

#         df_moy = calcul_moyennes_par_souris(
#             df_total,
#             session
#         )

#         resultat = comparaison_WT_SIS(
#             df_moy
#         )

#         resultat["Session"] = session

#         resultats.append(
#             resultat
#         )

#     return pd.DataFrame(
#         resultats
#     )

# resultats = comparaison_WT_SIS_toutes_sessions(df_total)
# print(resultats)

# def comparaison_trials_WT_SIS_condition(df_total, session, trial_type):

#     df_test = df_total[
#         (df_total["Session"] == session)
#         &
#         (df_total["Trial_type"] == trial_type)
#         &
#         (df_total["Licking_time"] > 0)
#     ].copy()

#     wt = df_test[df_test["Genotype"] == "WT"]["Licking_time"]
#     sis = df_test[df_test["Genotype"] == "SIS"]["Licking_time"]

#     print("\n--------------------------------")
#     print(f"Session : {session} | {trial_type}")
#     print("----------------------------------\n")

#     print(f"WT trials  : {len(wt)}")
#     print(f"SIS trials : {len(sis)}")

#     if len(wt) < 3 or len(sis) < 3:
#         print("Pas assez de données")
#         return None

#     # normalité (sécurité sur échantillon)
#     wt_sample = wt.sample(min(len(wt), 500), random_state=0)
#     sis_sample = sis.sample(min(len(sis), 500), random_state=0)

#     sh_wt = shapiro(wt_sample)
#     sh_sis = shapiro(sis_sample)

#     print("\nNormalité :")
#     print(f"WT  p = {sh_wt.pvalue:.4e}")
#     print(f"SIS p = {sh_sis.pvalue:.4e}")

#     if sh_wt.pvalue > 0.05 and sh_sis.pvalue > 0.05:

#         print("/// Test utilisé : t-test ///")

#         stat, p = ttest_ind(wt, sis, equal_var=False)

#         test = "t-test"

#     else:

#         print("/// Test utilisé : Wilcoxon ///")

#         stat, p = mannwhitneyu(wt, sis, alternative="two-sided")

#         test = "Wilcoxon"

#     print(f"p-value = {p:.4e}")
    
#     diff = ""
#     if p < 0.05:
#         diff = "Différence significative"
#         print("\n////// Différence significative //////\n")
#     else:
#         diff = "Pas de différence significative"
#         print("\n////// Pas de différence significative //////\n")

#     return {
#         "session": session,
#         "trial_type": trial_type,
#         "test": test,
#         "stat": stat,
#         "p_value": p,
#         "analyse": diff
#     }

# def analyse_trials_WT_SIS(df_total):

#     sessions = sorted(df_total["Session"].dropna().unique())
#     trial_types = ["Go-Touch", "Nogo-Touch"]

#     results = []

#     for session in sessions:
#         for trial_type in trial_types:

#             res = comparaison_trials_WT_SIS_condition(
#                 df_total,
#                 session,
#                 trial_type
#             )

#             if res is not None:
#                 results.append(res)

#     for trial_type in trial_types:

#         df_test = df_total[
#             (df_total["Trial_type"] == trial_type)
#             &
#             (df_total["Licking_time"] > 0)
#         ].copy()

#         wt = df_test[df_test["Genotype"] == "WT"]["Licking_time"]
#         sis = df_test[df_test["Genotype"] == "SIS"]["Licking_time"]

#         if len(wt) < 3 or len(sis) < 3:
#             continue

#         # sécurité Shapiro
#         wt_sample = wt.sample(min(len(wt), 500), random_state=0)
#         sis_sample = sis.sample(min(len(sis), 500), random_state=0)

#         sh_wt = shapiro(wt_sample)
#         sh_sis = shapiro(sis_sample)

#         if sh_wt.pvalue > 0.05 and sh_sis.pvalue > 0.05:
#             stat, p = ttest_ind(wt, sis, equal_var=False)
#             test = "t-test"
#         else:
#             stat, p = mannwhitneyu(wt, sis, alternative="two-sided")
#             test = "Wilcoxon"
        
#         diff = ""
#         if p < 0.05:
#             diff = "Différence significative"
#             print("\n////// Différence significative //////\n")
#         else:
#             diff = "Pas de différence significative"
#             print("\n////// Pas de différence significative //////\n")

#         results.append({
#             "session": "toutes sessions confondues",
#             "trial_type": trial_type,
#             "test": test,
#             "p_value": p,
#             "analyse" : diff
#         })

#     return pd.DataFrame(results)

# resultats = analyse_trials_WT_SIS(df_total)
# resultats = resultats.drop(columns=["stat"]).reset_index(drop=True)

# print(resultats)

def calcul_moyenne_par_souris(df_total):
    moyennes = (
        df_total[df_total["Licking_time"] > 0]
        .groupby("Mouse", as_index=False)
        ["Licking_time"]
        .mean()
    )
    moyennes = moyennes.rename(columns={"Licking_time": "Mean_Mouse"})
    return moyennes

def normaliser_licking_par_souris(df_total):
    moyennes = calcul_moyenne_par_souris(df_total)
    df_norm = df_total.merge(
        moyennes,
        on="Mouse",
        how="left"
    )
    df_norm["Licking_time_norm"] = df_norm["Licking_time"]/df_norm["Mean_Mouse"]
    return df_norm

def comparaison_WT_SIS_normalise(df_total,session,trial_type):

    df_norm = normaliser_licking_par_souris(df_total)

    df_test = df_norm[
        (df_norm["Session"] == session)
        &
        (df_norm["Trial_type"].isin(["Go-Touch","Nogo-Touch"]))
        &
        (df_norm["Licking_time"] > 0)
    ].copy()

    wt = df_test[
        df_test["Genotype"] == "WT"
    ]["Licking_time_norm"]

    sis = df_test[
        df_test["Genotype"] == "SIS"
    ]["Licking_time_norm"]

    if len(wt) < 3 or len(sis) < 3:
        return None

    sh_wt = shapiro(wt.sample(min(len(wt), 500), random_state=0))

    sh_sis = shapiro(sis.sample(min(len(sis), 500), random_state=0))

    if (sh_wt.pvalue > 0.05 and sh_sis.pvalue > 0.05):
        stat, p = ttest_ind(wt,sis,equal_var=False)
        test = "t-test"

    else:
        stat, p = mannwhitneyu(wt,sis,alternative="two-sided")
        test = "Wilcoxon"

    analyse = ""
    if p < 0.05:
        analyse = "Différence significative"
        
    else :
        analyse = "Pas de différence significative"

    return {
        "session": session,
        "trial_type": trial_type,
        "test": test,
        "p_value": p,
        "analyse": analyse
    }


def analyse_WT_SIS_normalise(df_total):

    sessions = sorted(df_total["Session"].dropna().unique())
    trial_types = ["Go-Touch", "Nogo-Touch"]

    results = []

    for session in sessions:
        for trial_type in trial_types:

            res = comparaison_WT_SIS_normalise(
                df_total,
                session,
                trial_type
            )

            if res is not None:
                results.append(res)

    return pd.DataFrame(results)

print("\n-------------------------- Souris normalisees !!! --------------------\n")
souris_normalisees = analyse_WT_SIS_normalise(df_total)
print(souris_normalisees)

def calcul_moyenne_par_souris_social_touch(df_total):
    moyennes = (
        df_total[df_total["Licking_time"] > 0]
        .groupby("Mouse", as_index=False)
        ["Licking_time"]
        .mean()
    )
    moyennes = moyennes.rename(
        columns={"Licking_time": "Mean_Mouse"}
    )
    return moyennes

def normaliser_par_souris_social_touch(df_total):
    moyennes = calcul_moyenne_par_souris_social_touch(df_total)
    df_norm = df_total.merge(
        moyennes,
        on="Mouse",
        how="left"
    )
    df_norm["Licking_time_norm"] = df_norm["Licking_time"]/df_norm["Mean_Mouse"]
    return df_norm

def comparaison_WT_SIS_normalise_social_touch(df_total,session):
    df_norm = normaliser_par_souris_social_touch(df_total)
    df_test = df_norm[
        (df_norm["Session"] == session)
        &
        (df_norm["Trial_type"].isin(["Go-Touch","Nogo-Touch"]))
        &
        (df_norm["Licking_time"] > 0)
    ].copy()

    wt = df_test[df_test["Genotype"] == "WT"]["Licking_time_norm"]

    sis = df_test[df_test["Genotype"] == "SIS"]["Licking_time_norm"]

    if len(wt) < 3 or len(sis) < 3:
        return None
    sh_wt = shapiro(wt.sample(min(len(wt), 500), random_state=0))
    sh_sis = shapiro(sis.sample(min(len(sis), 500), random_state=0))
    if (sh_wt.pvalue > 0.05 and sh_sis.pvalue > 0.05):
        stat, p = ttest_ind(wt,sis,equal_var=False)
        test = "t-test"
    else:
        stat, p = mannwhitneyu(wt,sis,alternative="two-sided")
        test = "Wilcoxon"

    analyse = ""
    if p < 0.05 :
        analyse = "Difference significative"
    else :
        analyse = "pas significatif"

    return {
        "session": session,
        "test": test,
        "p_value": p,
        "analyse": analyse
    }

def analyse_WT_SIS_normalise_social_touch(df_total):
    sessions = sorted(df_total["Session"].dropna().unique())
    results = []
    for session in sessions:

        res = comparaison_WT_SIS_normalise_social_touch(df_total,session)
        if res is not None:
            results.append(res)
    return pd.DataFrame(results)

souris_normalisees_sessions = analyse_WT_SIS_normalise_social_touch(df_total)
print("\n")
print(souris_normalisees_sessions)
print("\n")

# def prepare_df(df_total, session=None, trial_type=None, exclude_zero=True):
#     df = df_total.copy()
#     df["Trial_type"] = df["Trial_type"].astype(str).str.strip()
#     df = df[df["Trial_type"].isin(["Go-Touch", "Nogo-Touch"])]#suppression Go / Nogo "tout court"
#     if session is not None:
#         df = df[df["Session"] == session]
#     if trial_type is not None:
#         df = df[df["Trial_type"] == trial_type]
#     if exclude_zero:
#         df = df[df["Licking_time"] > 0]
#     return df

# def table_moyennes_session(df_total):
#     df = prepare_df(df_total)
#     return (
#         df.groupby(["Session", "Mouse", "Genotype"], as_index=False)
#         ["Licking_time"]
#         .mean()
#         .rename(columns={"Licking_time": "mean_licking"})
#     )

# def table_moyennes_session_trial(df_total):
#     df = prepare_df(df_total)

#     return (
#         df.groupby(
#             ["Session", "Trial_type", "Mouse", "Genotype"],
#             as_index=False
#         )["Licking_time"]
#         .mean()
#         .rename(columns={"Licking_time": "mean_licking"})
#     )

# def table_trials_session(df_total):
#     return prepare_df(df_total)

# def table_trials_session_trial(df_total):
#     return prepare_df(df_total)

# def compute_stats(df):
#     wt = df[df["Genotype"] == "WT"]["Licking_time"]
#     sis = df[df["Genotype"] == "SIS"]["Licking_time"]

#     if len(wt) < 3 or len(sis) < 3:
#         return None

#     p_wt = shapiro(wt.sample(min(len(wt), 500))).pvalue
#     p_sis = shapiro(sis.sample(min(len(sis), 500))).pvalue

#     if p_wt > 0.05 and p_sis > 0.05:
#         stat, p = ttest_ind(wt, sis, equal_var=False)
#         test = "t-test"
#     else:
#         stat, p = mannwhitneyu(wt, sis, alternative="two-sided")
#         test = "Wilcoxon"

#     return {
#         "test": test,
#         "p_value": p,
#         "analyse": "significatif" if p < 0.05 else "ns"
#     }


# resultats1 = analyse_WT_SIS_par_sessions_1(df_touch)

# print("\n===== TABLEAU 1 =====\n")
# print(resultats1)

# resultats2 = analyse_WT_SIS_par_sessions_et_trial(df_touch)

# print("\n===== TABLEAU 2 =====\n")
# print(resultats2)

# resultats3 = analyse_trials_WT_SIS(df_touch)

# print("\n===== TABLEAU 3 =====\n")
# print(resultats3)
# print("\n==================== TABLE 4 : Trials bruts + Go/Nogo ====================\n")
# table4 = compute_stats(df_touch)
# print(table4)