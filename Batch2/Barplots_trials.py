import matplotlib.pyplot as plt

import statsmodels.api as sm

from Program_A import *

import numpy as np

import os

DOSSIER_GRAPHS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Graphes",
    "Graphes_Batch2"
)

os.makedirs(DOSSIER_GRAPHS, exist_ok=True)

DOSSIER_ITI2_MIXED = os.path.join(
    DOSSIER_GRAPHS,
    "ITI2_Mixed_Models"
)

os.makedirs(DOSSIER_ITI2_MIXED,exist_ok=True)

DOSSIER_LICKTIME = os.path.join(
    DOSSIER_GRAPHS,
    "Licking_Time"
)

os.makedirs(
    DOSSIER_LICKTIME,
    exist_ok=True
)

def create_iti2_folders():

    for trial_type in [

        "Go",
        "Go-Touch",
        "Nogo",
        "Nogo-Touch"

    ]:

        os.makedirs(

            os.path.join(
                DOSSIER_ITI2_MIXED,
                trial_type
            ),

            exist_ok=True
        )

def plot_coeff(models, variable, titre, nom_fichier):

    plt.figure(figsize=(7,4))

    y = 0
    labels = []

    for name, model in models.items():

        if model is None:
            continue

        if variable not in model.params.index:
            continue

        coef = model.params[variable]

        conf = model.conf_int()

        lower = conf.loc[variable, 0]
        upper = conf.loc[variable, 1]

        plt.plot(
            [lower, upper],
            [y, y],
            color="gray",
            linewidth=2
        )

        plt.plot(
            coef,
            y,
            "o",
            color="black",
            markersize=8
        )

        plt.ylim(0, 1)

        labels.append(name)

        y += 1

    plt.axvline(
        0,
        color="black",
        linestyle="--"
    )

    plt.yticks(range(len(labels)), labels)

    plt.xlabel(f"Coefficient de {variable}")

    plt.title(titre)

    plt.tight_layout()

    plt.savefig(
    os.path.join(DOSSIER_GRAPHS, f"{nom_fichier}.png"),
    dpi=300,
    bbox_inches="tight"
)
    plt.show()
    #plt.close()


#WT-Perf-GoTouch-sans Timeout
models_wt_perf = {
    "Go": model_go_wt_perf,
    "Go-Touch": model_go_touch_wt_perf,
    "Nogo": model_nogo_wt_perf,
    "Nogo-Touch": model_nogo_touch_wt_perf
}
plot_coeff(models_wt_perf,"Performance","WT (n = 4)\nEffet de la performance à l'essai n-1 sur la performance à l'essai n","WT_PrevPerf_Performance")

#SIS-Perf
models_sis_perf = {
    "Go": model_go_sis_perf,
    "Go-Touch": model_go_touch_sis_perf,
    "Nogo": model_nogo_sis_perf,
    "Nogo-Touch": model_nogo_touch_sis_perf
}
plot_coeff(models_sis_perf,"Performance","Batch 2\nSIS\nEffet de Prev_Perf sur la performance\nToutes sessions et toutes SIS confondues","SIS_PrevPerf_Performance")

#WT-choice
models_wt_choice = {
    "Go": model_go_wt_lick,
    "Go-Touch": model_go_touch_wt_lick,
    "Nogo": model_nogo_wt_lick,
    "Nogo-Touch": model_nogo_touch_wt_lick
}
plot_coeff(models_wt_choice,"Choice","WT (n = 4)\nEffet du choix de léchage de l'essai n-1 sur le choix de l'essai n","WT_PrevLick_Choice")

#SIS-Choice
models_sis_choice = {
    "Go": model_go_sis_lick,
    "Go-Touch": model_go_touch_sis_lick,
    "Nogo": model_nogo_sis_lick,
    "Nogo-Touch": model_nogo_touch_sis_lick
}
plot_coeff(models_sis_choice,"Choice","Batch 2\nSIS\nEffet de Prev_Lick sur le choix (Lick / No Lick)\nToutes sessions et toutes SIS confondues", "SIS_PrevLick_Choice")

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

#tracé du marginal effect pour iti2
def plot_marginal_continu(model, variable, df, xlabel, ylabel, titre, nom_fichier):

    x = np.linspace(
        df[variable].min(),
        df[variable].max(),
        100
    )

    pred = model.predict(
        {variable: x}
    )

    plt.figure(figsize=(6,4))

    plt.plot(
        x,
        pred
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titre)

    plt.savefig(
    os.path.join(DOSSIER_GRAPHS, f"{nom_fichier}.png"),
    dpi=300,
    bbox_inches="tight"
)

    plt.draw()
    plt.close()

#Go-Touch - ITI2
plot_marginal_continu(
    model_iti2_gotouch,
    "ITI2",
    df_go_touch_hitmiss,
    "ITI2 (s)",
    "Probabilité de Hit",
    "Batch 2\nEffet marginal de ITI2\nTrials Go-Touch (Hit/Miss uniquement, sans Timeout)\nToutes souris et sessions confondues",
    "V-GoToutch-SansTimeout"
)

def plot_corr_iti2_hitrate(df, titre, nom_fichier):

    plt.figure(figsize=(6,4))

    plt.scatter(
        df["ITI2"],
        df["HitRate"]
    )

    if len(df) > 1:

        z = np.polyfit(
            df["ITI2"],
            df["HitRate"],
            1
        )

        p = np.poly1d(z)

        x = np.linspace(
            df["ITI2"].min(),
            df["ITI2"].max(),
            100
        )

        plt.plot(
            x,
            p(x)
        )

    plt.xlabel("ITI2 moyen (s)")
    plt.ylabel("Hit Rate")

    plt.title(titre)

    plt.savefig(
        os.path.join(
            DOSSIER_GRAPHS,
            f"{nom_fichier}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.draw()
    plt.close()

plot_corr_iti2_hitrate(
    df_mouse_first,
    "Batch 2\nPremière séance\nToutes souris",
    "ITI2_HitRate_All"
)

plot_corr_iti2_hitrate(
    df_mouse_first_wt,
    "Batch 2\nPremière séance\nWT",
    "ITI2_HitRate_WT"
)

plot_corr_iti2_hitrate(
    df_mouse_first_sis,
    "Batch 2\nPremière séance\nSIS",
    "ITI2_HitRate_SIS"
)

# iti2_range = np.linspace(
#     df_go_touch_hitmiss["ITI2"].min(),
#     df_go_touch_hitmiss["ITI2"].max(),
#     100
# )

# pred = model_iti2_gotouch.predict(
#     {"ITI2": iti2_range}
# )

# plt.figure(figsize=(6,4))

# plt.plot(
#     iti2_range,
#     pred
# )

# plt.xlabel("ITI2 (s)")

# plt.ylabel("Probabilité de Hit")

# plt.title(
#     "Effet marginal de ITI2\n"
#     "Trials Go-Touch (Hit/Miss uniquement, sans Timeout)\n"
#     "Toutes souris et sessions confondues"
# )


#tracé du marginal effect pour prev_perf et choice
def plot_marginal_binaire(model, variable, titre, ylabel, nom_fichier):

    pred0 = model.predict(
        {variable: [0]}
    )[0]

    pred1 = model.predict(
        {variable: [1]}
    )[0]

    plt.figure(figsize=(5,4))

    plt.plot(
        [0, 1],
        [pred0, pred1],
        marker="o"
    )

    plt.xticks(
        [0, 1],
        ["Échec précédent", "Succès précédent"]
    )

    plt.ylabel(ylabel)
    plt.title(titre)
    plt.savefig(
    os.path.join(DOSSIER_GRAPHS, f"{nom_fichier}.png"),
    dpi=300,
    bbox_inches="tight"
)

    plt.draw()
    plt.close()


#WT - Go - Prev_Perf
plot_marginal_binaire(
    model_go_wt_perf,
    "Performance",
    "Batch 2\nWT - Go\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "WT_Go_PrevPerf_Performance"
)

#WT - Go-Touch - Prev_Perf
plot_marginal_binaire(
    model_go_touch_wt_perf,
    "Performance",
    "Batch 2\nWT - GoTouch\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "WT_Go-Touch_PrevPerf_Performance"
)

#WT - Nogo - Prev_Perf
plot_marginal_binaire(
    model_nogo_wt_perf,
    "Performance",
    "Batch 2\nWT - Nogo\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "WT_Nogo_PrevPerf_Performance"
)

#WT - Nogo-Touch - Prev_Perf
plot_marginal_binaire(
    model_nogo_touch_wt_perf,
    "Performance",
    "Batch 2\nWT - NogoTouch\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "WT_Nogo-Touch_PrevPerf_Performance"
)

#SIS - Go - Prev_Perf
plot_marginal_binaire(
    model_go_sis_perf,
    "Performance",
    "Batch 2\nSIS - Go\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "SIS_Go_PrevPerf_Performance"
)

#SIS - Go-Touch - Prev_Perf
plot_marginal_binaire(
    model_go_touch_sis_perf,
    "Performance",
    "Batch 2\nSIS - GoTouch\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "SIS_Go-Touch_PrevPerf_Performance"
)

#SIS - Nogo - Prev_Perf
plot_marginal_binaire(
    model_nogo_sis_perf,
    "Performance",
    "Batch 2\nSIS - Nogo\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "SIS_Nogo_PrevPerf_Performance"
)

#SIS - Nogo-Touch - Prev_Perf
plot_marginal_binaire(
    model_nogo_touch_sis_perf,
    "Performance",
    "Batch 2\nSIS - NogoTouch\nEffet marginal de Prev_Perf sur la performance",
    "Probabilité de succès",
    "SIS_Nogo-Touch_PrevPerf_Performance"
)

#WT - Go - Choice
plot_marginal_binaire(
    model_go_wt_lick,
    "Choice",
    "Batch 2\nWT - Go\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "WT_Go_PrevLick_Choice"
)

#WT - Go-Touch - Choice
plot_marginal_binaire(
    model_go_touch_wt_lick,
    "Choice",
    "Batch 2\nWT - GoTouch\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "WT_Go-Touch_PrevLick_Choice"
)

#WT - Nogo - Choice
plot_marginal_binaire(
    model_nogo_wt_lick,
    "Choice",
    "Batch 2\nWT - Nogo\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "WT_Nogo_PrevLick_Choice"
)

#WT - Nogo-Touch - Choice
plot_marginal_binaire(
    model_nogo_touch_wt_lick,
    "Choice",
    "Batch 2\nWT - NogoTouch\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "WT_Nogo-Touch_PrevLick_Choice"
)

#SIS - Go - Choice
plot_marginal_binaire(
    model_go_sis_lick,
    "Choice",
    "Batch 2\nSIS - Go\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "SIS_Go_PrevLick_Choice"
)

#SIS - Go-Touch - Choice
plot_marginal_binaire(
    model_go_touch_sis_lick,
    "Choice",
    "Batch 2\nSIS - GoTouch\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "SIS_Go-Touch_PrevLick_Choice"
)

#SIS - Nogo - Choice
plot_marginal_binaire(
    model_nogo_sis_lick,
    "Choice",
    "Batch 2\nSIS - Nogo\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "SIS_Nogo_PrevLick_Choice"
)

#SIS - Nogo-Touch - Choice
plot_marginal_binaire(
    model_nogo_touch_sis_lick,
    "Choice",
    "Batch 2\nSIS - NogoTouch\nEffet marginal de Prev_Lick sur le choix",
    "Probabilité de Lick",
    "SIS_Nogo-Touch_PrevLick_Choice"
)
# pred0 = model_go_wt_perf.predict(
#     {"Prev_Perf": [0]}
# )[0]

# pred1 = model_go_wt_perf.predict(
#     {"Prev_Perf": [1]}
# )[0]

# plt.figure(figsize=(5,4))

# plt.plot(
#     [0, 1],
#     [pred0, pred1],
#     marker="o"
# )

# plt.xticks(
#     [0, 1],
#     ["Échec précédent", "Succès précédent"]
# )

# plt.ylabel("Probabilité de succès")

# plt.title(
#     "Effet marginal de Prev_Perf\n"
#     "WT - Go"
# )


#Logostic mixed model ITI2

def plot_iti2_genotype(model, df):

    iti2_range = np.linspace(
        df["ITI2"].min(),
        df["ITI2"].max(),
        100
    )

    souris_ref = df["Mouse"].iloc[0]

    pred_wt = model.predict(
        pd.DataFrame({
            "ITI2": iti2_range,
            "Genotype": "WT",
            "Mouse": souris_ref
        })
    )

    pred_sis = model.predict(
        pd.DataFrame({
            "ITI2": iti2_range,
            "Genotype": "SIS",
            "Mouse": souris_ref
        })
    )

    plt.figure(figsize=(6,4))

    plt.plot(
        iti2_range,
        pred_wt,
        label="WT"
    )

    plt.plot(
        iti2_range,
        pred_sis,
        label="SIS"
    )

    plt.xlabel("ITI2 (s)")
    plt.ylabel("Probabilité de Hit")

    plt.title(
        "Batch 2\nHit ~ ITI2 × Genotype"
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            DOSSIER_GRAPHS,
            "ITI2_Genotype_Interaction.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

plot_iti2_genotype(
    model_iti2_genotype,
    df_go_touch_hitmiss
)

def plot_une_souris(souris,model,df_souris,trial_type):
    iti2_range = np.linspace(df_souris["ITI2"].min(),df_souris["ITI2"].max(),100)

    X_pred = sm.add_constant(
        pd.DataFrame({
            "ITI2": iti2_range
        }),
        has_constant="add"
    )

    pred = model.predict(X_pred)

    beta0 = model.params["const"]
    beta1 = model.params["ITI2"]

    x_min = iti2_range.min()
    x_max = iti2_range.max()

    x_debut = (x_min + 1.5)/2
    x_fin = (3.5 + x_max)/2

    df_debut = df_souris[
        df_souris["ITI2"] <= 1.5
    ].copy()

    df_fin = df_souris[
        df_souris["ITI2"] >= 3.5
    ].copy()

    beta_debut = np.nan
    beta_fin = np.nan

    #calcul beta debut
    if (len(df_debut) >= 1 and df_debut["Performance"].nunique() == 2):
        X_debut = sm.add_constant(
            df_debut["ITI2"]
        )
        model_debut = sm.Logit(
            df_debut["Performance"],
            X_debut
        ).fit(disp=False)
        beta_debut = model_debut.params["ITI2"]

    #calcul beta fin
    if (len(df_fin) >= 1 and df_fin["Performance"].nunique() == 2):
        X_fin = sm.add_constant(
            df_fin["ITI2"]
        )

        model_fin = sm.Logit(
            df_fin["Performance"],
            X_fin
        ).fit(disp=False)

        beta_fin = model_fin.params["ITI2"]

    p_debut = model.predict(
        pd.DataFrame({
            "const":[1],
            "ITI2":[x_debut]
        })
    )[0]
    p_fin = model.predict(
        pd.DataFrame({
            "const":[1],
            "ITI2":[x_fin]
        })
    )[0]

    slope_debut = beta1 * p_debut * (1-p_debut)
    slope_fin = beta1 * p_fin * (1-p_fin)

    tangente_debut = p_debut + slope_debut*(iti2_range-x_debut)
    tangente_fin = p_fin + slope_fin*(iti2_range-x_fin)

   

    coef = model.params["ITI2"]

    p_milieu = pred[len(pred)//2]

    slope_bleue = coef * p_milieu * (1 - p_milieu)

    x0 = iti2_range[len(iti2_range)//2]
    y0 = p_milieu

    droite = y0 + slope_bleue * (iti2_range - x0)

    plt.figure(figsize=(6,4))

    #regression logistique
    plt.plot(
        iti2_range,
        pred,
        linewidth=3
    )

    #droite modelisee
    plt.plot(
        iti2_range,
        droite,
        "--",
        linewidth=2,
        label=f"cst + slope×ITI2 (tangente)"
    )

    plt.plot(
        iti2_range,
        tangente_debut,
        "--",
        label=f"Tangente début ({slope_debut:.3f})",
        color="green"
    )

    plt.plot(
        iti2_range,
        tangente_fin,
        "--",
        label=f"Tangente fin ({slope_fin:.3f})",
        color="red"
    )

    ratio = beta_debut / beta_fin

    plt.text(
        0.3,
        0.95,
        f"Slope milieu = {slope_bleue:.3f}",
        color="orange",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    plt.text(
        0.3,
        0.88,
        f"Slope début = {slope_debut:.3f}",
        color="green",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    plt.text(
        0.3,
        0.81,
        f"Slope fin = {slope_fin:.3f}",
        color="red",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )


    plt.text(
        0.98,
        0.18,
        f"βd = {beta_debut:.3f}",
        color="green",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    plt.text(
        0.98,
        0.11,
        f"βf = {beta_fin:.3f}",
        color="red",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    plt.text(
        0.98,
        0.04,
        f"βd / βf = {ratio:.3f}",
        color="blue",
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="bottom"
    )

    plt.ylim(0,1)

    plt.xlabel("ITI2 (s)")
    plt.ylabel("Probabilité prédite de succès")
    
    plt.title(
        f"{trial_type}\n"
        f"{souris}\nToutes sessions combinées"
    )

    

    print(f"{trial_type} | {souris} | pente courbe bleue = {slope_bleue}\n")
    # plt.text(
    # 0.05,
    # 0.95,
    # f"slope = {slope_bleue}",
    # transform=plt.gca().transAxes,
    # verticalalignment="top")

    plt.savefig(
        os.path.join(
            DOSSIER_ITI2_MIXED,
            trial_type,
            f"{souris}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_iti2_par_souris():

    create_iti2_folders()

    for trial_type, models in models_iti2_souris.items():

        df_trial = dfs_iti2[trial_type]

        for souris, model in models.items():

            df_souris = df_trial[
                df_trial["Mouse"] == souris
            ]

            plot_une_souris(
                souris,
                model,
                df_souris,
                trial_type
            )

plot_iti2_par_souris()

def plot_session1_iti2_logistique(df_total):

    dossier = os.path.join(
        DOSSIER_GRAPHS,
        "Session1_ITI2_Logistique"
    )

    os.makedirs(
        dossier,
        exist_ok=True
    )

    trial_types = [
        "Go",
        "Go-Touch",
        "Nogo",
        "Nogo-Touch"
    ]

    for trial_type in trial_types:

        dossier_trial = os.path.join(
            dossier,
            trial_type
        )

        os.makedirs(
            dossier_trial,
            exist_ok=True
        )

        df_trial = df_total[
            (df_total["Session"] == "session 1")
            &
            (df_total["Trial_type"] == trial_type)
        ].copy()

        for souris in df_trial["Mouse"].unique():

            df_souris = df_trial[
                df_trial["Mouse"] == souris
            ].copy()

            df_souris = df_souris[
                ["ITI2", "Performance"]
            ].dropna()

            if len(df_souris) < 10:
                continue

            if df_souris["Performance"].nunique() < 2:
                continue

            try:

                X = sm.add_constant(
                    df_souris["ITI2"]
                )

                model = sm.Logit(
                    df_souris["Performance"],
                    X
                ).fit(disp=False)

            except Exception:

                print(
                    f"Impossible de fitter : "
                    f"{trial_type} | {souris}"
                )

                continue

            iti2_range = np.linspace(
                df_souris["ITI2"].min(),
                df_souris["ITI2"].max(),
                200
            )

            X_pred = sm.add_constant(
                pd.DataFrame({
                    "ITI2": iti2_range
                }),
                has_constant="add"
            )

            pred = model.predict(
                X_pred
            )

            beta0 = model.params["const"]
            beta1 = model.params["ITI2"]

            # ==========================
            # Tangentes
            # ==========================

            x_debut = 0.75
            x_milieu = 2.5
            x_fin = 4.25

            p_debut = model.predict(
                pd.DataFrame({
                    "const": [1],
                    "ITI2": [x_debut]
                })
            )[0]

            p_milieu = model.predict(
                pd.DataFrame({
                    "const": [1],
                    "ITI2": [x_milieu]
                })
            )[0]

            p_fin = model.predict(
                pd.DataFrame({
                    "const": [1],
                    "ITI2": [x_fin]
                })
            )[0]

            slope_debut = (
                beta1
                * p_debut
                * (1 - p_debut)
            )

            slope_milieu = (
                beta1
                * p_milieu
                * (1 - p_milieu)
            )

            slope_fin = (
                beta1
                * p_fin
                * (1 - p_fin)
            )

            tangente_debut = (
                p_debut
                + slope_debut
                * (iti2_range - x_debut)
            )

            tangente_milieu = (
                p_milieu
                + slope_milieu
                * (iti2_range - x_milieu)
            )

            tangente_fin = (
                p_fin
                + slope_fin
                * (iti2_range - x_fin)
            )

            # ==========================
            # β locaux
            # ==========================

            df_debut = df_souris[
                df_souris["ITI2"] <= 1.5
            ]

            df_milieu = df_souris[
                (df_souris["ITI2"] > 1.5)
                &
                (df_souris["ITI2"] < 3.5)
            ]

            df_fin = df_souris[
                df_souris["ITI2"] >= 3.5
            ]

            beta_debut = np.nan
            beta_milieu = np.nan
            beta_fin = np.nan

            try:

                if (len(df_debut) >= 1 and df_debut["Performance"].nunique() == 2):
                    model_debut = sm.Logit(
                        df_debut["Performance"],
                        sm.add_constant(
                            df_debut["ITI2"]
                        )
                    ).fit(disp=False)

                    beta_debut = (
                        model_debut.params["ITI2"]
                    )

            except:
                pass

            try:

                if (
                    len(df_milieu) >= 1
                    and
                    df_milieu["Performance"].nunique() == 2
                ):

                    model_milieu = sm.Logit(
                        df_milieu["Performance"],
                        sm.add_constant(
                            df_milieu["ITI2"]
                        )
                    ).fit(disp=False)

                    beta_milieu = (
                        model_milieu.params["ITI2"]
                    )

            except:
                pass

            try:

                if (len(df_fin) >= 1 and df_fin["Performance"].nunique() == 2):
                    model_fin = sm.Logit(
                        df_fin["Performance"],
                        sm.add_constant(
                            df_fin["ITI2"]
                        )
                    ).fit(disp=False)

                    beta_fin = (
                        model_fin.params["ITI2"]
                    )

            except:
                pass
            if (
                not np.isnan(beta_debut)
                and
                not np.isnan(beta_fin)
                and
                beta_fin != 0
            ):
                ratio = beta_debut / beta_fin

            else:
                ratio = np.nan

            # ==========================
            # Graphe
            # ==========================

            plt.figure(figsize=(6,4))

            plt.plot(
                iti2_range,
                pred,
                color="blue",
                linewidth=3,
                label="Logistique"
            )

            plt.plot(
                iti2_range,
                tangente_debut,
                "--",
                color="green",
                linewidth=2
            )

            plt.plot(
                iti2_range,
                tangente_milieu,
                "--",
                color="orange",
                linewidth=2
            )

            plt.plot(
                iti2_range,
                tangente_fin,
                "--",
                color="red",
                linewidth=2
            )

            plt.text(
                0.98,
                0.35,
                f"Pente début = {slope_debut:.4f}",
                color="green",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.28,
                f"Pente milieu = {slope_milieu:.4f}",
                color="orange",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.21,
                f"Pente fin = {slope_fin:.4f}",
                color="red",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.14,
                f"βd = {beta_debut:.4f}",
                color="green",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.09,
                f"βm = {beta_milieu:.4f}",
                color="orange",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.05,
                f"βf = {beta_fin:.4f}",
                color="red",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.text(
                0.98,
                0.01,
                f"ratio = βd / βf = {ratio:.4f}",
                color="blue",
                transform=plt.gca().transAxes,
                ha="right"
            )

            plt.xlabel("ITI2 (s)")
            plt.ylabel(
                "Probabilité prédite de succès"
            )

            plt.ylim(
                -0.05,
                1.05
            )

            plt.title(
                f"{trial_type}\n"
                f"{souris}\n"
                f"Session 1"
            )

            plt.savefig(
                os.path.join(
                    dossier_trial,
                    f"{souris}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"{trial_type} | "
                f"{souris} | "
                f"β global = {beta1:.4f}"
            )

plot_session1_iti2_logistique(df_total)


################################Licking Time##########################################

def plot_licktime_moyennes(df_total):

    dossier = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Souris_Par_Session"
    )

    os.makedirs(
        dossier,
        exist_ok=True
    )

    trial_types = [
        "Go-Touch",
        "Nogo-Touch"
    ]

    for trial_type in trial_types:
       dossier_trial_base = os.path.join(
            dossier,
            trial_type
        )
       dossier_moyennes = os.path.join(
            dossier_trial_base,
            "Moyennes"
        )
       dossier_valeurs = os.path.join(
            dossier_trial_base,
            "Valeurs_brutes"
        )
       os.makedirs(dossier_valeurs, exist_ok=True)
       os.makedirs(dossier_moyennes, exist_ok=True)
       df_trial = df_total[
            (df_total["Trial_type"] == trial_type)
            &
            (df_total["Licking_time"].notna())
        ].copy()
       
       for souris in df_trial["Mouse"].unique():

            df_souris = df_trial[
                df_trial["Mouse"] == souris
            ].copy()

            if len(df_souris) == 0:
                continue

            df_plot = (
                df_souris[df_souris["Licking_time"] > 0]
                .groupby("Session")
                ["Licking_time"]
                .mean()
                .reset_index()
            )

            plt.figure(figsize=(6,4))

            plt.plot(
                df_plot["Session"],
                df_plot["Licking_time"],
                marker="o",
                linewidth=2
            )

            plt.ylabel(
                "Temps de léchage moyen"
            )

            plt.xlabel(
                "Session"
            )

            plt.title(
                f"{trial_type}\n{souris}\n(0 exclus)"
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    dossier_moyennes,
                    f"{souris}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()
plot_licktime_moyennes(df_total)

def plot_licking_time_par_socialtouch(df_total, trial_type):

    dossier_base = os.path.join(
        DOSSIER_GRAPHS,
        "Licking_Time",
        "Par_souris_Par_session",
        trial_type,
        "Valeurs_brutes"
    )

    os.makedirs(
        dossier_base,
        exist_ok=True
    )

    sessions = sorted(
        df_total["Session"].dropna().unique()
    )

    for session in sessions:

        dossier_session = os.path.join(
            dossier_base,
            str(session)
        )

        os.makedirs(
            dossier_session,
            exist_ok=True
        )

        df_session = df_total[
            (df_total["Session"] == session)
            &
            (df_total["Trial_type"] == trial_type)
            &
            (df_total["Licking_time"].notna())
        ].copy()

        for souris in df_session["Mouse"].unique():

            df_souris = df_session[
                df_session["Mouse"] == souris
            ].copy()

            if len(df_souris) == 0:
                continue

            df_souris = df_souris.reset_index(
                drop=True
            )

            # numéro du Social Touch
            df_souris["SocialTouch_Number"] = (
                np.arange(len(df_souris)) + 1
            )

            plt.figure(figsize=(7,4))

            plt.plot(
                df_souris["SocialTouch_Number"],
                df_souris["Licking_time"],
                marker="o"
            )

            plt.xlabel(
                "Numéro du Social Touch"
            )

            plt.ylabel(
                "Temps de léchage (s)"
            )

            plt.title(
                f"{trial_type}\n"
                f"{souris}\n"
                f"{session}"
            )

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    dossier_session,
                    f"{souris}.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

plot_licking_time_par_socialtouch(df_total,"Go-Touch")
plot_licking_time_par_socialtouch(df_total,"Nogo-Touch")

def plot_licktime_par_session_genotype(df_total):

    import numpy as np
    import os
    import matplotlib.pyplot as plt

    dossier_base = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Session_All_Mice_Genotype"
    )

    os.makedirs(dossier_base, exist_ok=True)

    sessions = sorted(df_total["Session"].dropna().unique())

    colors = {
        "WT": "blue",
        "SIS": "red"
    }

    for session in sessions:

        df_session = df_total[
            (df_total["Session"] == session)
            & (df_total["Licking_time"].notna())
        ].copy()

        plt.figure(figsize=(8,5))

        plotted = set()

        for mouse in df_session["Mouse"].unique():

            df_mouse = df_session[df_session["Mouse"] == mouse].copy()

            if df_mouse.empty:
                continue

            # IMPORTANT : définir l’ordre AVANT index
            df_mouse = df_mouse.sort_values("SocialTouch_Number") if "SocialTouch_Number" in df_mouse.columns else df_mouse.reset_index(drop=True)

            df_mouse["SocialTouch_Number"] = np.arange(1, len(df_mouse) + 1)

            genotype = str(df_mouse["Genotype"].iloc[0]).strip().upper()

            label = genotype if genotype not in plotted else None
            plotted.add(genotype)

            plt.plot(
                df_mouse["SocialTouch_Number"],
                df_mouse["Licking_time"],
                marker="o",
                linewidth=1,
                alpha=0.6,
                color=colors.get(genotype, "black"),
                label=label
            )

        # légende propre
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.title(f"Session {session}")
        plt.xlabel("Social Touch")
        plt.ylabel("Licking time (s)")

        plt.tight_layout()

        plt.savefig(
            os.path.join(dossier_base, f"session_{session}.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

#plot_licktime_par_session_genotype(df_total)

def plot_licktime_moyennes_toutes_souris(df_total):

    dossier = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Session_All_Mice_Genotype"
    )

    os.makedirs(dossier, exist_ok=True)

    colors = {
        "WT": "blue",
        "SIS": "red"
    }

    trial_types = [
        "Go-Touch",
        "Nogo-Touch"
    ]

    for trial_type in trial_types:

        plt.figure(figsize=(8,5))

        plotted = set()

        df_trial = df_total[
            (df_total["Trial_type"] == trial_type)
            &
            (df_total["Licking_time"] > 0)
        ].copy()

        for mouse in df_trial["Mouse"].unique():

            df_mouse = df_trial[
                df_trial["Mouse"] == mouse
            ].copy()

            if df_mouse.empty:
                continue

            # moyenne par session
            df_plot = (
                df_mouse
                .groupby("Session", as_index=False)
                ["Licking_time"]
                .mean()
            )

            genotype = str(
                df_mouse["Genotype"].iloc[0]
            ).strip().upper()

            label = genotype if genotype not in plotted else None
            plotted.add(genotype)

            plt.plot(
                df_plot["Session"],
                df_plot["Licking_time"],
                marker="o",
                color=colors.get(genotype, "black"),
                alpha=0.6,
                label=label
            )

        #plt.xlabel("Session")
        plt.ylabel("Temps de léchage moyen (0 exclus)")
        plt.title(trial_type)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                dossier,
                f"{trial_type}_moyennes_toutes_souris.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

plot_licktime_moyennes_toutes_souris(df_total)



def plot_licktime_par_session_genotype(df_total,trial_type):

    dossier_base = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Session_All_Mice_Genotype"
    )

    os.makedirs(
        dossier_base,
        exist_ok=True
    )

    sessions = sorted(
        df_total["Session"].dropna().unique()
    )

    colors = {
        "WT": "blue",
        "SIS": "red"
    }

    for session in sessions:

        df_session = df_total[
            (df_total["Session"] == session)
            &
            (df_total["Licking_time"].notna())
        ].copy()

        if trial_type is not None:

            df_session = df_session[
                df_session["Trial_type"] == trial_type
            ]

        plt.figure(figsize=(8,5))

        plotted = set()

        for mouse in df_session["Mouse"].unique():

            df_mouse = df_session[
                df_session["Mouse"] == mouse
            ].copy()

            if len(df_mouse) == 0:
                continue

            genotype = (
                str(
                    df_mouse["Genotype"].iloc[0]
                )
                .strip()
                .upper()
            )

            df_mouse = df_mouse.reset_index(
                drop=True
            )

            df_mouse["SocialTouch_Number"] = (
                np.arange(len(df_mouse)) + 1
            )

            label = (
                genotype
                if genotype not in plotted
                else None
            )

            plotted.add(genotype)

            plt.plot(
                df_mouse["SocialTouch_Number"],
                df_mouse["Licking_time"],
                marker="o",
                linewidth=1,
                alpha=0.5,
                color=colors.get(
                    genotype,
                    "black"
                ),
                label=label
            )

        plt.xlabel(
            "Social Touch"
        )

        plt.ylabel(
            "Licking time (s)"
        )

        if trial_type is None:

            titre = (
                f"Session {session}\n"
                f"Tous les Social Touch"
            )

            nom = f"{session}_ALL.png"

        else:

            titre = (
                f"Session {session}\n"
                f"{trial_type}"
            )

            nom = (
                f"{session}_{trial_type}.png"
            )

        plt.title(titre)

        handles, labels = (
            plt.gca()
            .get_legend_handles_labels()
        )

        by_label = dict(
            zip(labels, handles)
        )

        plt.legend(
            by_label.values(),
            by_label.keys()
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                dossier_base,
                nom
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

plot_licktime_par_session_genotype(df_total,"Go-Touch")
plot_licktime_par_session_genotype(df_total,"Nogo-Touch")

#séparés
def prepare_df(df_total, session=None, trial_type=None, only_touch=False):
    df = df_total.copy()
    if session is not None:
        df = df[df["Session"] == session]
    if trial_type is not None:
        df = df[df["Trial_type"] == trial_type]
    if only_touch:
        df = df[df["Trial_type"].isin(["Go-Touch", "Nogo-Touch"])]
    return df

def compute_mouse_session_mean(df):
    return df.groupby(["Mouse", "Genotype", "Session"], as_index=False)["Licking_time"].mean()

def plot_licktime_by_session_genotype(df_total, trial_type=None):
    dossier_base = os.path.join(DOSSIER_LICKTIME, "Graphes 200")
    os.makedirs(dossier_base, exist_ok=True)
    sessions = sorted(df_total["Session"].dropna().unique())

    colors = {"WT": "blue", "SIS": "red"}

    for session in sessions:

        df = prepare_df(df_total, session=session, trial_type=trial_type)

        if df.empty:
            continue

        plt.figure(figsize=(8,5))

        for mouse in df["Mouse"].unique():

            df_mouse = df[df["Mouse"] == mouse].copy()

            if df_mouse.empty:
                continue

            df_mouse = df_mouse.reset_index(drop=True)

            x = np.arange(1, len(df_mouse) + 1)

            genotype = str(df_mouse["Genotype"].iloc[0]).strip().upper()

            plt.plot(
                x,
                df_mouse["Licking_time"],
                marker="o",
                alpha=0.4,   # important pour voir superpositions
                linewidth=1,
                color=colors.get(genotype, "black")
            )

        plt.title(f"{session}" + (f"\n{trial_type}" if trial_type else ""))
        plt.xlabel("Trial index")
        plt.ylabel("Licking time")
        plt.tight_layout()

        name = f"{session}"
        if trial_type:
            name += f"_{trial_type}"

        plt.savefig(os.path.join(dossier_base, name + ".png"), dpi=300)
        plt.close()

def plot_session_means(df_total, trial_type=None):
    df = prepare_df(df_total, trial_type=trial_type, only_touch=False)
    df_mean = (
        df.groupby(["Mouse", "Genotype", "Session"], as_index=False)
        ["Licking_time"]
        .mean()
    )

    plt.figure(figsize=(8,5))
    colors = {"WT": "blue", "SIS": "red"}
    plotted = set()

    for mouse in df_mean["Mouse"].unique():
        df_mouse = df_mean[df_mean["Mouse"] == mouse]
        genotype = df_mouse["Genotype"].iloc[0]
        label = genotype if genotype not in plotted else None
        plotted.add(genotype)

        plt.plot(
            df_mouse["Session"],
            df_mouse["Licking_time"],
            marker="o",
            color=colors.get(genotype, "pink"),
            alpha=0.5,
            label=label
        )
    plt.xlabel("Session")
    plt.ylabel("Mean licking time per mouse")
    plt.title("Session evolution" + (f" | {trial_type}" if trial_type else ""))
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# plot_licktime_by_session_genotype(df_total, trial_type="Go-Touch")
# plot_licktime_by_session_genotype(df_total, trial_type="Nogo-Touch")
# plot_session_means(df_total, trial_type="Go-Touch")
# plot_session_means(df_total, trial_type="Nogo-Touch")

#Go-Touch + Nogo-Touch
df_touch = df_total[df_total["Trial_type"].isin(["Go-Touch", "Nogo-Touch"])].copy()

def plot_licktime_touch_regroupe(df_total):
    dossier = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Session_All_Mice_Genotype",
        "Touch_regroupe"
    )
    os.makedirs(dossier, exist_ok=True)

    df_touch = df_total[
        df_total["Trial_type"].isin(["Go-Touch", "Nogo-Touch"])
    ].copy()

    sessions = sorted(df_touch["Session"].dropna().unique())

    colors = {"WT": "blue", "SIS": "red"}

    for session in sessions:

        df_session = df_touch[df_touch["Session"] == session]

        if df_session.empty:
            continue

        plt.figure(figsize=(8,5))

        for mouse in df_session["Mouse"].unique():

            df_mouse = df_session[df_session["Mouse"] == mouse].copy()

            if df_mouse.empty:
                continue

            df_mouse = df_mouse.reset_index(drop=True)

            x = np.arange(1, len(df_mouse) + 1)

            genotype = str(df_mouse["Genotype"].iloc[0]).strip().upper()

            plt.plot(
                x,
                df_mouse["Licking_time"],
                marker="o",
                alpha=0.4,
                linewidth=1,
                color=colors.get(genotype, "black")
            )

        plt.title(f"{session}\nGo-Touch + Nogo-Touch")
        plt.xlabel("Trial index")
        plt.ylabel("Licking time")

        plt.tight_layout()

        plt.savefig(
            os.path.join(dossier, f"{session}_touch_regroupe.png"),
            dpi=300
        )
        plt.close()

plot_licktime_touch_regroupe(df_touch)

def plot_moyenne_licktime_touch_regroupe(df_total):

    dossier = os.path.join(
        DOSSIER_LICKTIME,
        "Par_Session_All_Mice_Genotype",
        "Touch_regroupe"
    )

    os.makedirs(dossier, exist_ok=True)

    df_touch = df_total[
        (df_total["Trial_type"].isin(["Go-Touch", "Nogo-Touch"]))
        &
        (df_total["Licking_time"] > 0)
    ].copy()

    colors = {
        "WT": "blue",
        "SIS": "red"
    }

    plt.figure(figsize=(8,5))

    plotted = set()

    for mouse in sorted(df_touch["Mouse"].unique()):

        df_mouse = df_touch[df_touch["Mouse"] == mouse].copy()
        df_mouse = df_mouse[df_mouse["Licking_time"] > 0].copy()

        df_plot = (
            df_mouse
            .groupby("Session", as_index=False)
            ["Licking_time"]
            .mean()
        )

        genotype = (
            df_mouse["Genotype"]
            .iloc[0]
        )

        label = genotype if genotype not in plotted else None
        plotted.add(genotype)

        plt.plot(
            df_plot["Session"],
            df_plot["Licking_time"],
            marker="o",
            alpha=0.6,
            color=colors.get(genotype, "black"),
            label=label
        )

    plt.title(
        "Mean licking time per mouse\n(Go-Touch + Nogo-Touch)"
    )

    plt.xlabel("Session")
    plt.ylabel("Mean licking time")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            dossier,
            "Mean_LickingTime_Touch_Regroupe.png"
        ),
        dpi=300
    )

    plt.close()

plot_moyenne_licktime_touch_regroupe(df_touch)

#print moyennes
def get_licktime_mean_table(df_total):

    df = df_total[
        (df_total["Trial_type"].isin(["Go-Touch", "Nogo-Touch"])) &
        (df_total["Licking_time"] > 0)
    ].copy()

    df_mean = (
        df
        .groupby(["Mouse", "Genotype", "Session"], as_index=False)["Licking_time"]
        .mean()
    )

    return df_mean.sort_values(["Mouse", "Session"])

print("\n ----------------------- INDIFFERENCIE ----------------------\n")
df_mean = get_licktime_mean_table(df_total)
print(df_mean)

print("\n ----------------------- GO-TOUCH ----------------------\n")
df_go_touch = df_total[df_total["Trial_type"] == "Go-Touch"]
df_mean_go_touch = get_licktime_mean_table(df_go_touch)
print(df_mean_go_touch)

print("\n ----------------------- NOGO-TOUCH ----------------------\n")
df_nogo_touch = df_total[df_total["Trial_type"] == "Nogo-Touch"]
df_mean_nogo_touch = get_licktime_mean_table(df_nogo_touch)
print(df_mean_nogo_touch)

import pandas as pd

with pd.ExcelWriter("licking_time_summary.xlsx") as writer:

    df_mean_go_touch.to_excel(writer, sheet_name="Go_Touch", index=False)
    df_mean_nogo_touch.to_excel(writer, sheet_name="Nogo_Touch", index=False)
    df_mean.to_excel(writer, sheet_name="Touch_regroupe", index=False)

plt.show()