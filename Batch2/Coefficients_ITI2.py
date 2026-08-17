import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

# def previous_trial(col):
#     """informations du trial precedent"""
#     col["Prev_trial_type"] = col["Trial_type"].shift(1)
#     col["Prev_SetUp"] = col["Set up"].shift(1)
#     col["Prev_Outcome"] = col["Outcome"].shift(1)
#     col["Prev_ITI2"] = col["ITI2"].shift(1)
#     col["Prev_Perf"] = col["Performance"].shift(1)
#     return col

def logistic_previous_outcome_ITI2(df):

    df = df.dropna(subset=["ITI2", "Performance"])

    model = None

    try:

        model = smf.logit(
            "Performance ~ ITI2",
            data=df
        ).fit(disp=False)

        coeffs = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "p_value": model.pvalues.values
        })

    except Exception as e:

        coeffs = pd.DataFrame({
            "Variable": ["ERROR"],
            "Coefficient": [np.nan],
            "p_value": [np.nan],
            "Message": [str(e)]
        })

    return coeffs, model

def logistic_iti2_genotype(df):

    df_model = df.copy()

    df_model = df_model.dropna(
        subset=["ITI2", "Hit", "Genotype", "Mouse"]
    )

    model = smf.logit(
        "Hit ~ ITI2 * Genotype + C(Mouse)",
        data=df_model
    ).fit()

    coeffs = pd.DataFrame({
        "Coefficient": model.params,
        "p-value": model.pvalues,
        "OddsRatio": np.exp(model.params)
    })

    return coeffs, model

#mixed model
def prepare_iti2_dataframe(df, trial_type):

    df_trial = df[
        df["Trial_type"] == trial_type
    ].copy()

    if trial_type in ["Go", "Go-Touch"]:

        df_trial = df_trial[
            df_trial["Outcome"].isin(
                ["Hit", "Miss"]
            )
        ]

        df_trial["Success"] = (
            df_trial["Outcome"] == "Hit"
        ).astype(int)

    else:

        df_trial["Success"] = (
            df_trial["Outcome"] == "Correct Rejection"
        ).astype(int)

    return df_trial

def fit_models_par_souris(df_trial):

    models = {}

    for souris in df_trial["Mouse"].unique():

        df_souris = df_trial[
            df_trial["Mouse"] == souris
        ].copy()


        try:

            X = sm.add_constant(
                df_souris["ITI2"]
            )

            y = df_souris["Success"]

            model = sm.GLM(
                y,
                X,
                family=sm.families.Binomial()
            ).fit()

            models[souris] = model

        except:
            continue

    return models

def build_all_iti2_models(df_total):

    dfs_trials = {}

    models_trials = {}

    trial_types = [
        "Go",
        "Go-Touch",
        "Nogo",
        "Nogo-Touch"
    ]

    for trial_type in trial_types:

        df_trial = prepare_iti2_dataframe(
            df_total,
            trial_type
        )

        dfs_trials[trial_type] = df_trial

        models_trials[trial_type] = (
            fit_models_par_souris(
                df_trial
            )
        )
        print(trial_type, len(models_trials[trial_type])
)

    return dfs_trials, models_trials