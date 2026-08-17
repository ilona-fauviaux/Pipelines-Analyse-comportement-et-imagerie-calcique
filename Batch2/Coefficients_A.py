import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from Program_A import *

def logistic_previous_outcome_trial_perf(df):

    df = df.dropna(subset=["Prev_Perf", "Performance"])

    model = None

    try:

        model = smf.logit(
            "Prev_Perf ~ Performance",
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

# def logistic_previous_outcome_trial_choice(df):

#     df = df.dropna(subset=["Prev_Perf", "Choice"])

#     model = None

#     try:

#         model = smf.logit(
#             "Choice ~ Prev_Perf",
#             data=df
#         ).fit(disp=False)

#         coeffs = pd.DataFrame({
#             "Variable": model.params.index,
#             "Coefficient": model.params.values,
#             "p_value": model.pvalues.values
#         })

#     except Exception as e:

#         coeffs = pd.DataFrame({
#             "Variable": ["ERROR"],
#             "Coefficient": [np.nan],
#             "p_value": [np.nan],
#             "Message": [str(e)]
#         })

#     return coeffs, model

def logistic_previous_lick_choice(df):

    df = df.dropna(
        subset=["Prev_Lick", "Choice"]
    )

    model = smf.logit(
        "Prev_Lick ~ Choice",
        data=df
    ).fit(disp=False)

    coeffs = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    return coeffs, model


#coefficients ITI2
def mixed_previous_perf(df):

    df = df.dropna(
        subset=[
            "Performance",
            "Prev_Perf",
            "Mouse"
        ]
    )

    model = smf.mixedlm(
        "Performance ~ Prev_Perf",
        data=df,
        groups=df["Mouse"]
    ).fit()

    coeffs = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    return coeffs, model

def mixed_previous_lick(df):

    df = df.dropna(
        subset=[
            "Choice",
            "Prev_Lick",
            "Mouse"
        ]
    )

    model = smf.mixedlm(
        "Choice ~ Prev_Lick",
        data=df,
        groups=df["Mouse"]
    ).fit()

    coeffs = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    return coeffs, model