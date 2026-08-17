import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.stats import iqr
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, savgol_filter
from scipy.integrate import trapezoid
import seaborn as sns
import os
import json

# =========================================================
# PATHS
# =========================================================
base_dir      = r"D:\ENSC\1A\Stage 1A\Imageries calciques\Donnees_brutes\COMPORTEMENT\20260430_M3_synchro"
xls_path      = os.path.join(base_dir, "bpod.xls")
json_path     = os.path.join(base_dir, "params_trial.json")
analog_path   = os.path.join(base_dir, "analog.txt")
suite2p_path  = os.path.join(base_dir, "suite2p")
# --- Nom de la souris extrait automatiquement du dossier ---
mouse_name  = "M3"
output_root = r"D:\ENSC\1A\Stage 1A\Imageries calciques\graphes"
output_dir  = os.path.join(output_root, "COMPORTEMENT", mouse_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Souris détectée : {mouse_name} -> figures dans {output_dir}")

def save_current_fig(name, close=True):
    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)

# =========================================================
# LOAD SUITE2P
# =========================================================
F      = np.load(os.path.join(suite2p_path, "F.npy"))
Fneu   = np.load(os.path.join(suite2p_path, "Fneu.npy"))
iscell = np.load(os.path.join(suite2p_path, "iscell.npy"))

sampling_rate = 30.9609

cells = np.where(iscell[:, 0] == 1)[0]
F     = F[cells]
Fneu  = Fneu[cells]

# =========================================================
# PREPROCESSING DF/F
# =========================================================
Fcorr = F - 0.7 * Fneu
Fcorr = gaussian_filter1d(Fcorr, sigma=0.1 * sampling_rate)

window  = 300
sliding = np.lib.stride_tricks.sliding_window_view(Fcorr, window, axis=1)
std_sw  = sliding.std(axis=2)
idx_min = np.argmin(std_sw, axis=1)

baseline      = np.array([sliding[i, idx_min[i]].mean() for i in range(Fcorr.shape[0])])
baseline_safe = np.where(baseline == 0, np.finfo(float).eps, baseline)
dff           = (Fcorr - baseline_safe[:, None]) / baseline_safe[:, None]
dff           = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)

# =========================================================
# LOAD ANALOG
# Layout (colonnes 1-indexées) :
#   Col 1 -> temps vibration (ms)     Col 2 -> signal vibration (> 2.5)
#   Col 3 -> temps social touch (ms)  Col 4 -> signal social touch (== 1)
# =========================================================
analog = np.loadtxt(analog_path)

time_vib_s    = analog[:, 0] / 1000.0
vib_signal    = analog[:, 1]
time_social_s = analog[:, 2] / 1000.0
social_signal = analog[:, 3].astype(int)

# =========================================================
# LOAD XLS
# =========================================================
xls = pd.read_excel(xls_path, header=6)
print(xls.shape)
print(xls.columns)
print(xls.head())
# =========================================================
# LOAD JSON
# =========================================================
with open(json_path, "r") as f:
    js = json.load(f)
events_col = xls["MSG"].fillna("").astype(str)             # colonne E
col_mouse  = xls.iloc[:, 5].fillna("").astype(str).values   # colonne F

xls_times_full = pd.to_datetime(xls.iloc[:, 1], errors="coerce")
t0_full        = xls_times_full.dropna().iloc[0]
time_col       = (xls_times_full - t0_full).dt.total_seconds().values

# ==============================================================
# MASQUE "DANS UN ESSAI" ENTRE "NEW TRIAL" ET "THE TRIAL ENDED"
# ==============================================================
def build_in_trial_mask(events):
    mask     = np.zeros(len(events), dtype=bool)
    in_trial = False
    for i, ev in enumerate(events):
        if ev == "New trial":
            in_trial = True
        if in_trial:
            mask[i] = True
        if ev == "The trial ended":
            in_trial = False
    return mask


in_trial_mask = build_in_trial_mask(events_col)
print(f"Lignes dans un essai : {in_trial_mask.sum()} / {len(in_trial_mask)}")

# ==============================================================
# SYNCHRONISATION EXCEL - ANALOG
# ==============================================================
# -------------------------------------------
# synchronisation vibration
# -------------------------------------------
def detect_vibrations_analog(time, signal, height=3.0, distance=2000):
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    vib_intervals = []
    for p in peaks:
        start = p
        end   = p
        while start > 0 and signal[start] > 1.0:
            start -= 1
        while end < len(signal) - 1 and signal[end] > 1.0:
            end += 1
        vib_intervals.append((time[start], time[end]))
    return vib_intervals


def detect_vibrations_excel(xls, json_params, t0):

    times  = pd.to_datetime(xls.iloc[:, 1], errors="coerce")
    events = xls.iloc[:, 4].fillna("").astype(str)

    trial_amp  = [item["amp"] for item in json_params]
    trial_type = [item["trial_type"] for item in json_params]

    vib = []
    in_trial = False
    trial_idx = -1
    stim_on = None

    stim_count = 0
    off_count = 0

    for i in range(len(xls)):
        ev = events.iloc[i]
        t  = times.iloc[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            stim_on = None
            continue

        if ev == "The trial ended":
            in_trial = False
            stim_on = None
            continue

        if not in_trial:
            continue

        if trial_idx < 0 or trial_idx >= len(trial_amp):
            continue

        amp = trial_amp[trial_idx]
        ttype = trial_type[trial_idx]

        if ttype == "Nogo-Touch":
            amp = 0

        # detect stim
        if ev == "Stimulus" and amp > 0 and pd.notna(t):
            stim_on = t
            stim_count += 1
        elif ev == "StimOff" and stim_on is not None and amp > 0 and pd.notna(t):
            vib.append((
                (stim_on - t0).total_seconds(),
                (t - t0).total_seconds()
            ))
            stim_on = None
            off_count += 1
    print("Stimulus utilisés:", stim_count)
    print("StimOff utilisés:", off_count)
    print("Vib détectées:", len(vib))
    return vib


def match_vibrations(excel_intervals, analog_intervals):
    excel_intervals  = sorted(excel_intervals, key=lambda x: x[0])
    analog_intervals = sorted(analog_intervals, key=lambda x: x[0])
    n = min(len(excel_intervals), len(analog_intervals))
    matches = []
    used_excel = set()
    for i, (an_s, an_e) in enumerate(analog_intervals):
        an_center = (an_s + an_e) / 2
        best_j = None
        best_dist = float("inf")
        for j, (ex_s, ex_e) in enumerate(excel_intervals):
            if j in used_excel:
                continue
            ex_center = (ex_s + ex_e) / 2
            dist = abs(an_center - ex_center)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            ex_s, ex_e = excel_intervals[best_j]
            dt_start = an_s - ex_s
            dt_end = an_e - ex_e

            matches.append(((ex_s, ex_e), (an_s, an_e), (dt_start, dt_end)))

            used_excel.add(best_j)

    return matches

def align_excel_vibrations_to_analog(matches):
    aligned = []
    for (ex_start, ex_end), (an_start, an_end), (dt_start,dt_end) in matches:
        aligned.append(
            (
                (ex_start, ex_end),
                (ex_start + dt_start, ex_end + dt_end),
                (an_start, an_end),
                (dt_start, dt_end)
            )
        )

    return aligned


# ---------------------------------
# TEST DETECTION DES VIBRATIONS
# ---------------------------------
print("\nVIBRATION\n")

vib_analog  = detect_vibrations_analog(time_vib_s, vib_signal)
vib_excel   = detect_vibrations_excel(xls, js, t0_full)
vib_matches = match_vibrations(vib_analog, vib_excel)


aligned_vibrations = align_excel_vibrations_to_analog(vib_matches)
print("\nPremières vibrations alignées :")
for vib in aligned_vibrations[:5]:
    print(vib)
print("vib_excel :")
for i, v in enumerate(vib_excel[:5]):
    print(i, v)

print("\nvib_matches :")
for i, m in enumerate(vib_matches[:5]):
    print(i, m[0])

print("Nb analog :", len(vib_analog))
print("Nb excel  :", len(vib_excel))
print("Nb matches :", len(vib_matches))

# -------------------------------------------
# deboggage
# -------------------------------------------
n_amp_pos = sum(1 for item in js if item.get("amp", 0) > 0)
print(f"[DEBUG] Nb essais avec amp > 0 dans le json : {n_amp_pos}")
stim_count = (events_col == "Stimulus").sum()
stimoff_count = (events_col == "StimOff").sum()
print("Stimulus:", stim_count)
print("StimOff:", stimoff_count)
print("Nb paires Excel brutes:", len(detect_vibrations_excel(xls, js, t0_full)))

vib_matches_excel_first = match_vibrations(vib_excel, vib_analog)


# -------------------------------------------
# synchronisation social touch
# -------------------------------------------
print("\nSOCIAL TOUCH\n")


def detect_social_touch_analog(time, signal):
    signal = np.asarray(signal).astype(int)
    diff = np.diff(signal)
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1
    if signal[0] == 1:
        starts = np.insert(starts, 0, 0)
    if signal[-1] == 1:
        ends = np.append(ends, len(signal) - 1)
    social_intervals = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        social_intervals.append((time[s], time[e]))
    return social_intervals


def detect_social_touch_excel(xls, t0):
    times  = pd.to_datetime(xls.iloc[:, 1], errors="coerce")  # colonne B
    events = xls.iloc[:, 4].fillna("").astype(str)             # colonne E
    mask   = build_in_trial_mask(events)
    times  = times[mask].reset_index(drop=True)
    events = events[mask].reset_index(drop=True)
    social_intervals = []
    social_on = None
    for i in range(len(times)):
        ev = events.iloc[i]
        t  = times.iloc[i]
        if ev == "SocialTouch" and pd.notna(t):
            social_on = t
        elif ev != "SocialTouch" and social_on is not None:
            if pd.notna(t):
                social_intervals.append((
                    (social_on - t0).total_seconds(),
                    (t - t0).total_seconds()
                ))
            social_on = None
    return social_intervals


def match_social_touch(excel_intervals, analog_intervals):
    excel_intervals  = sorted(excel_intervals, key=lambda x: x[0])
    analog_intervals = sorted(analog_intervals, key=lambda x: x[0])
    n = min(len(excel_intervals), len(analog_intervals))
    matches = []
    for i in range(n):
        ex_s, ex_e = excel_intervals[i]
        an_s, an_e = analog_intervals[i]
        dt = an_s - ex_s
        matches.append(((ex_s, ex_e), (an_s, an_e), dt))
    return matches


def align_excel_events_to_analog(matches):
    aligned_events = []
    for (ex_start, ex_end), (an_start, an_end), dt in matches:
        aligned_ex_start = ex_start + dt
        aligned_ex_end   = ex_end + dt
        aligned_events.append((
            (ex_start, ex_end),
            (aligned_ex_start, aligned_ex_end),
            (an_start, an_end),
            dt
        ))
    return aligned_events


# ----------------------------------
# Test
# ----------------------------------
social_analog = detect_social_touch_analog(time_social_s, social_signal)
social_excel  = detect_social_touch_excel(xls, t0_full)

social_matches = match_social_touch(social_excel, social_analog)
print("Nb matches:", len(social_matches))

for m in social_matches[:5]:
    print(m)

aligned_social = align_excel_events_to_analog(social_matches)

for ev in aligned_social[:5]:
    print(ev)

# =========================================================
# NORMALISATION AVEC LES 500ms PRECEDENTES
# =========================================================
def extract_event_dff(Fcorr, intervals, sampling_rate,baseline_duration=0.5,remove_start=0,post_duration=None):
    baseline_frames = int(baseline_duration * sampling_rate)
    remove_frames = int(remove_start * sampling_rate)
    events_dff = []
    for start, end in intervals:
        start_frame = int(start * sampling_rate)
        end_frame   = int(end * sampling_rate)
        baseline_start = start_frame - baseline_frames
        baseline_end = start_frame
        if baseline_start < 0:
            continue
        F0 = np.mean(Fcorr[:, baseline_start:baseline_end],axis=1)
        F0[F0 == 0] = np.finfo(float).eps
        response_start = start_frame + remove_frames
        if post_duration is not None:
            response_end = response_start + int(post_duration*sampling_rate)
        else:
            response_end = end_frame
        if response_end > Fcorr.shape[1]:
            continue
        response = Fcorr[:, response_start:response_end]
        dff_event = (response - F0[:,None]) / F0[:,None]
        events_dff.append(dff_event)
    return events_dff

# Vibrations :
dff_vibrations = extract_event_dff(
    Fcorr,
    vib_analog,
    sampling_rate,
    baseline_duration=0.5,
    remove_start=0
)

# Social Touch :
dff_social = extract_event_dff(
    Fcorr,
    social_analog,
    sampling_rate,
    baseline_duration=0.5,
    remove_start=0.3
)
# =========================================================
# EXTRACTION DES REPONSES EVENEMENTIELLES
# =========================================================
def extract_aligned_dff(Fcorr, intervals, sampling_rate,pre_time=0.5,post_time=1.0,remove_start=0):
    pre_frames = int(pre_time * sampling_rate)
    post_frames = int(post_time * sampling_rate)
    remove_frames = int(remove_start * sampling_rate)
    events = []
    for start, end in intervals:
        event_frame = int(start * sampling_rate)
        baseline_start = event_frame - pre_frames
        baseline_end = event_frame
        response_start = event_frame + remove_frames
        response_end = response_start + post_frames
        if baseline_start < 0:
            continue
        if response_end > Fcorr.shape[1]:
            continue
        F0 = np.mean(Fcorr[:, baseline_start:baseline_end],axis=1)
        F0[F0 == 0] = np.finfo(float).eps
        start_window = event_frame - pre_frames
        end_window = response_end
        signal = Fcorr[:, start_window:end_window]
        dff_event = (signal - F0[:,None]) / F0[:,None]
        events.append(dff_event)
    return np.array(events)
# =========================================================
# DETECTION DES TIMEOUT
# =========================================================
def match_timeout_to_social_touch_with_trials(events, mask, social_intervals, time_col):
    events   = np.asarray(events.fillna("").astype(str))
    mask     = np.asarray(mask).astype(bool)
    time_col = np.asarray(time_col)
    results = []
    in_trial = False
    trial_has_timeout = False
    trial_start_time = None
    trial_end_time = None
    trial_idx = -1
    for i in range(len(events)):
        if not mask[i]:
            continue
        ev = events[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            trial_has_timeout = False
            trial_start_time = time_col[i]
            trial_end_time = None
        if not in_trial:
            continue
        trial_end_time = time_col[i]
        if "Timeout" in ev:
            trial_has_timeout = True
        if ev == "The trial ended":
            if trial_has_timeout:
                best = None
                for s_start, s_end in social_intervals:
                    if trial_start_time <= s_start <= trial_end_time:
                        best = (s_start, s_end)
                        break
                if best is not None:
                    results.append((trial_idx,trial_start_time,trial_end_time,best[0],best[1]))
            in_trial = False
    return results


def project_timeout_trials_to_analog(timeout_social_indices, social_matches, excel_trials):
    projected = []
    for social_idx, (trial_start, trial_end) in zip(timeout_social_indices, excel_trials):
        (ex_s, ex_e), (an_s, an_e), dt = social_matches[social_idx]
        an_trial_start = trial_start + dt
        an_trial_end   = trial_end + dt
        projected.append((social_idx, (trial_start, trial_end), (an_trial_start, an_trial_end), dt))
    return projected
#=================================================
# GO
#=================================================

# =========================================================
# DETECTION DES GO-TIMEOUT
# =========================================================

def select_go_timeouts(projected_timeouts, json_params):
    go_timeouts = []
    for timeout in projected_timeouts:
        trial_idx = timeout[0]
        trial_type = json_params[trial_idx]["trial_type"]
        if trial_type in ["Go", "Go-Touch"]:
            go_timeouts.append(timeout)
    return go_timeouts

# -------------------------------
# Test go-timeout
# -------------------------------
mask = in_trial_mask
social_intervals = detect_social_touch_excel(xls, t0_full)
social_matches   = match_social_touch(social_intervals, social_analog)

timeout_results = match_timeout_to_social_touch_with_trials(events_col, mask, social_intervals, time_col)

timeout_social_indices = []
excel_trials = []

for trial_idx, trial_start, trial_end, s_start, s_end in timeout_results:

    for i, (ex, an, dt) in enumerate(social_matches):

        if ex[0] == s_start and ex[1] == s_end:

            timeout_social_indices.append(i)
            excel_trials.append((trial_start, trial_end))

            break

projected_timeouts = project_timeout_trials_to_analog(timeout_social_indices, social_matches, excel_trials)

print("\nNb essais Timeout retrouvés :", len(timeout_results))
print("Nb timeouts projetés sur l'analogique :", len(projected_timeouts))
for p in projected_timeouts[:5]:
    print(p)
go_timeouts = select_go_timeouts(
    projected_timeouts,
    js
)

print("\nNb Go-Timeout :", len(go_timeouts))
for t in go_timeouts[:5]:
    print(t)

# =========================================================
# DETECTION DES MISS (jaune)
# =========================================================
def detect_miss_excel(xls, json_params, t0):
    times = pd.to_datetime(xls.iloc[:, 1], errors="coerce")
    events = xls.iloc[:, 4].fillna("").astype(str)
    trial_amp = [item["amp"] for item in json_params]
    trial_type = [item["trial_type"] for item in json_params]
    miss = []
    trial_idx = -1
    in_trial = False
    trial_start = None
    stim_start = None
    stim_end = None
    has_reward = False
    for i in range(len(xls)):
        ev = events.iloc[i]
        t = times.iloc[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            trial_start = t
            stim_start = None
            stim_end = None
            has_reward = False
            continue
        if not in_trial:
            continue
        amp = trial_amp[trial_idx]
        ttype = trial_type[trial_idx]
        if ttype == "Nogo-Touch":
            amp = 0
        if ev == "Reward":
            has_reward = True
        # vibration du trial
        if ev == "Stimulus" and amp > 0:
            stim_start = t
        elif ev == "StimOff" and stim_start is not None and amp > 0:
            stim_end = t
        # fin du trial
        elif ev == "The trial ended":
            trial_end = t
            if amp > 0 and not has_reward and stim_start is not None:
                miss.append((((trial_start - t0).total_seconds(),(trial_end - t0).total_seconds()),((stim_start - t0).total_seconds(),(stim_end - t0).total_seconds())))
            in_trial = False
    return miss

def project_miss_to_analog(miss_list, matches, tol=1e-6):
    projected = []
    for (trial_start, trial_end), (vib_start, vib_end) in miss_list:
        found = False
        for (ex_s, ex_e), (_, _), (dt_s, dt_e) in matches:
            # Matching sur la vibration associée
            if (abs(ex_s - vib_start) < tol and
                abs(ex_e - vib_end) < tol):
                # On garde le trial complet + son décalage analog
                projected.append(((trial_start, trial_end),(dt_s, dt_e)))
                found = True
                break
        if not found:
            print(f"WARNING : vibration Excel {vib_start:.3f}-{vib_end:.3f} "f"(associée au miss {trial_start:.3f}-{trial_end:.3f}) non matchée")
    return projected

def apply_dt_to_intervals(intervals_with_dt):
    projected_intervals = []
    for (miss_start, miss_end), (dt_s, dt_e) in intervals_with_dt:
        projected_intervals.append((miss_start + dt_s,miss_end + dt_e))
    return projected_intervals

# -------------------------------
# Test miss
# -------------------------------
# Détection des vibrations Excel
vib_intervals = detect_vibrations_excel(xls, js, t0_full)

# Matching Excel -> Analog
vib_matches = match_vibrations(vib_intervals, vib_analog)

# Détection des Miss
miss_results = detect_miss_excel(xls, js, t0_full)

# Projection
projected_miss_dt = project_miss_to_analog(miss_results,vib_matches)

# Application des décalages
projected_miss = apply_dt_to_intervals(projected_miss_dt)

print("Nb Miss :", len(miss_results))
print("Nb Miss projetés :", len(projected_miss))
print("\n")

for p in projected_miss[:5]:
    print(p)
print("\n")

# =========================================================
# DETECTION DES HIT (bleu)
# =========================================================
def detect_hit_excel(xls, json_params, t0):
    times = pd.to_datetime(xls.iloc[:, 1], errors="coerce")
    events = xls.iloc[:, 4].fillna("").astype(str)
    trial_amp = [item["amp"] for item in json_params]
    trial_type = [item["trial_type"] for item in json_params]
    hit = []
    trial_idx = -1
    in_trial = False
    trial_start = None
    stim_start = None
    stim_end = None
    has_reward = False
    for i in range(len(xls)):
        ev = events.iloc[i]
        t = times.iloc[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            trial_start = t
            stim_start = None
            stim_end = None
            has_reward = False
            continue
        if not in_trial:
            continue
        amp = trial_amp[trial_idx]
        ttype = trial_type[trial_idx]
        if ttype == "Nogo-Touch":
            amp = 0
        if ev == "Reward":
            has_reward = True
        # vibration du trial
        if ev == "Stimulus" and amp > 0:
            stim_start = t
        elif ev == "StimOff" and stim_start is not None and amp > 0:
            stim_end = t
        # fin du trial
        elif ev == "The trial ended":
            trial_end = t
            if amp > 0 and has_reward and stim_start is not None:
                hit.append((((trial_start - t0).total_seconds(),(trial_end - t0).total_seconds()),((stim_start - t0).total_seconds(),(stim_end - t0).total_seconds())))
            in_trial = False
    return hit

def project_hit_to_analog(miss_list, matches, tol=1e-6):
    projected = []
    for (trial_start, trial_end), (vib_start, vib_end) in miss_list:
        found = False
        for (ex_s, ex_e), (_, _), (dt_s, dt_e) in matches:
            # Matching sur la vibration associée
            if (abs(ex_s - vib_start) < tol and
                abs(ex_e - vib_end) < tol):
                # On garde le trial complet + son décalage analog
                projected.append(((trial_start, trial_end),(dt_s, dt_e)))
                found = True
                break
        if not found:
            print(f"WARNING : vibration Excel {vib_start:.3f}-{vib_end:.3f} "f"(associée au hit {trial_start:.3f}-{trial_end:.3f}) non matchée")
    return projected

def apply_dt_to_intervals(intervals_with_dt):
    projected_intervals = []
    for (hit_start, hit_end), (dt_s, dt_e) in intervals_with_dt:
        projected_intervals.append((hit_start + dt_s,hit_end + dt_e))
    return projected_intervals

# -------------------------------
# Test hit
# -------------------------------
# Détection des vibrations Excel
vib_intervals = detect_vibrations_excel(xls, js, t0_full)

# Matching Excel -> Analog
vib_matches = match_vibrations(vib_intervals, vib_analog)

# Détection des Miss
hit_results = detect_hit_excel(xls, js, t0_full)

# Projection
projected_hit_dt = project_hit_to_analog(hit_results,vib_matches)

# Application des décalages
projected_hit = apply_dt_to_intervals(projected_hit_dt)

print("Nb Hit :", len(hit_results))
print("Nb Hit projetés :", len(projected_hit))
print("\n")

for p in projected_hit[:5]:
    print(p)
print("\n")

#=================================================
# NOGO
#=================================================
# =========================================================
# DETECTION DES FALSE ALARM
# =========================================================
def select_false_alarm(projected_timeouts, json_params):
    false_alarms = []
    for timeout in projected_timeouts:
        trial_idx = timeout[0]
        trial_type = json_params[trial_idx]["trial_type"]
        if trial_type == "Nogo-Touch":
            false_alarms.append(timeout)
    return false_alarms

# -------------------------------
# Test false alarm
# -------------------------------
projected_timeouts = project_timeout_trials_to_analog(timeout_social_indices, social_matches, excel_trials)
print("\nNb essais Timeout retrouvés :", len(timeout_results))
print("Nb timeouts projetés sur l'analogique :", len(projected_timeouts))
for p in projected_timeouts[:5]:
    print(p)
false_alarms = select_false_alarm(projected_timeouts,js)
print("\nNb False alarm :", len(false_alarms))
for fa in false_alarms[:5]:
    print(fa)

# =========================================================
# DETECTION DES CORRECT REJECTION (vert)
# =========================================================
def detect_nogo_touch_trials(xls, json_params, t0, social_intervals):
    times = pd.to_datetime(xls.iloc[:, 1], errors="coerce")
    events = xls.iloc[:, 4].fillna("").astype(str)
    trial_type = [item["trial_type"] for item in json_params]
    nogo_cr = []
    trial_idx = -1
    in_trial = False
    trial_start = None
    has_timeout = False
    for i in range(len(xls)):
        ev = events.iloc[i]
        t = times.iloc[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            trial_start = t
            has_timeout = False
            continue
        if not in_trial:
            continue
        if "Timeout" in ev:
            has_timeout = True
        if ev == "The trial ended":
            trial_end = t
            if trial_type[trial_idx] == "Nogo-Touch" and not has_timeout:
                trial_start_s = (trial_start - t0).total_seconds()
                trial_end_s   = (trial_end - t0).total_seconds()
                social_found = None
                for st_start, st_end in social_intervals:
                    if st_start >= trial_start_s and st_end <= trial_end_s:
                        social_found = (st_start, st_end)
                        break
                if social_found is not None:
                    nogo_cr.append(((trial_start_s, trial_end_s),social_found))
            in_trial = False
    return nogo_cr

def project_cr_to_analog(nogo_social, social_matches, tol=1e-6):
    projected = []
    for (trial_start, trial_end), (social_start, social_end) in nogo_social:
        found = False
        for (ex_s, ex_e), (_, _), dt in social_matches:
            if (abs(ex_s - social_start) < tol and abs(ex_e - social_end) < tol):
                projected.append(((trial_start, trial_end),dt))
                found = True
                break
        if not found:
            print(
                f"WARNING : Social Touch {social_start:.3f}-{social_end:.3f} "
                f"(associé au trial {trial_start:.3f}-{trial_end:.3f}) non matché"
            )
    return projected

def apply_dt_to_cr(intervals):
    projected_intervals = []
    for (cr_start, cr_end), dt in intervals:
        projected_intervals.append((cr_start + dt,cr_end + dt))
    return projected_intervals

# -------------------------------
# Test Correct rejection
# -------------------------------
# Détection des Social Touch Excel
social_excel = detect_social_touch_excel(xls, t0_full)
# Matching Excel -> Analog
social_matches = match_social_touch(
    social_excel,
    social_analog
)
# Détection des trials Nogo-Touch avec leur Social Touch associé
cr_results = detect_nogo_touch_trials(
    xls,
    js,
    t0_full,
    social_excel
)
# Projection du trial complet avec le décalage Social Touch
projected_cr_dt = project_cr_to_analog(cr_results,social_matches)
# Application des décalages
projected_cr = apply_dt_to_cr(projected_cr_dt)

print("Nb Correct rejection :", len(cr_results))
print("Nb CR projetés :", len(projected_cr))
print("\n")
for p in projected_cr[:5]:
    print(p)
print("\n")
# =========================================================
# HEATMAP
# =========================================================
n_frames = dff.shape[1]
time_imaging = np.arange(n_frames) / sampling_rate

t_max = max(
    time_imaging[-1],
    vib_analog[-1][1] if len(vib_analog) > 0 else 0,
    social_analog[-1][1] if len(social_analog) > 0 else 0,
)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.05)

ax = fig.add_subplot(gs[1])
ax_top = fig.add_subplot(gs[0], sharex=ax)
ax_top.tick_params(axis="x", labelbottom=False)


ax.set_xlim(0, t_max)
ax_top.set_xlim(0, t_max)

# -------------------------------------------------
# HEATMAP (neurons x time) : du moins actif (bas) au plus actif (haut)
# -------------------------------------------------
activity = np.mean(dff, axis=1)
order = np.argsort(activity)
dff_sorted = dff[order, :]

vmin = np.percentile(dff_sorted, 5)
vmax = np.percentile(dff_sorted, 99)

im = ax.imshow(
    dff_sorted,
    aspect="auto",
    cmap="RdPu",
    interpolation="nearest",
    origin="lower",
    extent=[0, time_imaging[-1], 0, dff_sorted.shape[0]],
    vmin=vmin,
    vmax=vmax
)
cbar = fig.colorbar(im, ax=[ax_top, ax], fraction=0.03, pad=0.02)
cbar.set_label("ΔF/F")
ax.set_ylabel("Neurons")
ax.set_xlabel("Time (s)")

# ---------------------------------------------------------
# VIBRATIONS (ANALOG) -> barres vertes
# ---------------------------------------------------------
for start, end in vib_analog:
    ax_top.axvspan(start, end, color="green", alpha=0.75)

# ---------------------------------------------------------
# SOCIAL TOUCH (ANALOG) -> segments bleus
# ---------------------------------------------------------
# ---------------------------
# -0.3 premieres secondes
# ---------------------------
def remove_start_intervals(intervals, remove_start=0.3):
    trimmed_intervals = []

    for start, end in intervals:

        new_start = start + remove_start

        # On garde uniquement les intervalles encore valides
        if new_start < end:
            trimmed_intervals.append((new_start, end))

    return trimmed_intervals


social_analog_trimmed = remove_start_intervals(social_analog,remove_start=0.3)
for start, end in social_analog_trimmed:
    ax_top.hlines(
        y=0.3,
        xmin=start,
        xmax=end,
        color="blue",
        linewidth=6
    )

# ---------------------------------------------------------
# GO-TIMEOUT (rose)
# ---------------------------------------------------------

for timeout in go_timeouts:

    # récupération de l'intervalle analog du trial
    t_start, t_end = timeout[2]

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="pink",
        linewidth=2,
        alpha=0.5,
        zorder=2
    )

    ax_top.add_patch(rect)

# ---------------------------------------------------------
# MISS (jaune)
# ---------------------------------------------------------
for t_start, t_end in projected_miss:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        1,
        facecolor="gold",
        alpha=0.25,
        zorder=10
    )

    ax_top.add_patch(rect)

ax_top.set_ylim(0, 1)

# ---------------------------------------------------------
# HIT (bleu turquoise)
# ---------------------------------------------------------

for t_start, t_end in projected_hit:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="turquoise",
        alpha=0.3,
        zorder=2
    )

    ax_top.add_patch(rect)

# ---------------------------------------------------------
# CORRECT REJECTION (vert)
# ---------------------------------------------------------
for t_start, t_end in projected_cr:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="limegreen",
        alpha=0.3,
        zorder=2
    )

    ax_top.add_patch(rect)

# ------------------------------------------------------
# FALSE ALARM -> rectangles rouges
# ------------------------------------------------------

for false_alarm in false_alarms:
    t_start, t_end = false_alarm[2]

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="red",
        linewidth=2,
        alpha=0.25,
        zorder=3
    )

    ax_top.add_patch(rect)


ax_top.set_yticks([])
ax_top.set_xticks([])

ax_top.set_ylim(0, 1)
#ax_top.axis("off")
print("\n")
print("\n")
print("barres vertes : vibrations")
print("segment bleu : social touch")
print("\n")
print(" rose : Go-timeout")
print(" rouge : False alarm")
print(" jaune : Miss")
print(" vert : Correct rejection")
print(" bleu : Hit")
fig.savefig(os.path.join(output_dir, "Heatmap_dff_brute.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

#plt.show()

# =========================================================
# HEATMAP NORMALISEE
# =========================================================
def replace_local_dff(dff_local,Fcorr,intervals,sampling_rate,baseline_duration=0.5,remove_start=0):
    baseline_frames = int(baseline_duration * sampling_rate)
    remove_frames = int(remove_start * sampling_rate)
    for start, end in intervals:
        start_frame = int(start * sampling_rate)
        end_frame   = int(end * sampling_rate)
        baseline_start = start_frame - baseline_frames
        baseline_end   = start_frame
        if baseline_start < 0:
            continue
        response_start = start_frame + remove_frames
        if response_start >= end_frame:
            continue
        F0 = np.mean(Fcorr[:, baseline_start:baseline_end],axis=1)
        F0[F0 == 0] = np.finfo(float).eps
        local_dff = (Fcorr[:, response_start:end_frame]- F0[:, None]) / F0[:, None]
        dff_local[:, response_start:end_frame] = local_dff
    return dff_local

dff_local = dff.copy()

dff_local = replace_local_dff(dff_local,Fcorr,vib_analog,sampling_rate,baseline_duration=0.5,remove_start=0)
dff_local = replace_local_dff(dff_local,Fcorr,social_analog,sampling_rate,baseline_duration=0.5,remove_start=0.3)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.05)

ax = fig.add_subplot(gs[1])
ax_top = fig.add_subplot(gs[0], sharex=ax)
ax_top.tick_params(axis="x", labelbottom=False)
ax.set_xlim(0, t_max)
ax_top.set_xlim(0, t_max)

activity = np.mean(dff_local, axis=1)
order = np.argsort(activity)
dff_sorted = dff_local[order]

vmin = np.percentile(dff_sorted, 5)
vmax = np.percentile(dff_sorted, 99)

im = ax.imshow(
    dff_sorted,
    aspect="auto",
    cmap="RdPu",
    interpolation="nearest",
    origin="lower",
    extent=[0, time_imaging[-1], 0, dff_sorted.shape[0]],
    vmin=vmin,
    vmax=vmax
)
cbar = fig.colorbar(im, ax=[ax_top, ax], fraction=0.03, pad=0.02)
cbar.set_label("Z-score")
ax.set_ylabel("Neurons")
ax.set_xlabel("Time (s)")

# ---------------------------------------------------------
# VIBRATIONS (ANALOG) -> barres vertes
# ---------------------------------------------------------
for start, end in vib_analog:
    ax_top.axvspan(start, end, color="green", alpha=0.75)

# ---------------------------------------------------------
# SOCIAL TOUCH (ANALOG) -> segments bleus
# ---------------------------------------------------------
# ---------------------------
# -0.3 premieres secondes
# ---------------------------
def remove_start_intervals(intervals, remove_start=0.3):
    trimmed_intervals = []

    for start, end in intervals:

        new_start = start + remove_start

        # On garde uniquement les intervalles encore valides
        if new_start < end:
            trimmed_intervals.append((new_start, end))

    return trimmed_intervals


# Application aux social touch
social_analog_trimmed = remove_start_intervals(social_analog,remove_start=0.3)
for start, end in social_analog_trimmed:
    ax_top.hlines(
        y=0.3,
        xmin=start,
        xmax=end,
        color="blue",
        linewidth=6
    )

# ---------------------------------------------------------
# GO-TIMEOUT (rose)
# ---------------------------------------------------------

for timeout in go_timeouts:

    # récupération de l'intervalle analog du trial
    t_start, t_end = timeout[2]

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="pink",
        linewidth=2,
        alpha=0.5,
        zorder=2
    )

    ax_top.add_patch(rect)

# ---------------------------------------------------------
# MISS (jaune)
# ---------------------------------------------------------
for t_start, t_end in projected_miss:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        1,
        facecolor="gold",
        alpha=0.25,
        zorder=10
    )

    ax_top.add_patch(rect)

ax_top.set_ylim(0, 1)

# ---------------------------------------------------------
# HIT (bleu turquoise)
# ---------------------------------------------------------

for t_start, t_end in projected_hit:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="turquoise",
        alpha=0.3,
        zorder=2
    )

    ax_top.add_patch(rect)

# ---------------------------------------------------------
# CORRECT REJECTION (vert)
# ---------------------------------------------------------
for t_start, t_end in projected_cr:

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="limegreen",
        alpha=0.3,
        zorder=2
    )

    ax_top.add_patch(rect)

# ------------------------------------------------------
# FALSE ALARM -> rectangles rouges
# ------------------------------------------------------

for false_alarm in false_alarms:
    t_start, t_end = false_alarm[2]

    rect = Rectangle(
        (float(t_start), 0),
        float(t_end - t_start),
        dff_sorted.shape[0],
        facecolor="red",
        linewidth=2,
        alpha=0.25,
        zorder=3
    )

    ax_top.add_patch(rect)


ax_top.set_yticks([])
ax_top.set_xticks([])

ax_top.set_ylim(0, 1)
#ax_top.axis("off")
print("\n")
print("\n")
print("barres vertes : vibrations")
print("segment bleu : social touch")
print("\n")
print(" rose : Go-timeout")
print(" rouge : False alarm")
print(" jaune : Miss")
print(" vert : Correct rejection")
print(" bleu : Hit")
fig.savefig(os.path.join(output_dir, "Heatmap_dff_normalisee.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# =========================================================
# HEATMAP NORMALISEE - GO TOUCH
# =========================================================
def get_social_from_trials(trials, social_intervals, remove_start=0.3):
    selected = []
    for trial_start, trial_end in trials:
        for social_start, social_end in social_intervals:
            if trial_start <= social_start and social_end <= trial_end:
                social_start += remove_start
                if social_start < social_end:
                    selected.append((social_start, social_end))
                break
    return selected

hit_social = get_social_from_trials(projected_hit,social_analog,remove_start=0.3)
miss_social = get_social_from_trials(projected_miss,social_analog,remove_start=0.3)
go_trials = [trial[2] for trial in go_timeouts]
timeout_trials = go_trials
timeout_social = get_social_from_trials(timeout_trials,social_analog,remove_start=0.3)
# ------------------------------------------------------------
# Test
# ------------------------------------------------------------
print("Nombre de Hit avec Social Touch :", len(hit_social))
for s in hit_social[:10]:
    print(s)
print("\nNombre de Miss avec Social Touch :", len(miss_social))
for s in miss_social[:10]:
    print(s)
print("\nNombre de Timeout avec Social Touch :", len(timeout_social))
for s in timeout_social[:10]:
    print(s)
# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
heatmap_segments = []
segment_colors = []
hit_segments = []
timeout_segments = []
miss_segments = []

for start, end in hit_social:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    hit_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("turquoise")

for start, end in timeout_social:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    timeout_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("pink")

for start, end in miss_social:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    miss_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("gold")

# ------------------------------------------------------------
# Préparation heatmap
# ------------------------------------------------------------
heatmap = np.concatenate(
    heatmap_segments,
    axis=1
)

activity = np.mean(heatmap, axis=1)
order = np.argsort(activity)
heatmap = heatmap[order]
# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(18,10))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1,6],
    hspace=0.05
)
ax_top = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_top)
# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
vmin = np.percentile(heatmap,5)
vmax = np.percentile(heatmap,99)

ncols = heatmap.shape[1]

im = ax.imshow(
    heatmap,
    aspect="auto",
    cmap="RdPu",
    origin="lower",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
    extent=[0, ncols, 0, heatmap.shape[0]]
)

cbar = fig.colorbar(im, ax=[ax, ax_top])
cbar.set_label("Z-score")

ax.set_ylabel("Neurons")
ax.set_xlabel("Social touch - Go-touch")
# ------------------------------------------------------------
# Largeurs des groupes
# ------------------------------------------------------------
hit_width = sum(
    seg.shape[1]
    for seg in hit_segments
)

timeout_width = sum(
    seg.shape[1]
    for seg in timeout_segments
)

miss_width = sum(
    seg.shape[1]
    for seg in miss_segments
)

# ------------------------------------------------------------
# Rectangles du dessus
# ------------------------------------------------------------
ax_top.add_patch(
    Rectangle(
        (0,0),
        hit_width,
        1,
        color="turquoise"
    )
)

ax_top.add_patch(
    Rectangle(
        (hit_width,0),
        timeout_width,
        1,
        color="pink"
    )
)

ax_top.add_patch(
    Rectangle(
        (hit_width+timeout_width,0),
        miss_width,
        1,
        color="gold"
    )
)

# titres
ax_top.text(
    hit_width/2,
    1.05,
    "Hit",
    ha="center"
)

ax_top.text(
    hit_width+timeout_width/2,
    1.05,
    "Timeout",
    ha="center"
)

ax_top.text(
    hit_width+timeout_width+miss_width/2,
    1.05,
    "Miss",
    ha="center"
)

ax_top.set_ylim(0,1)
ax_top.set_yticks([])
ax_top.set_xticks([])

fig.savefig(os.path.join(output_dir, "Heatmap_normalisee_go_touch.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# =========================================================
# HEATMAP NORMALISEE - VIBRATIONS
# =========================================================
def get_vibration_from_trials(trials, vibration_intervals):
    selected = []
    for trial in trials:
        if len(trial) == 2:
            trial_start, trial_end = trial
        elif len(trial) == 4:
            trial_start, trial_end = trial[2]
        else:
            continue
        for vib_start, vib_end in vibration_intervals:
            if trial_start <= vib_start and vib_end <= trial_end:
                selected.append((vib_start, vib_end))
                break
    return selected

hit_vibration = get_vibration_from_trials(projected_hit,vib_analog)
miss_vibration = get_vibration_from_trials(projected_miss,vib_analog)
timeout_vibration = get_vibration_from_trials(projected_timeouts,vib_analog)
# ------------------------------------------------------------
# test
# ------------------------------------------------------------
print("Nombre de Hit avec Social Touch :", len(hit_vibration))
for s in hit_vibration[:10]:
    print(s)
print("\nNombre de Miss avec Social Touch :", len(miss_vibration))
for s in miss_vibration[:10]:
    print(s)
print("\nNombre de Timeout avec Social Touch :", len(timeout_vibration))
for s in timeout_vibration[:10]:
    print(s)

# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
heatmap_segments = []
segment_colors = []
hit_segments = []
timeout_segments = []
miss_segments = []

for start, end in hit_vibration:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    if end_frame <= start_frame:
        print("Vibration HIT vide :", start, end)
        continue
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    hit_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("turquoise")

for start, end in timeout_vibration:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    if end_frame <= start_frame:
        print("Vibration TIMEOUT vide :", start, end)
        continue
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    timeout_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("pink")

for start, end in miss_vibration:
    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)
    if end_frame <= start_frame:
        print("Vibration MISS vide :", start, end)
        continue
    mean_activity = np.mean(dff_local[:, start_frame:end_frame],axis=1,keepdims=True)
    miss_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
    segment_colors.append("gold")

# ------------------------------------------------------------
# Préparation heatmap
# ------------------------------------------------------------
heatmap = np.concatenate(
    heatmap_segments,
    axis=1
)

activity = np.mean(heatmap, axis=1)
order = np.argsort(activity)
heatmap = heatmap[order]
# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig = plt.figure(figsize=(18,10))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1,6],
    hspace=0.05
)
ax_top = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_top)
# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
vmin = np.percentile(heatmap,5)
vmax = np.percentile(heatmap,99)

ncols = heatmap.shape[1]

im = ax.imshow(
    heatmap,
    aspect="auto",
    cmap="RdPu",
    origin="lower",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
    extent=[0, ncols, 0, heatmap.shape[0]]
)

cbar = fig.colorbar(im, ax=[ax, ax_top])
cbar.set_label("Z-score")

ax.set_ylabel("Neurons")
ax.set_xlabel("Social touch - Go - Vibration")
# ------------------------------------------------------------
# Largeurs des groupes
# ------------------------------------------------------------
hit_width = sum(
    seg.shape[1]
    for seg in hit_segments
)

timeout_width = sum(
    seg.shape[1]
    for seg in timeout_segments
)

miss_width = sum(
    seg.shape[1]
    for seg in miss_segments
)

# ------------------------------------------------------------
# Rectangles du dessus
# ------------------------------------------------------------
ax_top.add_patch(
    Rectangle(
        (0,0),
        hit_width,
        1,
        color="turquoise"
    )
)

ax_top.add_patch(
    Rectangle(
        (hit_width,0),
        timeout_width,
        1,
        color="pink"
    )
)

ax_top.add_patch(
    Rectangle(
        (hit_width+timeout_width,0),
        miss_width,
        1,
        color="gold"
    )
)

# titres
ax_top.text(
    hit_width/2,
    1.05,
    "Hit",
    ha="center"
)


ax_top.text(
    hit_width+timeout_width+miss_width/2,
    1.05,
    "Miss",
    ha="center"
)

ax_top.set_ylim(0,1)
ax_top.set_yticks([])
ax_top.set_xticks([])
fig.savefig(os.path.join(output_dir, "Heatmap_normalisee_vibrations.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# =========================================================
# HEATMAP NORMALISEE - Nogo-Touch
# =========================================================
def get_social_from_trials(trials, social_intervals, remove_start=0.3):
    selected = []
    for trial in trials:
        if len(trial) == 2:
            trial_start, trial_end = trial
        elif len(trial) == 4:
            trial_start, trial_end = trial[2]
        else:
            continue
        for social_start, social_end in social_intervals:
            if trial_start <= social_start and social_end <= trial_end:
                social_start += remove_start
                if social_start < social_end:
                    selected.append((social_start, social_end))

                break
    return selected

cr_social = get_social_from_trials(projected_cr,social_analog,remove_start=0.3)
fa_social = get_social_from_trials(false_alarms,social_analog,remove_start=0.3)
# ------------------------------------------------------------
# test
# ------------------------------------------------------------
print("Correct rejection avec social touch :", len(cr_social))
print(cr_social[:5])
print("False alarm avec social touch :", len(fa_social))
print(fa_social[:5])
# ------------------------------------------------------------
# heatmap
# ------------------------------------------------------------
heatmap_segments = []
cr_segments = []
fa_segments = []
# ---------------------------------------------------
# Correct Rejection
# ---------------------------------------------------

for start, end in cr_social:

    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    if end_frame <= start_frame:
        continue

    mean_activity = np.mean(
        dff_local[:, start_frame:end_frame],
        axis=1,
        keepdims=True
    )

    cr_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)
# ---------------------------------------------------
# False Alarm
# ---------------------------------------------------
for start, end in fa_social:

    start_frame = int(start * sampling_rate)
    end_frame = int(end * sampling_rate)

    if end_frame <= start_frame:
        continue

    mean_activity = np.mean(
        dff_local[:, start_frame:end_frame],
        axis=1,
        keepdims=True
    )

    fa_segments.append(mean_activity)
    heatmap_segments.append(mean_activity)

# ---------------------------------------------------
# heatmap
# ---------------------------------------------------
heatmap = np.concatenate(
    heatmap_segments,
    axis=1
)

activity = np.mean(heatmap, axis=1)
order = np.argsort(activity)
heatmap = heatmap[order]

fig = plt.figure(figsize=(18,10))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1,6],
    hspace=0.05
)

ax_top = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_top)

vmin = np.percentile(heatmap,5)
vmax = np.percentile(heatmap,99)

im = ax.imshow(
    heatmap,
    aspect="auto",
    cmap="RdPu",
    origin="lower",
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
    extent=[0, heatmap.shape[1], 0, heatmap.shape[0]]
)

# ------------------------------------------------------------
# Rectangles du dessus
# ------------------------------------------------------------
cr_width = sum(seg.shape[1]for seg in cr_segments)
fa_width = sum(seg.shape[1]for seg in fa_segments)
ax_top.add_patch(
    Rectangle(
        (0, 0),
        cr_width,
        1,
        color="limegreen",
        alpha=0.5
    )
)


ax_top.add_patch(
    Rectangle(
        (cr_width, 0),
        fa_width,
        1,
        color="red",
        alpha=0.5
    )
)

ax_top.text(
    cr_width / 2,
    1.05,
    "Correct Rejection",
    ha="center",
    fontsize=14,
)


ax_top.text(
    cr_width + fa_width / 2,
    1.05,
    "False Alarm",
    ha="center",
    fontsize=14,
)

ax_top.set_yticks([])
ax_top.set_xticks([])
cbar = fig.colorbar(im, ax=[ax,ax_top])
cbar.set_label("Z-score")
ax.set_ylabel("Neurons")
ax.set_xlabel("Social touch - Nogo-Touch")
fig.savefig(os.path.join(output_dir, "Heatmap_normalisee_nogo_touch.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# =========================================================
# TRACER DE L'ACTIVATION DES NEURONES
# =========================================================
n_frames = dff.shape[1]
time_imaging = np.arange(n_frames) / sampling_rate
t_max = max(
    time_imaging[-1],
    vib_analog[-1][1] if len(vib_analog) > 0 else 0,
    social_analog[-1][1] if len(social_analog) > 0 else 0,
)
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1,6],
    hspace=0.05
)

ax_top = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_top)
ax_top.tick_params(axis="x", labelbottom=False)
ax.set_xlim(0, t_max)
ax_top.set_xlim(0, t_max)

# -------------------------------------------------
# TRI DES NEURONES PAR ACTIVATION MOYENNE
# -------------------------------------------------
activity = np.mean(dff, axis=1)
order = np.argsort(activity)
dff_sorted = dff[order,:]

# -------------------------------------------------
# TRACES DES NEURONES
# -------------------------------------------------
offset = 0.5
for neuron in range(dff_sorted.shape[0]):
    trace = dff_sorted[neuron,:]
    trace = trace + neuron * offset
    ax.plot(
        time_imaging,
        trace,
        linewidth=0.5
    )

ax.set_ylim(
    -0.5,
    dff_sorted.shape[0] * offset
)

ax.set_ylabel("Neurons")
ax.set_xlabel("Time (s)")

yticks = np.arange(
    0,
    dff_sorted.shape[0],
    10
)
ax.set_yticks(
    yticks * offset
)
ax.set_yticklabels(
    yticks
)

# -------------------------------------------------
# VIBRATIONS
# -------------------------------------------------
for start,end in vib_analog:
    ax_top.axvspan(
        start,
        end,
        color="green",
        alpha=0.75
    )

# -------------------------------------------------
# SOCIAL TOUCH
# -------------------------------------------------
# ---------------------------
# -0.3 premieres secondes
# ---------------------------
def remove_start_intervals(intervals, remove_start=0.3):
    trimmed_intervals = []
    for start, end in intervals:
        new_start = start + remove_start
        if new_start < end:
            trimmed_intervals.append((new_start, end))
    return trimmed_intervals

# Application aux social touch
social_analog_trimmed = remove_start_intervals(social_analog,remove_start=0.3)
for start, end in social_analog_trimmed:
    ax_top.hlines(
        y=0.3,
        xmin=start,
        xmax=end,
        color="blue",
        linewidth=6
    )
# -------------------------------------------------
# RECTANGLES EVENEMENTS
# -------------------------------------------------
# hauteur du bandeau supérieur uniquement
for timeout in go_timeouts:
    t_start,t_end = timeout[2]
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="pink",
            alpha=0.5
        )
    )

for t_start,t_end in projected_miss:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="gold",
            alpha=0.25
        )
    )

for t_start,t_end in projected_hit:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="turquoise",
            alpha=0.3
        )
    )

for t_start,t_end in projected_cr:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="limegreen",
            alpha=0.3
        )
    )

for false_alarm in false_alarms:
    t_start,t_end = false_alarm[2]
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="red",
            alpha=0.25
        )
    )

# -------------------------------------------------
# AXE SUPERIEUR
# -------------------------------------------------
ax_top.set_ylim(0,1)
ax_top.set_yticks([])
ax_top.set_xticks([])
fig.savefig(os.path.join(output_dir, "Traces_tous_neurones.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
# =========================================================
# TRACER DE L'ACTIVATION DE 10 NEURONES ALEATOIRES
# =========================================================
n_frames = dff.shape[1]
time_imaging = np.arange(n_frames) / sampling_rate

t_max = max(
    time_imaging[-1],
    vib_analog[-1][1] if len(vib_analog) > 0 else 0,
    social_analog[-1][1] if len(social_analog) > 0 else 0,
)

fig = plt.figure(figsize=(18, 10))

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1,6],
    hspace=0.05
)
ax_top = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], sharex=ax_top)
ax_top.tick_params(axis="x", labelbottom=False)
ax.set_xlim(0, t_max)
ax_top.set_xlim(0, t_max)

# -------------------------------------------------
# TRI DES NEURONES PAR ACTIVATION MOYENNE
# -------------------------------------------------
activity = np.mean(dff, axis=1)
order = np.argsort(activity)
dff_sorted = dff[order,:]

# -------------------------------------------------
# SOCIAL TOUCH
# -------------------------------------------------
# ---------------------------
# -0.3 premieres secondes
# ---------------------------
def remove_start_intervals(intervals, remove_start=0.3):
    trimmed_intervals = []
    for start, end in intervals:
        new_start = start + remove_start
        if new_start < end:
            trimmed_intervals.append((new_start, end))
    return trimmed_intervals

# Application aux social touch
social_analog_trimmed = remove_start_intervals(social_analog,remove_start=0.3)
for start, end in social_analog_trimmed:
    ax_top.hlines(
        y=0.3,
        xmin=start,
        xmax=end,
        color="blue",
        linewidth=6
    )

# -------------------------------------------------
# RECTANGLES EVENEMENTS
# -------------------------------------------------
for timeout in go_timeouts:
    t_start,t_end = timeout[2]
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="pink",
            alpha=0.5
        )
    )
for t_start,t_end in projected_miss:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="gold",
            alpha=0.25
        )
    )
for t_start,t_end in projected_hit:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="turquoise",
            alpha=0.3
        )
    )
for t_start,t_end in projected_cr:
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="limegreen",
            alpha=0.3
        )
    )
for false_alarm in false_alarms:
    t_start,t_end = false_alarm[2]
    ax_top.add_patch(
        Rectangle(
            (float(t_start),0),
            float(t_end-t_start),
            1,
            facecolor="red",
            alpha=0.25
        )
    )
# -------------------------------------------------
# AXE SUPERIEUR
# -------------------------------------------------
ax_top.set_ylim(0,1)
ax_top.set_yticks([])
ax_top.set_xticks([])
plt.show()

#--------------------------------------------------------
#Selection des neurones
#--------------------------------------------------------
def get_all_social_touch_intervals(projected_hit, projected_miss, go_timeouts,false_alarms, projected_cr, social_analog,remove_start=0.3):
    hit_social = get_social_from_trials(projected_hit, social_analog, remove_start)
    miss_social = get_social_from_trials(projected_miss, social_analog, remove_start)
    go_trials = [t[2] for t in go_timeouts]
    timeout_social = get_social_from_trials(go_trials, social_analog, remove_start)
    fa_social = get_social_from_trials(false_alarms, social_analog, remove_start)
    cr_social = get_social_from_trials(projected_cr, social_analog, remove_start)
    all_social = hit_social + miss_social + timeout_social + fa_social + cr_social
    print(f"Social touch retenus : Hit={len(hit_social)}, Miss={len(miss_social)}, "
          f"Timeout={len(timeout_social)}, FA={len(fa_social)}, CR={len(cr_social)} "
          f"-> total={len(all_social)}")
    return all_social

def build_behavior_mask(signal_length, intervals, sampling_rate):
    mask = np.zeros(signal_length)
    for start, end in intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        mask[i0:i1] = 1
    return mask

def compute_observed_similarity(dff, behavior_mask):
    n_neurons = dff.shape[0]
    similarities = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        trace = dff[neuron].astype(float)
        similarities[neuron] = 2 * np.dot(behavior_mask, trace) / (np.dot(behavior_mask, behavior_mask) + np.dot(trace, trace))
    return similarities

def compute_per_event_similarity(dff, intervals, sampling_rate):
    n_neurons, n_frames = dff.shape
    traces = dff.astype(np.float64)
    csum = np.zeros((n_neurons, n_frames + 1))
    np.cumsum(traces, axis=1, out=csum[:, 1:])
    ss = np.sum(traces ** 2, axis=1)
    similarities = np.zeros((len(intervals), n_neurons))
    for idx, (start, end) in enumerate(intervals):
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        dur = i1 - i0
        window_sum = csum[:, i1] - csum[:, i0]
        similarities[idx] = 2 * window_sum / (dur + ss)
    return similarities

def build_valid_shuffle_mask(signal_length, vib_intervals, social_intervals, sampling_rate):

    valid = np.ones(signal_length, dtype=bool)
    for intervals in (vib_intervals, social_intervals):
        for start, end in intervals:
            i0 = max(0, int(start * sampling_rate))
            i1 = min(signal_length, int(end * sampling_rate))
            valid[i0:i1] = False
    return valid

def get_valid_windows(valid_mask, min_length):
    windows = []
    n = len(valid_mask)
    i = 0
    while i < n:
        if valid_mask[i]:
            j = i
            while j < n and valid_mask[j]:
                j += 1
            if j - i >= min_length:
                windows.append((i, j))
            i = j
        else:
            i += 1
    return windows

def generate_shuffled_mask_excluding_events(durations_frames, valid_mask, rng):
    mask = np.zeros(len(valid_mask))
    available = valid_mask.copy()

    for dur in durations_frames:
        windows = get_valid_windows(available, dur)
        if not windows:
            raise RuntimeError(
                f"Pas assez d'espace hors vibration/social touch pour placer "
                f"un bloc de {dur} frames -- réduis n_shuffles ou vérifie "
                f"la densité des essais."
            )
        weights = np.array([w1 - w0 - dur + 1 for w0, w1 in windows], dtype=float)
        window_idx = rng.choice(len(windows), p=weights / weights.sum())
        w0, w1 = windows[window_idx]
        start = rng.integers(w0, w1 - dur + 1)
        mask[start:start + dur] = 1
        available[start:start + dur] = False
    return mask

def build_null_distribution_per_event(dff, intervals, vib_intervals, social_intervals,
                                       sampling_rate, n_shuffles=5000, seed=42):
    n_neurons, n_frames = dff.shape
    traces = dff.astype(np.float64)

    # Sommes cumulées (paddées) -> somme sur une fenêtre en O(1)
    csum = np.zeros((n_neurons, n_frames + 1))
    np.cumsum(traces, axis=1, out=csum[:, 1:])

    # dot(trace, trace) est constant quel que soit le shuffle -> calculé une seule fois
    ss = np.sum(traces ** 2, axis=1)  # shape (n_neurons,)

    rng = np.random.default_rng(seed)
    valid_mask = build_valid_shuffle_mask(n_frames, vib_intervals, social_intervals, sampling_rate)

    n_events = len(intervals)
    null_per_event = np.zeros((n_events, n_shuffles, n_neurons))

    for idx, (start, end) in enumerate(intervals):
        dur = int((end - start) * sampling_rate)
        windows = get_valid_windows(valid_mask, dur)
        if not windows:
            raise RuntimeError(f"Pas assez d'espace hors vibration/social touch pour l'essai {idx} "
                                f"(durée {dur} frames).")

        windows_arr = np.array(windows)
        weights = (windows_arr[:, 1] - windows_arr[:, 0] - dur + 1).astype(float)
        weights /= weights.sum()

        # Tirage vectorisé des n_shuffles en une fois
        window_idxs = rng.choice(len(windows), size=n_shuffles, p=weights)
        w0_arr = windows_arr[window_idxs, 0]
        w1_arr = windows_arr[window_idxs, 1]
        starts = rng.integers(w0_arr, w1_arr - dur + 1)
        ends = starts + dur

        # Somme de fenêtre pour tous les neurones/shuffles à la fois, sans boucle
        window_sums = (csum[:, ends] - csum[:, starts]).T  # (n_shuffles, n_neurons)

        denom = dur + ss[None, :]
        null_per_event[idx] = 2 * window_sums / denom

        if idx % 5 == 0:
            print(f"  Null distribution essai {idx}/{n_events}...")

    return null_per_event

def diagnose_per_event_significance(observed_per_event_similarity, null_per_event,
                                     low_pct=0.83, high_pct=99.17):
    n_events, n_neurons = observed_per_event_similarity.shape
    sig_exc = np.zeros((n_events, n_neurons), dtype=bool)
    sig_inh = np.zeros((n_events, n_neurons), dtype=bool)
    for idx in range(n_events):
        low_thresh  = np.percentile(null_per_event[idx], low_pct, axis=0)
        high_thresh = np.percentile(null_per_event[idx], high_pct, axis=0)
        sig_exc[idx] = observed_per_event_similarity[idx] > high_thresh
        sig_inh[idx] = observed_per_event_similarity[idx] < low_thresh
    n_sig_exc = sig_exc.sum(axis=0)
    n_sig_inh = sig_inh.sum(axis=0)
    print("Distribution du nb d'essais significatifs (excitateur) par neurone :")
    print(np.bincount(n_sig_exc))
    print("Distribution du nb d'essais significatifs (inhibiteur) par neurone :")
    print(np.bincount(n_sig_inh))
    return n_sig_exc, n_sig_inh


def classify_recruited_neurons_per_event(observed_per_event_similarity, null_per_event,min_events=2, low_pct=0.83, high_pct=99.17):
    n_events, n_neurons = observed_per_event_similarity.shape
    sig_exc = np.zeros((n_events, n_neurons), dtype=bool)
    sig_inh = np.zeros((n_events, n_neurons), dtype=bool)

    for idx in range(n_events):
        low_thresh  = np.percentile(null_per_event[idx], low_pct, axis=0)
        high_thresh = np.percentile(null_per_event[idx], high_pct, axis=0)
        sig_exc[idx] = observed_per_event_similarity[idx] > high_thresh
        sig_inh[idx] = observed_per_event_similarity[idx] < low_thresh

    n_sig_exc = sig_exc.sum(axis=0)
    n_sig_inh = sig_inh.sum(axis=0)

    classif = np.zeros(n_neurons, dtype=int)
    classif[n_sig_exc >= min_events] = 1
    classif[(n_sig_inh >= min_events) & (classif != 1)] = -1

    print(f"Recrutement confirmé (>= {min_events} essais significatifs) : "
          f"{np.sum(classif == 1)} excitateur(s), {np.sum(classif == -1)} inhibiteur(s)")
    return classif, n_sig_exc, n_sig_inh

def compute_thresholds(null_similarities, low_pct=0.83, high_pct=99.17):
    low_thresh  = np.percentile(null_similarities, low_pct, axis=0)
    high_thresh = np.percentile(null_similarities, high_pct, axis=0)
    return low_thresh, high_thresh


def build_trial_type_intervals(projected_hit, projected_miss, go_timeouts,false_alarms, projected_cr, social_analog,remove_start=0.3):
    hit_social  = get_social_from_trials(projected_hit, social_analog, remove_start)
    miss_social = get_social_from_trials(projected_miss, social_analog, remove_start)
    go_social   = get_social_from_trials([t[2] for t in go_timeouts], social_analog, remove_start)
    fa_social   = get_social_from_trials(false_alarms, social_analog, remove_start)
    cr_social   = get_social_from_trials(projected_cr, social_analog, remove_start)

    groups = {
        "Hit": hit_social,
        "Miss": miss_social,
        "Go-Timeout": go_social,
        "False Alarm": fa_social,
        "Correct Rejection": cr_social,
    }

    ordered_intervals = []
    boundaries = []        
    centers = {}   
    pos = 0
    for name, intervals in groups.items():
        n = len(intervals)
        ordered_intervals.extend(intervals)
        centers[name] = (pos + n / 2, n)
        pos += n
        if pos > 0:
            boundaries.append(pos)
        print(f"{name:20s}: {n} essais")
    boundaries = boundaries[:-1]
    return ordered_intervals, groups, boundaries, centers
#--------------------
#heatmap Z-score
#--------------------
def extract_social_activity_matrix(dff, social_intervals, sampling_rate):
    segments = []
    for start, end in social_intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        segments.append(np.mean(dff[:, i0:i1], axis=1))
    return np.array(segments)  # (n_essais, n_neurones)


def plot_recruited_vs_not_recruited(dff, all_social_intervals, neuron_classif, sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    not_recruited_idx = np.where(neuron_classif == 0)[0]
    dff_recruited = dff[recruited_idx]
    dff_not_recruited = dff[not_recruited_idx]
    activity_recruited = extract_social_activity_matrix(dff_recruited, all_social_intervals, sampling_rate).T
    activity_not_recruited = extract_social_activity_matrix(dff_not_recruited, all_social_intervals, sampling_rate).T
    order_r  = np.argsort(np.mean(activity_recruited, axis=1))
    order_nr = np.argsort(np.mean(activity_not_recruited, axis=1))
    activity_recruited = activity_recruited[order_r]
    activity_not_recruited = activity_not_recruited[order_nr]
    combined = np.concatenate([activity_recruited.flatten(), activity_not_recruited.flatten()])
    vmin, vmax = np.percentile(combined, [5, 99])
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    im1 = axes[0].imshow(activity_recruited, aspect="auto", cmap="RdPu",origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Neurones recrutés (n={len(recruited_idx)})")
    axes[0].set_ylabel("Neurones")
    im2 = axes[1].imshow(activity_not_recruited, aspect="auto", cmap="RdPu",origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Neurones non recrutés (n={len(not_recruited_idx)})")
    axes[1].set_ylabel("Neurones")
    axes[1].set_xlabel("Social Touch trials")
    fig.colorbar(im1, ax=axes.tolist(), fraction=0.025, pad=0.02, label="Z-score")
    plt.show()

def plot_recruited_vs_not_recruited_by_trial_type(dff, ordered_intervals, neuron_classif,sampling_rate, boundaries, centers):
    recruited_idx = np.where(neuron_classif != 0)[0]
    not_recruited_idx = np.where(neuron_classif == 0)[0]
    activity_recruited = extract_social_activity_matrix(dff[recruited_idx], ordered_intervals, sampling_rate).T
    activity_not_recruited = extract_social_activity_matrix(dff[not_recruited_idx], ordered_intervals, sampling_rate).T
    order_r = np.argsort(np.mean(activity_recruited, axis=1))
    order_nr = np.argsort(np.mean(activity_not_recruited, axis=1))
    activity_recruite = activity_recruited[order_r]
    activity_not_recruited = activity_not_recruited[order_nr]
    combined = np.concatenate([activity_recruited.flatten(), activity_not_recruited.flatten()])
    vmin, vmax = np.percentile(combined, [5, 99])
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    im1 = axes[0].imshow(activity_recruited, aspect="auto", cmap="RdPu",origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Neurones recrutés (n={len(recruited_idx)})")
    axes[0].set_ylabel("Neurones")
    im2 = axes[1].imshow(activity_not_recruited, aspect="auto", cmap="RdPu",origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Neurones non recrutés (n={len(not_recruited_idx)})")
    axes[1].set_ylabel("Neurones")
    for ax in axes:
        for x in boundaries:
            ax.axvline(x, color="black", linewidth=1.5)
    axes[1].set_xticks([c for c, n in centers.values()])
    axes[1].set_xticklabels([f"{name}\n(n={n})" for name, (c, n) in centers.items()])
    axes[1].set_xlabel("Type d'essai")
    fig.colorbar(im1, ax=axes.tolist(), fraction=0.025, pad=0.02, label="Z-score")
    plt.draw()

def plot_recruited_neurons_traces_st(dff, neuron_classif, auc_recruited, recruited_idx,vib_analog, social_analog, go_timeouts, projected_miss,projected_hit, projected_cr, false_alarms,sampling_rate, offset=2.0):
    n_frames = dff.shape[1]
    time_imaging = np.arange(n_frames) / sampling_rate

    t_max = max(
        time_imaging[-1],
        vib_analog[-1][1] if len(vib_analog) > 0 else 0,
        social_analog[-1][1] if len(social_analog) > 0 else 0,
    )

    mean_auc = np.nanmean(auc_recruited, axis=1)
    order = np.argsort(mean_auc)[::-1]
    sorted_idx = recruited_idx[order]
    sorted_types = neuron_classif[sorted_idx]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(0, t_max)
    ax_top.set_xlim(0, t_max)

    n_neurons = len(sorted_idx)
    for i, (neuron, resp_type) in enumerate(zip(sorted_idx, sorted_types)):
        trace = dff[neuron, :] + (n_neurons - 1 - i) * offset
        color = "crimson" if resp_type == 1 else "royalblue"
        ax.plot(time_imaging, trace, linewidth=0.6, color=color)

    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_ylabel("Neurones recrutés (triés par AUC)")
    ax.set_xlabel("Time (s)")
    yticks = np.arange(n_neurons)
    ax.set_yticks((n_neurons - 1 - yticks) * offset)
    ax.set_yticklabels(sorted_idx,fontsize=7)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="crimson", lw=2, label="Excitateur"),
        Line2D([0], [0], color="royalblue", lw=2, label="Inhibiteur"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    for start, end in vib_analog:
        ax_top.axvspan(start, end, color="green", alpha=0.75)
    social_analog_trimmed = remove_start_intervals(social_analog, remove_start=0.3)
    for start, end in social_analog_trimmed:
        ax_top.hlines(y=0.3, xmin=start, xmax=end, color="blue", linewidth=6)
    for timeout in go_timeouts:
        t_start, t_end = timeout[2]
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="pink", alpha=0.5))
    for t_start, t_end in projected_miss:
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="gold", alpha=0.25))
    for t_start, t_end in projected_hit:
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="turquoise", alpha=0.3))
    for t_start, t_end in projected_cr:
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="limegreen", alpha=0.3))
    for false_alarm in false_alarms:
        t_start, t_end = false_alarm[2]
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="red", alpha=0.25))
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_title(f"Activité des neurones recrutés (n={n_neurons})")
    plt.tight_layout()
    plt.draw()

def plot_recruited_neurons_traces(dff, neuron_classif, auc_recruited, recruited_idx,social_analog, sampling_rate,remove_start=0.3, offset=2.0):
    social_trimmed = remove_start_intervals(social_analog, remove_start=remove_start)
    mean_auc = np.nanmean(auc_recruited, axis=1)
    order = np.argsort(mean_auc)[::-1]
    sorted_idx = recruited_idx[order]
    sorted_types = neuron_classif[sorted_idx]
    n_neurons = len(sorted_idx)
    concatenated_traces = []
    segment_boundaries = []
    pos = 0.0
    first_pass = True
    for start, end in social_trimmed:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        duration = (i1 - i0) / sampling_rate
        if first_pass:
            for neuron in sorted_idx:
                concatenated_traces.append([dff[neuron, i0:i1]])
        else:
            for k, neuron in enumerate(sorted_idx):
                concatenated_traces[k].append(dff[neuron, i0:i1])
        pos += duration
        segment_boundaries.append(pos)
        first_pass = False
    concatenated_traces = [np.concatenate(segs) for segs in concatenated_traces]
    segment_boundaries = segment_boundaries[:-1]
    n_points = len(concatenated_traces[0])
    x_axis = np.linspace(0, pos, n_points)

    fig, ax = plt.subplots(figsize=(18, 10))
    for i, (trace, resp_type) in enumerate(zip(concatenated_traces, sorted_types)):
        shifted = trace + (n_neurons - 1 - i) * offset
        color = "crimson" if resp_type == 1 else "royalblue"
        ax.plot(x_axis, shifted, linewidth=0.6, color=color)
    for x in segment_boundaries:
        ax.axvline(x, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_ylim(-offset, n_neurons * offset)
    ax.set_xlim(0, pos)
    ax.set_xlabel("Temps cumulé pendant les Social Touch (s)")
    ax.set_ylabel("Neurones recrutés (triés par AUC)")
    ax.set_title(f"Activité des neurones recrutés pendant le Social Touch "
                 f"(n={n_neurons} neurones, {len(social_trimmed)} essais)")
    yticks = np.arange(n_neurons)
    ax.set_yticks((n_neurons - 1 - yticks) * offset)
    ax.set_yticklabels(sorted_idx,fontsize=7)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="crimson", lw=2, label="Excitateur"),
        Line2D([0], [0], color="royalblue", lw=2, label="Inhibiteur"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.draw()

def build_full_trial_dataset(events_col, time_col, in_trial_mask, json_params,vib_matches_excel_first, social_matches):
    events = np.asarray(events_col.fillna("").astype(str))
    mask   = np.asarray(in_trial_mask).astype(bool)
    time_col = np.asarray(time_col)
    trial_amp  = [item["amp"] for item in json_params]
    trial_type_list = [item["trial_type"] for item in json_params]
    trials = []
    in_trial = False
    trial_idx = -1
    trial_start_t = stim_start_t = stim_end_t = None
    has_reward = has_timeout = False
    for i in range(len(events)):
        if not mask[i]:
            continue
        ev = events[i]
        t = time_col[i]
        if ev == "New trial":
            in_trial = True
            trial_idx += 1
            trial_start_t = t
            stim_start_t = stim_end_t = None
            has_reward = has_timeout = False
            continue
        if not in_trial or trial_idx >= len(trial_amp):
            continue
        amp = trial_amp[trial_idx]
        ttype = trial_type_list[trial_idx]
        if ttype == "Nogo-Touch":
            amp = 0
        if ev == "Reward":
            has_reward = True
        if "Timeout" in ev:
            has_timeout = True
        if ev == "Stimulus" and amp > 0:
            stim_start_t = t
        elif ev == "StimOff" and stim_start_t is not None and amp > 0:
            stim_end_t = t
        if ev == "The trial ended":
            trial_end_t = t
            if amp > 0:
                if has_reward:
                    label = "Hit"
                elif has_timeout:
                    label = "Go-Timeout"
                else:
                    label = "Miss"
            else:
                label = "False Alarm" if has_timeout else "Correct Rejection"
            dt = None
            if stim_start_t is not None and stim_end_t is not None:
                for (ex_s, ex_e), (_, _), (dt_s, dt_e) in vib_matches_excel_first:
                    if abs(ex_s - stim_start_t) < 1e-6 and abs(ex_e - stim_end_t) < 1e-6:
                        dt = (dt_s, dt_e)
                        break
            if dt is None:
                for (ex_s, ex_e), (_, _), sdt in social_matches:
                    if trial_start_t <= ex_s <= trial_end_t:
                        dt = (sdt, sdt)
                        break
            trials.append({
                "trial_idx": trial_idx, "label": label,
                "excel_start": trial_start_t, "excel_end": trial_end_t,
                "dt": dt,
            })
            in_trial = False
    return trials

def filter_trials_with_social_touch(full_trials, social_excel_intervals):
    filtered = []
    for tr in full_trials:
        has_touch = any(
            tr["excel_start"] <= ex_s and ex_e <= tr["excel_end"]
            for ex_s, ex_e in social_excel_intervals
        )
        if has_touch:
            filtered.append(tr)
    return filtered

def plot_recruited_neurons_traces_full_trials(dff, neuron_classif, auc_recruited, recruited_idx,full_trials, social_excel_intervals, sampling_rate, offset=2.0):
    type_colors = {
        "Hit": "turquoise", "Miss": "gold", "Go-Timeout": "pink",
        "False Alarm": "red", "Correct Rejection": "limegreen", "None": "white",
    }
    label_order = ["Hit", "Miss", "Go-Timeout", "False Alarm", "Correct Rejection", "None"]
    mean_auc = np.nanmean(auc_recruited, axis=1)
    order = np.argsort(mean_auc)[::-1]
    sorted_idx = recruited_idx[order]
    sorted_types = neuron_classif[sorted_idx]
    n_neurons = len(sorted_idx)
    trials_by_label = {lbl: [] for lbl in label_order}
    for tr in full_trials:
        trials_by_label[tr["label"] if tr["dt"] is not None else "None"].append(tr)
    concatenated_traces = [[] for _ in sorted_idx]
    group_spans = []
    touch_marks = []
    pos = 0.0
    for label in label_order:
        group_start = pos
        for tr in trials_by_label[label]:
            if tr["dt"] is None:
                continue
            dt_s, dt_e = tr["dt"]
            an_start = tr["excel_start"] + dt_s
            an_end   = tr["excel_end"] + dt_e
            i0, i1 = int(an_start * sampling_rate), int(an_end * sampling_rate)
            if i1 <= i0:
                continue
            trial_duration = (i1 - i0) / sampling_rate
            for ex_s, ex_e in social_excel_intervals:
                if tr["excel_start"] <= ex_s and ex_e <= tr["excel_end"]:
                    ts, te = ex_s + dt_s, ex_e + dt_e
                    rel_s = pos + max(0.0, ts - an_start)
                    rel_e = pos + min(trial_duration, te - an_start)
                    if rel_e > rel_s:
                        touch_marks.append((rel_s, rel_e))
                    break
            for k, neuron in enumerate(sorted_idx):
                concatenated_traces[k].append(dff[neuron, i0:i1])
            pos += trial_duration
        if pos > group_start:
            group_spans.append((label, group_start, pos))
    if pos == 0.0:
        print("Aucun essai projetable trouvé.")
        return
    concatenated_traces = [np.concatenate(s) for s in concatenated_traces]
    x_axis = np.linspace(0, pos, len(concatenated_traces[0]))
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(0, pos)
    ax_top.set_xlim(0, pos)
    for i, (trace, resp_type) in enumerate(zip(concatenated_traces, sorted_types)):
        shifted = trace + (n_neurons - 1 - i) * offset
        color = "crimson" if resp_type == 1 else "royalblue"
        ax.plot(x_axis, shifted, linewidth=0.6, color=color)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_ylabel("Neurones recrutés (triés par AUC)")
    ax.set_xlabel("Temps cumulé, essai complet (s)")
    yticks = np.arange(n_neurons)
    ax.set_yticks((n_neurons - 1 - yticks) * offset)
    ax.set_yticklabels(sorted_idx, fontsize=7)

    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([0], [0], color="crimson", lw=2, label="Excitateur"),
        Line2D([0], [0], color="royalblue", lw=2, label="Inhibiteur"),
    ], loc="upper right")
    for label, x_start, x_end in group_spans:
        if label != "None":
            ax_top.add_patch(Rectangle((x_start, 0), x_end - x_start, 1,
                                        facecolor=type_colors[label], alpha=0.4, edgecolor="none"))
        ax_top.text((x_start + x_end) / 2, 1.15, label, ha="center", fontsize=10)
    for x_start, x_end in touch_marks:
        ax_top.hlines(y=0.3, xmin=x_start, xmax=x_end, color="blue", linewidth=6)
    ax_top.set_ylim(0, 1.3)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_title(f"Activité des neurones recrutés — essais complets (n={n_neurons} neurones)")
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.draw()

def plot_mean_zscore_all_neurons(dff, social_intervals, sampling_rate):
    activity = []
    start_times = []
    for start, end in social_intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        activity.append(np.mean(dff[:, i0:i1], axis=1))
        start_times.append(start)
    activity = np.array(activity).T
    order = np.argsort(np.mean(activity, axis=1))
    activity = activity[order]
    vmin = np.percentile(activity, 5)
    vmax = np.percentile(activity, 99)
    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(
        activity,
        aspect="auto",
        origin="lower",
        cmap="RdPu",
        vmin=vmin,
        vmax=vmax
    )
    n_ticks = 10
    tick_pos = np.linspace(
        0,
        len(start_times) - 1,
        min(n_ticks, len(start_times)),
        dtype=int
    )
    tick_labels = [f"{start_times[i]:.0f}" for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Neurones")
    ax.set_title("Activité moyenne des neurones pendant chaque Social Touch")
    cbar = plt.colorbar(im)
    cbar.set_label("Z-score moyen")
    plt.tight_layout()
    plt.draw()

#--------------------------------------------------------
#AUC des neurones recrutes - social touch
#--------------------------------------------------------
def filter_valid_intervals(intervals, sampling_rate, min_duration_s=0.05):
    min_frames = max(2, int(min_duration_s * sampling_rate))
    filtered = []
    removed = 0
    for start, end in intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 - i0 >= min_frames:
            filtered.append((start, end))
        else:
            removed += 1
    if removed > 0:
        print(f"{removed} intervalle(s) trop court(s) retiré(s) "
              f"(< {min_frames} frames, {min_duration_s*1000:.0f}ms)")
    return filtered

def compute_auc_recruited_social(dff,social_intervals,neuron_classif,sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_trials = len(social_intervals)
    auc = np.full((len(recruited_idx), n_trials),np.nan)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        for trial, (start, end) in enumerate(social_intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            signal = dff[neuron, i0:i1].copy()
            if signal.size < 2:
                continue
            if response_type == 1:
                signal[signal < 0] = 0
            elif response_type == -1:
                signal[signal > 0] = 0
            auc[i, trial] = trapezoid(signal,dx=1 / sampling_rate)
    return auc, recruited_idx
#--------------------------------------------------------
#Heatmap de l'AUC - social touch
#--------------------------------------------------------
def plot_auc_recruited_heatmap(auc, recruited_idx):
    mean_auc = np.nanmean(np.abs(auc), axis=1)
    order = np.argsort(mean_auc)[::-1]
    auc_sorted = auc[order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        auc_sorted, cmap = "coolwarm", center = 0,
        mask = np.isnan(auc_sorted), xticklabels = False, yticklabels = False,
    )
    plt.xlabel("Social Touch trials")
    plt.ylabel(f"Neurones recrutés (n={len(recruited_idx)})")
    plt.title("AUC Social-touch signée (rouge = excité, bleu = inhibé)")
    plt.tight_layout()
    plt.draw()

def plot_auc_recruited_heatmap_grouped(auc, recruited_idx, neuron_classif):
    types = neuron_classif[recruited_idx]
    mean_auc = np.nanmean(auc, axis=1) 

    exc_rows = np.where(types == 1)[0]
    inh_rows = np.where(types == -1)[0]

    exc_order = exc_rows[np.argsort(mean_auc[exc_rows])[::-1]]
    inh_order = inh_rows[np.argsort(mean_auc[inh_rows])]

    order = np.concatenate([exc_order, inh_order])
    auc_sorted = auc[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        auc_sorted, cmap="coolwarm", center=0,
        mask=np.isnan(auc_sorted), xticklabels=False, yticklabels=False,
        ax=ax,
    )
    # ligne de séparation nette entre les deux groupes
    ax.axhline(len(exc_order), color="black", linewidth=2)

    ax.set_xlabel("Social Touch trials")
    ax.set_ylabel(f"Activés (n={len(exc_order)}) / Inhibés (n={len(inh_order)})")
    ax.set_title("AUC Social-touch — rouge = activé, bleu = inhibé")
    plt.tight_layout()
    plt.draw()

def plot_auc_by_trial_type(auc, recruited_idx, neuron_classif, boundaries, centers):
    mean_auc = np.nanmean(auc, axis=1)
    order = np.argsort(mean_auc)[::-1]
    auc_sorted = auc[order]
    n_exc = np.sum(mean_auc[order] > 0)
    finite_vals = auc_sorted[~np.isnan(auc_sorted)]
    vmax = np.percentile(np.abs(finite_vals), 95)
    vmin = -vmax
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        auc_sorted, cmap="coolwarm", center=0,
        vmin=vmin, vmax=vmax,
        mask=np.isnan(auc_sorted), xticklabels=False, yticklabels=False, ax=ax,
    )
    ax.axhline(n_exc, color="black", linewidth=2)
    for x in boundaries:
        ax.axvline(x, color="black", linewidth=1.5)
    ax.set_xticks([c for c, n in centers.values()])
    ax.set_xticklabels([f"{name}\n(n={n})" for name, (c, n) in centers.items()])
    ax.set_ylabel(f"Activés (n={n_exc}) / Inhibés (n={len(order)-n_exc}), "
                  f"triés par AUC décroissante")
    ax.set_xlabel("Type d'essai")
    ax.set_title("AUC Social-touch — rouge = activé, bleu = inhibé, triés par intensité de réponse\nWT")
    plt.tight_layout()
    #plt.draw()

def check_nan_fraction(auc, recruited_idx, neuron_classif):
    types = neuron_classif[recruited_idx]
    frac_nan = np.mean(np.isnan(auc), axis=1)

    print(f"Fraction NaN moyenne (excitateurs) : {frac_nan[types == 1].mean()*100:.1f}%")
    print(f"Fraction NaN moyenne (inhibiteurs)  : {frac_nan[types == -1].mean()*100:.1f}%")
    print(f"Fraction NaN globale : {frac_nan.mean()*100:.1f}%")

    suspects = recruited_idx[frac_nan > 0.9]
    print(f"\nNeurones recrutés mais NaN sur >90% des essais : {len(suspects)}")
    print(suspects)


def compute_auc_recruited(dff,intervals,neuron_classif,sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    auc = np.full((len(recruited_idx), len(intervals)), np.nan)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        for j, (start, end) in enumerate(intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            signal = dff[neuron, i0:i1].copy()
            if signal.size < 2:
                continue
            if response_type == 1:
                signal[signal < 0] = 0
            elif response_type == -1:
                signal[signal > 0] = 0
            auc[i, j] = trapezoid(signal, dx=1/sampling_rate)
    return auc, recruited_idx
#--------------------------------------------------------
#AUC neurones recrutes - vibrations
#--------------------------------------------------------
def build_vibration_windows(vib_analog, duration=0.5):
    return [(start, start + duration) for start, _ in vib_analog]

def get_vibrations_from_social_trials(social_trials, vib_analog):
    selected = []
    for trial in social_trials:
        if len(trial) == 2:
            trial_start, trial_end = trial
        elif len(trial) == 4:
            trial_start, trial_end = trial[2]
        else:
            continue
        for vib_start, vib_end in vib_analog:
            if trial_start <= vib_start and vib_end <= trial_end:
                selected.append((vib_start, vib_end))
    return selected


def compute_auc_recruited_vibrations(dff,vib_intervals,neuron_classif,sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    auc = np.full((len(recruited_idx), len(vib_intervals)), np.nan)
    for i, neuron in enumerate(recruited_idx):
        for j, (start, end) in enumerate(vib_intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            if i1 <= i0:
                continue
            signal = dff[neuron, i0:i1]
            if len(signal) < 2:
                continue
            auc[i, j] = trapezoid(signal, dx=1/sampling_rate)
    return auc, recruited_idx

def plot_auc_heatmap(auc,recruited_idx,neuron_classif):
    types = neuron_classif[recruited_idx]
    mean_auc = np.nanmean(auc, axis=1)
    exc = np.where(types == 1)[0]
    inh = np.where(types == -1)[0]
    exc = exc[np.argsort(mean_auc[exc])[::-1]]
    inh = inh[np.argsort(mean_auc[inh])]
    order = np.concatenate([exc, inh])
    auc = auc[order]
    vmax = np.nanpercentile(np.abs(auc),95)
    plt.figure(figsize=(12,8))
    sns.heatmap(
        auc,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        mask=np.isnan(auc)
    )
    plt.axhline(len(exc),color="black",linewidth=2)
    plt.xlabel("Vibrations pendant un Social Touch")
    plt.ylabel("Neurones recrutés")
    plt.title("AUC des neurones recrutés pendant les vibrations")
    plt.tight_layout()
    #plt.draw()

def plot_recruited_neurons_traces_vibration(dff,neuron_classif,auc_recruited,recruited_idx,vibration_intervals,sampling_rate,offset=2.0):
    mean_auc = np.nanmean(auc_recruited, axis=1)
    order = np.argsort(mean_auc)[::-1]
    sorted_idx = recruited_idx[order]
    sorted_types = neuron_classif[sorted_idx]
    n_neurons = len(sorted_idx)
    concatenated_traces = [[] for _ in sorted_idx]
    segment_boundaries = []
    pos = 0.0
    for start, end in vibration_intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        for k, neuron in enumerate(sorted_idx):
            concatenated_traces[k].append(dff[neuron, i0:i1])
        pos += (i1 - i0) / sampling_rate
        segment_boundaries.append(pos)
    if pos == 0:
        print("Aucune vibration à afficher.")
        return
    concatenated_traces = [np.concatenate(x) for x in concatenated_traces]
    segment_boundaries = segment_boundaries[:-1]
    x = np.linspace(0, pos, len(concatenated_traces[0]))
    fig, ax = plt.subplots(figsize=(18,10))
    for i, (trace, resp_type) in enumerate(zip(concatenated_traces, sorted_types)):
        shifted = trace + (n_neurons-1-i)*offset
        color = "crimson" if resp_type == 1 else "royalblue"
        ax.plot(x, shifted, color=color, linewidth=0.6)
    for b in segment_boundaries:
        ax.axvline(b, color="gray", alpha=0.4, linewidth=0.5)
    yticks = np.arange(n_neurons)
    ax.set_yticks((n_neurons-1-yticks)*offset)
    ax.set_yticklabels(sorted_idx, fontsize=7)
    ax.set_xlim(0, pos)
    ax.set_xlabel("Temps cumulé pendant les vibrations (s)")
    ax.set_ylabel("Neurones recrutés")
    ax.set_title(
        f"Activité des neurones recrutés pendant les vibrations (n={n_neurons})"
    )

    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([0],[0],color="crimson",lw=2,label="Excitateur"),
        Line2D([0],[0],color="royalblue",lw=2,label="Inhibiteur")
    ])
    plt.tight_layout()
    #plt.draw()

# -----------------------------------------------
# VIBRATIONS CLASSEES PAR TYPE D'ESSAI
# -----------------------------------------------
def extract_trial_interval(trial):
    if len(trial) == 2:
        return trial
    if len(trial) >= 3:
        return trial[2]

    return None
def vibrations_in_trials(trials):
    selected = []
    for trial in trials:
        interval = extract_trial_interval(trial)
        if interval is None:
            continue
        trial_start, trial_end = interval
        for vib_start, vib_end in vib_analog:
            if trial_start <= vib_start and vib_end <= trial_end:
                selected.append((vib_start, vib_end))
    return selected
def build_trial_type_vibration_intervals(projected_hit,projected_miss,go_timeouts,false_alarms,projected_cr,vib_analog):
    hit_vib = vibrations_in_trials(projected_hit)
    miss_vib = vibrations_in_trials(projected_miss)
    timeout_trials = [t[2] for t in go_timeouts]
    timeout_vib = vibrations_in_trials(timeout_trials)
    fa_vib = vibrations_in_trials(false_alarms)
    cr_vib = vibrations_in_trials(projected_cr)
    groups = [
        hit_vib,
        miss_vib,
        timeout_vib,
        fa_vib,
        cr_vib
    ]
    labels = [
        "Hit",
        "Miss",
        "Go Timeout",
        "False Alarm",
        "Correct Rejection"
    ]
    ordered_intervals = []
    boundaries = []
    centers = []
    pos = 0
    for g in groups:
        ordered_intervals.extend(g)
        pos += len(g)
        boundaries.append(pos)
        centers.append(pos - len(g)/2)
    print(f"Hit               : {len(hit_vib)} vibrations")
    print(f"Miss              : {len(miss_vib)} vibrations")
    print(f"Go Timeout        : {len(timeout_vib)} vibrations")
    print(f"False Alarm       : {len(fa_vib)} vibrations")
    print(f"Correct Rejection : {len(cr_vib)} vibrations")
    return ordered_intervals, labels, boundaries[:-1], centers

def plot_auc_vibration_by_trial_type(auc,recruited_idx,neuron_classif,n_hit,n_miss):
    mean_auc = np.nanmean(auc, axis=1)
    exc = np.where(neuron_classif[recruited_idx] == 1)[0]
    inh = np.where(neuron_classif[recruited_idx] == -1)[0]
    exc = exc[np.argsort(mean_auc[exc])[::-1]]
    inh = inh[np.argsort(mean_auc[inh])]
    order = np.concatenate([exc, inh])
    auc = auc[order]
    recruited_idx = recruited_idx[order]
    finite = auc[np.isfinite(auc)]
    if finite.size == 0:
        print("Aucune AUC valide.")
        return
    vmax = np.percentile(np.abs(finite),95)
    plt.figure(figsize=(14,8))
    ax = sns.heatmap(
        auc,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label":"AUC"},
        yticklabels=recruited_idx,
        xticklabels=False
    )
    ax.axhline(len(exc), color="black", lw=2)
    ax.axvline(n_hit, color="black", lw=2)

    centers = [
        n_hit/2,
        n_hit + n_miss/2
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [f"Hit (n = {n_hit})",f"Miss (n = {n_miss})"],
        fontsize=12
    )
    step = max(1, len(recruited_idx)//15)
    yticks = np.arange(0, len(recruited_idx), step)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(
        recruited_idx[yticks],
        fontsize=8
    )
    ax.set_xlabel("Type d'essai")
    ax.set_ylabel("Neurones recrutés")
    ax.set_title("AUC des neurones recrutés des social-touch pendant les vibrations")
    plt.tight_layout()
    #plt.draw()
#--------------------------------------------------------
#test
#--------------------------------------------------------
all_social_intervals = get_all_social_touch_intervals(projected_hit, projected_miss, go_timeouts, false_alarms, projected_cr,social_analog, remove_start=0.3)
behavior_mask = build_behavior_mask(dff.shape[1], all_social_intervals, sampling_rate)
observed_similarity = compute_observed_similarity(dff, behavior_mask)
per_event_similarity = compute_per_event_similarity(dff, all_social_intervals, sampling_rate)
print("\n=== SHUFFLE PAR ESSAI (restreint aux périodes sans vibration ni social touch) ===")
null_per_event = build_null_distribution_per_event(dff, all_social_intervals, vib_analog, social_analog,sampling_rate, n_shuffles=5000, seed=42)
neuron_classif, n_sig_exc, n_sig_inh = classify_recruited_neurons_per_event(per_event_similarity, null_per_event, min_events=4, low_pct=0.83, high_pct=99.17)
print("Neurones excitateurs recrutés :", np.sum(neuron_classif == 1))
print("Neurones inhibiteurs recrutés :", np.sum(neuron_classif == -1))
print("Neurones non recrutés         :", np.sum(neuron_classif == 0))
# =========================================================
# TABLEAU RECAPITULATIF DU RECRUTEMENT
# =========================================================
def summarize_recruitment(neuron_classif, label):
    n_exc = int(np.sum(neuron_classif == 1))
    n_inh = int(np.sum(neuron_classif == -1))
    n_rec = n_exc + n_inh
    n_tot = len(neuron_classif)
    return {
        "Condition": label,
        "Neurones totaux": n_tot,
        "Recrutés (total)": n_rec,
        "Excitateurs": n_exc,
        "Inhibiteurs": n_inh,
        "Non recrutés": n_tot - n_rec,
    }
summary_comportement = pd.DataFrame([
    summarize_recruitment(neuron_classif, "Comportement (tous social touch)"),
])
print("\n=== RECAPITULATIF RECRUTEMENT — COMPORTEMENT ===")
print(summary_comportement.to_string(index=False))
summary_comportement.to_excel(os.path.join(output_dir, "recap_recrutement_comportement.xlsx"), index=False)

all_social_intervals = filter_valid_intervals(all_social_intervals, sampling_rate, min_duration_s=0.05)
auc_recruited, recruited_idx = compute_auc_recruited_social(dff, all_social_intervals, neuron_classif, sampling_rate)
plot_auc_recruited_heatmap_grouped(auc_recruited, recruited_idx, neuron_classif)
plot_auc_recruited_heatmap(auc_recruited, recruited_idx)
plot_recruited_vs_not_recruited(dff, all_social_intervals, neuron_classif, sampling_rate)
check_nan_fraction(auc_recruited, recruited_idx, neuron_classif)
ordered_intervals, groups, boundaries, centers = build_trial_type_intervals(projected_hit, projected_miss, go_timeouts, false_alarms, projected_cr,social_analog, remove_start=0.3)
auc_recruited, recruited_idx = compute_auc_recruited_social(dff, ordered_intervals, neuron_classif, sampling_rate)
plot_auc_by_trial_type(auc_recruited, recruited_idx, neuron_classif, boundaries, centers)
plot_recruited_vs_not_recruited_by_trial_type(dff, ordered_intervals, neuron_classif,sampling_rate, boundaries, centers)
plot_recruited_neurons_traces_st(dff, neuron_classif, auc_recruited, recruited_idx,vib_analog, social_analog, go_timeouts, projected_miss,projected_hit, projected_cr, false_alarms,sampling_rate, offset=2.0)
plot_recruited_neurons_traces(dff, neuron_classif, auc_recruited, recruited_idx,social_analog, sampling_rate, remove_start=0.3, offset=2.0)
full_trials = build_full_trial_dataset(events_col, time_col, in_trial_mask, js,vib_matches_excel_first, social_matches)
full_trials_with_touch = filter_trials_with_social_touch(full_trials, social_excel)
plot_recruited_neurons_traces_full_trials(dff, neuron_classif, auc_recruited, recruited_idx,full_trials_with_touch, social_excel, sampling_rate, offset=2.0)
plot_mean_zscore_all_neurons(dff,all_social_intervals,sampling_rate)
# ----------------------------------Vibrations--------------------------------------
hit_vib = get_vibrations_from_social_trials(projected_hit, vib_analog)
miss_vib = get_vibrations_from_social_trials(projected_miss, vib_analog)
timeout_vib = get_vibrations_from_social_trials([t[2] for t in go_timeouts],vib_analog)
fa_vib = get_vibrations_from_social_trials(false_alarms, vib_analog)
cr_vib = get_vibrations_from_social_trials(projected_cr, vib_analog)
vibration_social = (
    hit_vib
    + miss_vib
    + timeout_vib
    + fa_vib
    + cr_vib
)

print("Nombre de vibrations dans les essais Social Touch :", len(vibration_social))
print("Nombre de vibrations sélectionnées :", len(vibration_social))
print(vibration_social[:10])
for start, end in vibration_social[:10]:
    print(start, end, end-start)
vibration_windows = build_vibration_windows(vib_analog, duration=0.5)

auc_vibration, recruited_idx = compute_auc_recruited_vibrations(
    dff,
    vibration_windows,
    neuron_classif,
    sampling_rate
)
ordered_vibrations = (
    hit_vib +
    miss_vib +
    timeout_vib +
    fa_vib +
    cr_vib
)
boundaries = np.cumsum([
    len(hit_vib),
    len(miss_vib),
    len(timeout_vib),
    len(fa_vib)
])
centers = [
    len(hit_vib)/2,
    len(hit_vib)+len(miss_vib)/2,
    len(hit_vib)+len(miss_vib)+len(timeout_vib)/2,
    len(hit_vib)+len(miss_vib)+len(timeout_vib)+len(fa_vib)/2,
    len(hit_vib)+len(miss_vib)+len(timeout_vib)+len(fa_vib)+len(cr_vib)/2
]
print(auc_vibration.shape)
print(np.sum(~np.isnan(auc_vibration)))
plot_auc_recruited_heatmap_grouped(
    auc_vibration,
    recruited_idx,
    neuron_classif
)

plot_recruited_neurons_traces_vibration(
    dff,
    neuron_classif,
    auc_vibration,
    recruited_idx,
    vibration_social,
    sampling_rate
)
hit_vib_windows = build_vibration_windows(hit_vib, duration=0.5)
miss_vib_windows = build_vibration_windows(miss_vib, duration=0.5)
timeout_vib_windows = build_vibration_windows(timeout_vib, duration=0.5)
fa_vib_windows = build_vibration_windows(fa_vib, duration=0.5)
cr_vib_windows = build_vibration_windows(cr_vib, duration=0.5)
ordered_vib = (
    hit_vib_windows
    + miss_vib_windows
    + timeout_vib_windows
    + fa_vib_windows
    + cr_vib_windows
)
n_hit = len(hit_vib_windows)
n_miss = len(miss_vib_windows)
n_timeout = len(timeout_vib_windows)
boundaries = np.cumsum([
    len(hit_vib_windows),
    len(miss_vib_windows),
    len(timeout_vib_windows),
    len(fa_vib_windows)
])

centers = [
    len(hit_vib_windows)/2,
    len(hit_vib_windows)+len(miss_vib_windows)/2,
    len(hit_vib_windows)+len(miss_vib_windows)+len(timeout_vib_windows)/2,
    len(hit_vib_windows)+len(miss_vib_windows)+len(timeout_vib_windows)+len(fa_vib_windows)/2,
    len(hit_vib_windows)+len(miss_vib_windows)+len(timeout_vib_windows)+len(fa_vib_windows)+len(cr_vib_windows)/2
]
plot_auc_vibration_by_trial_type(
    auc_vibration,
    recruited_idx,
    neuron_classif,
    len(hit_vib_windows),
    len(miss_vib_windows)
)

# ===================================================================================
# Peak amplitude
# ===================================================================================
def compute_peak_amplitude_recruited_social(dff, social_intervals, neuron_classif, sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_trials = len(social_intervals)
    peak = np.full((len(recruited_idx), n_trials), np.nan)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        for trial, (start, end) in enumerate(social_intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            signal = dff[neuron, i0:i1]
            if signal.size < 2:
                continue
            if response_type == 1:
                peak[i, trial] = np.max(signal)
            elif response_type == -1:
                peak[i, trial] = np.min(signal)
    return peak, recruited_idx

def compute_peak_amplitude_recruited_social(dff, social_intervals, neuron_classif, sampling_rate,baseline_duration=0.5):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_trials = len(social_intervals)
    peak = np.full((len(recruited_idx), n_trials), np.nan)
    baseline_frames = int(baseline_duration * sampling_rate)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        trace = dff[neuron]
        for trial, (start, end) in enumerate(social_intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            signal = trace[i0:i1]
            if signal.size < 2:
                continue
            baseline_start = max(0, i0 - baseline_frames)
            baseline = trace[baseline_start:i0]
            if baseline.size < 2:
                continue
            baseline_mean = baseline.mean()
            corrected = signal - baseline_mean
            if response_type == 1:
                peak[i, trial] = np.max(corrected)
            elif response_type == -1:
                peak[i, trial] = np.min(corrected)
    return peak, recruited_idx

def plot_peak_amplitude_heatmap(peak, recruited_idx, neuron_classif, xlabel="Social touch trials"):
    types = neuron_classif[recruited_idx]
    mean_peak = np.nanmean(peak, axis=1)

    exc_rows = np.where(types == 1)[0]
    inh_rows = np.where(types == -1)[0]
    exc_order = exc_rows[np.argsort(mean_peak[exc_rows])[::-1]]
    inh_order = inh_rows[np.argsort(mean_peak[inh_rows])]
    order = np.concatenate([exc_order, inh_order])
    peak_sorted = peak[order]

    finite_vals = peak_sorted[~np.isnan(peak_sorted)]
    if finite_vals.size == 0:
        print("Aucune valeur de peak amplitude valide.")
        return None
    vmax = np.percentile(np.abs(finite_vals), 95)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        peak_sorted, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
        mask=np.isnan(peak_sorted), xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Peak amplitude (dF/F - baseline)"}, ax=ax,
    )
    ax.axhline(len(exc_order), color="black", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Excitateurs (n={len(exc_order)}, haut) / Inhibiteurs (n={len(inh_order)}, bas)")
    ax.set_title("Peak amplitude relatif à la baseline — rouge = excitateur, bleu = inhibiteur")
    plt.tight_layout()
    return fig

peak_recruited, recruited_idx = compute_peak_amplitude_recruited_social(dff, all_social_intervals, neuron_classif, sampling_rate, baseline_duration=0.5)
plot_peak_amplitude_heatmap(peak_recruited, recruited_idx, neuron_classif)


all_social_intervals = filter_valid_intervals(all_social_intervals, sampling_rate, min_duration_s=0.05)
auc_recruited, recruited_idx = compute_auc_recruited_social(dff, all_social_intervals, neuron_classif, sampling_rate)

plot_auc_recruited_heatmap_grouped(auc_recruited, recruited_idx, neuron_classif)
save_current_fig("AUC_recrutement_groupe")

plot_auc_recruited_heatmap(auc_recruited, recruited_idx)
save_current_fig("AUC_recrutement")

plot_recruited_vs_not_recruited(dff, all_social_intervals, neuron_classif, sampling_rate)
save_current_fig("Recrutes_vs_non_recrutes")

check_nan_fraction(auc_recruited, recruited_idx, neuron_classif)

ordered_intervals, groups, boundaries, centers = build_trial_type_intervals(projected_hit, projected_miss, go_timeouts, false_alarms, projected_cr, social_analog, remove_start=0.3)
auc_recruited, recruited_idx = compute_auc_recruited_social(dff, ordered_intervals, neuron_classif, sampling_rate)

plot_auc_by_trial_type(auc_recruited, recruited_idx, neuron_classif, boundaries, centers)
save_current_fig("AUC_par_type_essai")

plot_recruited_vs_not_recruited_by_trial_type(dff, ordered_intervals, neuron_classif, sampling_rate, boundaries, centers)
save_current_fig("Recrutes_vs_non_recrutes_par_type_essai")

plot_recruited_neurons_traces_st(dff, neuron_classif, auc_recruited, recruited_idx, vib_analog, social_analog, go_timeouts, projected_miss, projected_hit, projected_cr, false_alarms, sampling_rate, offset=2.0)
save_current_fig("Traces_neurones_recrutes_st")

plot_recruited_neurons_traces(dff, neuron_classif, auc_recruited, recruited_idx, social_analog, sampling_rate, remove_start=0.3, offset=2.0)
save_current_fig("Traces_neurones_recrutes_social_touch")

full_trials = build_full_trial_dataset(events_col, time_col, in_trial_mask, js, vib_matches_excel_first, social_matches)
full_trials_with_touch = filter_trials_with_social_touch(full_trials, social_excel)

plot_recruited_neurons_traces_full_trials(dff, neuron_classif, auc_recruited, recruited_idx, full_trials_with_touch, social_excel, sampling_rate, offset=2.0)
save_current_fig("Traces_neurones_recrutes_essais_complets")

plot_mean_zscore_all_neurons(dff, all_social_intervals, sampling_rate)
save_current_fig("Zscore_moyen_tous_neurones")

# ----------------------------------Vibrations--------------------------------------
hit_vib = get_vibrations_from_social_trials(projected_hit, vib_analog)
miss_vib = get_vibrations_from_social_trials(projected_miss, vib_analog)
timeout_vib = get_vibrations_from_social_trials([t[2] for t in go_timeouts], vib_analog)
fa_vib = get_vibrations_from_social_trials(false_alarms, vib_analog)
cr_vib = get_vibrations_from_social_trials(projected_cr, vib_analog)
vibration_social = hit_vib + miss_vib + timeout_vib + fa_vib + cr_vib

vibration_windows = build_vibration_windows(vib_analog, duration=0.5)
auc_vibration, recruited_idx = compute_auc_recruited_vibrations(dff, vibration_windows, neuron_classif, sampling_rate)

plot_auc_recruited_heatmap_grouped(auc_vibration, recruited_idx, neuron_classif)
save_current_fig("AUC_vibrations_recrutement_groupe")

plot_recruited_neurons_traces_vibration(dff, neuron_classif, auc_vibration, recruited_idx, vibration_social, sampling_rate)
save_current_fig("Traces_neurones_recrutes_vibrations")

hit_vib_windows = build_vibration_windows(hit_vib, duration=0.5)
miss_vib_windows = build_vibration_windows(miss_vib, duration=0.5)

plot_auc_vibration_by_trial_type(auc_vibration, recruited_idx, neuron_classif, len(hit_vib_windows), len(miss_vib_windows))
save_current_fig("AUC_vibrations_par_type_essai")

# ===================================================================================
# Peak amplitude
# ===================================================================================
peak_recruited, recruited_idx = compute_peak_amplitude_recruited_social(dff, all_social_intervals, neuron_classif, sampling_rate, baseline_duration=0.5)
fig_peak = plot_peak_amplitude_heatmap(peak_recruited, recruited_idx, neuron_classif)
if fig_peak is not None:
    fig_peak.savefig(os.path.join(output_dir, "Peak_amplitude.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_peak)

plt.show()

