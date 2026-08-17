import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import seaborn as sns
from scipy.stats import linregress
import os
####################################################################### test en avance de 3 sur habituation ###############################################################
# =========================================================
# PATHS
# =========================================================
base_dir      = r"D:\ENSC\1A\Stage 1A\Imageries calciques\Donnees_brutes\HABITUATION\20260710_w5_habituation_synchro" #à changer
analog_path   = os.path.join(base_dir, "analog.txt")
suite2p_path  = os.path.join(base_dir, "suite2p")
# --- Nom de la souris extrait automatiquement du dossier ---
mouse_name  = "W5" #à changer
output_root = r"D:\ENSC\1A\Stage 1A\Imageries calciques\graphes"
output_dir  = os.path.join(output_root, "HABITUATION", mouse_name)
os.makedirs(output_dir, exist_ok=True)
print(f"Souris détectée : {mouse_name} -> figures dans {output_dir}")

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
# LOAD ANALOG (social touch uniquement, pas de vibration)
# Col 3 -> temps social touch (ms)   Col 4 -> signal social touch (== 1)
# =========================================================
analog = np.loadtxt(analog_path)
time_social_s = analog[:, 2] / 1000.0
social_signal = analog[:, 3].astype(int)

def detect_social_touch_analog(time, signal, remove_start=0.3):
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
        new_start = time[s] + remove_start
        if new_start < time[e]:
            social_intervals.append((new_start, time[e]))
    return social_intervals

social_intervals = detect_social_touch_analog(time_social_s, social_signal, remove_start=0.3)
social_intervals = sorted(social_intervals, key=lambda x: x[0])
print(f"Nb social touch détectés (trimmed) : {len(social_intervals)}")

# =========================================================
# PLOTS SESSION GLOBALE
# =========================================================
def plot_session_activity(dff, social_intervals, sampling_rate, cmap="RdPu", vmin=None, vmax=None):
    n_neurons, n_frames = dff.shape
    duration = n_frames / sampling_rate
    fig, (ax_top, ax_heat) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [0.4, 6]}
    )
    for start, end in social_intervals:
        ax_top.axvspan(start, end, facecolor="dodgerblue", edgecolor="none", alpha=0.35)
    ax_top.set_xlim(0, duration)
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_ylabel("ST")
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    im = ax_heat.imshow(dff, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_heat.set_xlabel("Temps (s)")
    ax_heat.set_ylabel("Neurones")
    step = 100
    ax_heat.set_xticks(np.arange(0, duration + step, step))
    ax_top.set_title(f"{mouse_name} — Activité globale (dF/F)")
    plt.tight_layout()
    return fig

def plot_neurons_traces(dff, social_intervals, sampling_rate, offset=2.0, linewidth=0.8):
    n_neurons, n_frames = dff.shape
    time = np.arange(n_frames) / sampling_rate
    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [0.4, 8]}
    )
    for t_start, t_end in social_intervals:
        ax_top.add_patch(Rectangle((float(t_start), 0), float(t_end - t_start), 1,
                                    facecolor="dodgerblue", edgecolor="none", alpha=0.35, zorder=2))
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_ylabel("ST")
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    colors = plt.cm.turbo(np.linspace(0, 1, n_neurons))
    for i in range(n_neurons):
        ax.plot(time, dff[i] + i * offset, color=colors[i], lw=linewidth)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Neurones")
    ax.set_ylim(-offset, n_neurons * offset)
    ax_top.set_title(f"{mouse_name} — Activité de tous les neurones, toute la séance")
    plt.tight_layout()
    return fig

def plot_social_touch_zscore(dff, social_intervals, sampling_rate, baseline_duration=1.0,cmap="RdPu", vmin=-2, vmax=2, title=None):
    n_neurons = dff.shape[0]
    n_trials = len(social_intervals)
    zscore_activity = np.zeros((n_neurons, n_trials))
    baseline_frames = int(baseline_duration * sampling_rate)
    for j, (t_start, t_end) in enumerate(social_intervals):
        start = int(t_start * sampling_rate)
        end = int(t_end * sampling_rate)
        baseline_start = max(0, start - baseline_frames)
        baseline = dff[:, baseline_start:start]
        response = dff[:, start:end].mean(axis=1)
        baseline_mean = baseline.mean(axis=1)
        baseline_std = baseline.std(axis=1)
        baseline_std[baseline_std == 0] = np.finfo(float).eps
        zscore_activity[:, j] = (response - baseline_mean) / baseline_std

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(zscore_activity, aspect="auto", cmap=cmap,
                    interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Social touch (n° dans le groupe)")
    ax.set_ylabel("Neurones")
    step = max(1, n_trials // 10)
    ax.set_xticks(np.arange(0, n_trials, step))
    ax.set_xticklabels(np.arange(1, n_trials + 1, step))
    step_neurons = max(1, n_neurons // 10)
    ax.set_yticks(np.arange(0, n_neurons, step_neurons))
    ax.set_yticklabels(np.arange(1, n_neurons + 1, step_neurons))
    if title:
        ax.set_title(f"{mouse_name} — {title}")
    else:
        ax.set_title(f"{mouse_name}")
    # axe dédié pour la colorbar -> ne recouvre jamais la heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-score")
    plt.tight_layout()
    return zscore_activity, fig

# ============================Enregistrement figures====================================
fig = plot_session_activity(dff, social_intervals, sampling_rate)
fig.savefig(os.path.join(output_dir,"df_f_time_Heatmap.png"))
plt.close(fig)
fig = plot_neurons_traces(dff, social_intervals, sampling_rate, offset=2)
fig.savefig(os.path.join(output_dir,"Activite_tous_neurones_toute_la_session.png"))
plt.close(fig)
# ======================================================================================

# =========================================================
# OUTILS COMMUNS (recrutement + AUC)
# =========================================================
def filter_valid_intervals(intervals, sampling_rate, min_duration_s=0.05):
    min_frames = max(2, int(min_duration_s * sampling_rate))
    filtered, removed = [], 0
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

def build_behavior_mask(signal_length, intervals, sampling_rate):
    mask = np.zeros(signal_length)
    for start, end in intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        mask[i0:i1] = 1
    return mask

def compute_observed_similarity(dff, behavior_mask):
    traces = dff.astype(np.float64)
    ss = np.sum(traces ** 2, axis=1)
    numerator = traces @ behavior_mask 
    mm = np.dot(behavior_mask, behavior_mask)
    return 2 * numerator / (mm + ss)

def compute_per_event_similarity(dff, intervals, sampling_rate):
    n_neurons, n_frames = dff.shape
    traces = dff.astype(np.float64)
    csum = np.zeros((n_neurons, n_frames + 1))
    np.cumsum(traces, axis=1, out=csum[:, 1:])
    ss = np.sum(traces ** 2, axis=1)

    n_events = len(intervals)
    similarities = np.zeros((n_events, n_neurons))
    for idx, (start, end) in enumerate(intervals):
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        dur = i1 - i0
        window_sum = csum[:, i1] - csum[:, i0]
        similarities[idx] = 2 * window_sum / (dur + ss)
    return similarities

def build_valid_shuffle_mask(signal_length, excluded_interval_lists, sampling_rate):
    valid = np.ones(signal_length, dtype=bool)
    for intervals in excluded_interval_lists:
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
                f"Pas assez d'espace hors social touch pour placer un bloc de "
                f"{dur} frames -- réduis n_shuffles ou vérifie la densité des essais."
            )
        weights = np.array([w1 - w0 - dur + 1 for w0, w1 in windows], dtype=float)
        window_idx = rng.choice(len(windows), p=weights / weights.sum())
        w0, w1 = windows[window_idx]
        start = rng.integers(w0, w1 - dur + 1)
        mask[start:start + dur] = 1
        available[start:start + dur] = False
    return mask

def build_null_distribution_per_event(dff, intervals, excluded_interval_lists, sampling_rate,n_shuffles=5000, seed=42):
    n_neurons, n_frames = dff.shape
    traces = dff.astype(np.float64)
    #  en O(1)
    csum = np.zeros((n_neurons, n_frames + 1))
    np.cumsum(traces, axis=1, out=csum[:, 1:])
    ss = np.sum(traces ** 2, axis=1)
    rng = np.random.default_rng(seed)
    valid_mask = build_valid_shuffle_mask(n_frames, excluded_interval_lists, sampling_rate)
    n_events = len(intervals)
    null_per_event = np.zeros((n_events, n_shuffles, n_neurons))
    for idx, (start, end) in enumerate(intervals):
        dur = int((end - start) * sampling_rate)
        windows = get_valid_windows(valid_mask, dur)
        if not windows:
            raise RuntimeError(f"Pas assez d'espace hors social touch pour l'essai {idx} "
                                f"(durée {dur} frames).")
        windows_arr = np.array(windows)
        weights = (windows_arr[:, 1] - windows_arr[:, 0] - dur + 1).astype(float)
        weights /= weights.sum()
        window_idxs = rng.choice(len(windows), size=n_shuffles, p=weights)
        w0_arr = windows_arr[window_idxs, 0]
        w1_arr = windows_arr[window_idxs, 1]
        starts = rng.integers(w0_arr, w1_arr - dur + 1)
        ends = starts + dur
        window_sums = (csum[:, ends] - csum[:, starts]).T
        denom = dur + ss[None, :]
        null_per_event[idx] = 2 * window_sums / denom
        if idx % 5 == 0:
            print(f"  Null distribution essai {idx}/{n_events}...")
    return null_per_event

def build_null_distribution_excluding_events(dff, intervals, excluded_interval_lists,sampling_rate, n_shuffles=5000, seed=42):
    n_neurons = dff.shape[0]
    signal_length = dff.shape[1]
    rng = np.random.default_rng(seed)
    valid_mask = build_valid_shuffle_mask(signal_length, excluded_interval_lists, sampling_rate)
    durations_frames = [int((e - s) * sampling_rate) for s, e in intervals]
    null_similarities = np.zeros((n_shuffles, n_neurons))
    for s in range(n_shuffles):
        if s % 500 == 0:
            print(f"  Shuffle {s}/{n_shuffles}...")
        shuffled_mask = generate_shuffled_mask_excluding_events(durations_frames, valid_mask, rng)
        for neuron in range(n_neurons):
            trace = dff[neuron].astype(float)
            null_similarities[s, neuron] = 2 * np.dot(shuffled_mask, trace) / (
                np.dot(shuffled_mask, shuffled_mask) + np.dot(trace, trace)
            )
    return null_similarities

def classify_recruited_neurons_per_event(observed_per_event_similarity, null_per_event,min_events=4, low_pct=0.83, high_pct=99.17):
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

#---------------------------------------------------
# AUC df/f
#---------------------------------------------------
def compute_auc_recruited_social(dff, social_intervals, neuron_classif, sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_trials = len(social_intervals)
    auc = np.full((len(recruited_idx), n_trials), np.nan)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        for trial, (start, end) in enumerate(social_intervals):
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            signal = dff[neuron, i0:i1].copy()
            if signal.size < 2:
                continue
            if response_type == 1:
                signal = np.maximum(signal, 0)
            elif response_type == -1:
                signal = np.minimum(signal, 0)
            auc[i, trial] = trapezoid(signal, dx=1 / sampling_rate)
    return auc, recruited_idx
#---------------------------------------------------
# AUC Z-score
#---------------------------------------------------
def compute_auc_recruited_social_zscore(dff, social_intervals, neuron_classif, sampling_rate,baseline_duration=0.5):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_trials = len(social_intervals)
    auc = np.full((len(recruited_idx), n_trials), np.nan)
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
            baseline_std = baseline.std()
            if baseline_std == 0:
                baseline_std = np.finfo(float).eps
            zsignal = (signal - baseline_mean) / baseline_std
            if response_type == 1:
                zsignal = np.maximum(zsignal, 0)
            elif response_type == -1:
                zsignal = np.minimum(zsignal, 0)
            auc[i, trial] = trapezoid(zsignal, dx=1 / sampling_rate)
    return auc, recruited_idx

def extract_social_activity_matrix(dff, social_intervals, sampling_rate):
    segments = []
    for start, end in social_intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        segments.append(np.mean(dff[:, i0:i1], axis=1))
    return np.array(segments)

def extract_social_zscore_matrix(dff, social_intervals, sampling_rate, baseline_duration=1.0):
    n_neurons = dff.shape[0]
    baseline_frames = int(baseline_duration * sampling_rate)
    segments = []
    for start, end in social_intervals:
        i0 = int(start * sampling_rate)
        i1 = int(end * sampling_rate)
        if i1 <= i0:
            continue
        baseline_start = max(0, i0 - baseline_frames)
        baseline = dff[:, baseline_start:i0]
        response = dff[:, i0:i1].mean(axis=1)
        baseline_mean = baseline.mean(axis=1)
        baseline_std = baseline.std(axis=1)
        baseline_std[baseline_std == 0] = np.finfo(float).eps
        segments.append((response - baseline_mean) / baseline_std)
    return np.array(segments)

def check_nan_fraction(auc, recruited_idx, neuron_classif):
    types = neuron_classif[recruited_idx]
    frac_nan = np.mean(np.isnan(auc), axis=1)
    print(f"Fraction NaN moyenne (excitateurs) : {frac_nan[types == 1].mean()*100:.1f}%")
    print(f"Fraction NaN moyenne (inhibiteurs)  : {frac_nan[types == -1].mean()*100:.1f}%")
    print(f"Fraction NaN globale : {frac_nan.mean()*100:.1f}%")

# =========================================================
# SELECTION DES 20 PREMIERS ET 20 DERNIERS SOCIAL TOUCH
# =========================================================
n_touch_total  = len(social_intervals)
n_select_early = 20
n_select_late  = 20

if n_touch_total < n_select_early + n_select_late:
    print(f"ATTENTION : seulement {n_touch_total} social touch au total, "
          f"les groupes 'Early' et 'Late' vont se chevaucher.")

early_social = filter_valid_intervals(social_intervals[:n_select_early], sampling_rate, min_duration_s=0.05)
late_social  = filter_valid_intervals(social_intervals[-n_select_late:], sampling_rate, min_duration_s=0.05)

print(f"Nb social touch total           : {n_touch_total}")
print(f"Nb social touch 'Early' retenus : {len(early_social)}")
print(f"Nb social touch 'Late' retenus  : {len(late_social)}")

# =========================================================
# METHODE DE RECRUTEMENT (shuffle, hors social touch — pas de vibration ici)
# =========================================================
def diagnose_per_event_significance(observed_per_event_similarity, null_per_event,low_pct=0.83, high_pct=99.17):
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

def compute_recruitment(dff, intervals, all_social_intervals, sampling_rate,n_shuffles=5000, seed=42, label="", min_events=4,low_pct=0.83, high_pct=99.17):
    per_event_similarity = compute_per_event_similarity(dff, intervals, sampling_rate)
    print(f"\n=== SHUFFLE PAR ESSAI {label} (hors social touch) ===")
    null_per_event = build_null_distribution_per_event(dff, intervals, excluded_interval_lists=[all_social_intervals],sampling_rate=sampling_rate, n_shuffles=n_shuffles, seed=seed)
    diagnose_per_event_significance(per_event_similarity, null_per_event, low_pct, high_pct)
    neuron_classif, n_sig_exc, n_sig_inh = classify_recruited_neurons_per_event(per_event_similarity, null_per_event, min_events=min_events,low_pct=low_pct, high_pct=high_pct)
    return neuron_classif

neuron_classif_early = compute_recruitment(dff, early_social, social_intervals, sampling_rate, n_shuffles=5000, seed=42, label="Early (20 premiers)", min_events=2)
neuron_classif_late  = compute_recruitment(dff, late_social, social_intervals, sampling_rate, n_shuffles=5000, seed=42, label="Late (20 derniers)", min_events=2)

# =========================================================
# NEURONE UNIQUE : CLASSIFICATION COMBINEE EARLY/LATE
# (calculée ici pour être disponible dans le récapitulatif et le graphe de proportions)
# =========================================================
def build_combined_classif(neuron_classif_early, neuron_classif_late):
    n_neurons = len(neuron_classif_early)
    combined = np.zeros(n_neurons, dtype=int)
    for n in range(n_neurons):
        if neuron_classif_early[n] != 0:
            combined[n] = neuron_classif_early[n]
        elif neuron_classif_late[n] != 0:
            combined[n] = neuron_classif_late[n]
    return combined

neuron_classif_combined = build_combined_classif(neuron_classif_early, neuron_classif_late)
print(f"\nNeurones recrutés (union Early/Late) : {np.sum(neuron_classif_combined != 0)} "
      f"({np.sum(neuron_classif_combined == 1)} activés, {np.sum(neuron_classif_combined == -1)} inhibés)")

def build_overlap(neuron_classif_early, neuron_classif_late):
    n_neurons = len(neuron_classif_early)
    combined = np.zeros(n_neurons, dtype=int)
    change_direction = np.zeros(n_neurons, dtype=int)
    for n in range(n_neurons):
        e = neuron_classif_early[n]
        l = neuron_classif_late[n]
        if e == 0 or l == 0:
            continue 
        if e == l:
            combined[n] = e
        else:
            combined[n] = l 
            change_direction[n] = 1 if (e == -1 and l == 1) else -1
    return combined, change_direction

neuron_overlap, neuron_change_direction = build_overlap(neuron_classif_early, neuron_classif_late)
n_changeants_inh_to_act = int(np.sum(neuron_change_direction == 1))
n_changeants_act_to_inh = int(np.sum(neuron_change_direction == -1))
n_changeants_overlap = n_changeants_inh_to_act + n_changeants_act_to_inh
print(f"\nNeurones recrutés (intersection Early ET Late) : {np.sum(neuron_overlap != 0)} "
      f"({np.sum(neuron_overlap == 1)} activés, {np.sum(neuron_overlap == -1)} inhibés, "
      f"dont {n_changeants_overlap} changeants : {n_changeants_inh_to_act} Inh->Act, "
      f"{n_changeants_act_to_inh} Act->Inh)")

# =========================================================
# TABLEAU RECAPITULATIF DU RECRUTEMENT
# =========================================================
def summarize_recruitment(neuron_classif, label, change_direction=None):
    n_exc = int(np.sum(neuron_classif == 1))
    n_inh = int(np.sum(neuron_classif == -1))
    n_rec = n_exc + n_inh
    n_tot = len(neuron_classif)
    result = {
        "Condition": label,
        "Neurones totaux": n_tot,
        "Recrutés (total)": n_rec,
        "Activés": n_exc,
        "Inhibés": n_inh,
        "Non recrutés": n_tot - n_rec,
    }
    if change_direction is not None:
        result["Changeants Inh->Act (sous-groupe Activés)"] = int(np.sum(change_direction == 1))
        result["Changeants Act->Inh (sous-groupe Inhibés)"] = int(np.sum(change_direction == -1))
    else:
        result["Changeants Inh->Act (sous-groupe Activés)"] = 0
        result["Changeants Act->Inh (sous-groupe Inhibés)"] = 0
    return result

summary_habituation = pd.DataFrame([
    summarize_recruitment(neuron_classif_early, "Early (20 premiers)"),
    summarize_recruitment(neuron_classif_late, "Late (20 derniers)"),
    summarize_recruitment(neuron_overlap, "Inetersection Early/Late)",
                           change_direction=neuron_change_direction),
])
print("\n=== RECAPITULATIF RECRUTEMENT — HABITUATION ===")
print(summary_habituation.to_string(index=False))
summary_habituation.to_excel(os.path.join(output_dir, "recap_recrutement_habituation.xlsx"), index=False)

# =========================================================
# PROPORTION DE NEURONES ACTIVES / INHIBES / NON RECRUTES
# =========================================================
def plot_recruitment_proportions(summary_df, mouse_name):
    conditions = summary_df["Condition"].tolist()
    n_tot = summary_df["Neurones totaux"].to_numpy(dtype=float)
    n_exc = summary_df["Activés"].to_numpy(dtype=float)
    n_inh = summary_df["Inhibés"].to_numpy(dtype=float)
    n_chg_ia = summary_df["Changeants Inh->Act (sous-groupe Activés)"].to_numpy(dtype=float)
    n_chg_ai = summary_df["Changeants Act->Inh (sous-groupe Inhibés)"].to_numpy(dtype=float)
    pct_exc = 100 * n_exc / n_tot
    pct_inh = 100 * n_inh / n_tot
    pct_chg_ia = 100 * n_chg_ia / n_tot
    pct_chg_ai = 100 * n_chg_ai / n_tot
    x = np.arange(len(conditions))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 7))
    b1 = ax.bar(x - 1.5*width, pct_exc, width, color="crimson", label="Activés (total)")
    b2 = ax.bar(x - 0.5*width, pct_inh, width, color="royalblue", label="Inhibés (total)")
    b3 = ax.bar(x + 0.5*width, pct_chg_ia, width, color="darkorange",
                label="dont changeants Inh→Act")
    b4 = ax.bar(x + 1.5*width, pct_chg_ai, width, color="seagreen",
                label="dont changeants Act→Inh")
    for bars, counts, pct in zip((b1, b2, b3, b4),
                                  (n_exc, n_inh, n_chg_ia, n_chg_ai),
                                  (pct_exc, pct_inh, pct_chg_ia, pct_chg_ai)):
        for rect, n, p in zip(bars, counts, pct):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                    f"{p:.1f}%\n(n={int(n)})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Proportion de neurones (%)")
    max_pct = max(np.max(pct_exc), np.max(pct_inh), 1)
    ax.set_ylim(0, max_pct * 1.3)
    ax.set_title(f"{mouse_name} — Proportion de neurones activés / inhibés\n"
                 f"(barres oranges/vertes = sous-parts changeantes Early→Late)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    return fig

fig_proportions = plot_recruitment_proportions(summary_habituation, mouse_name)
fig_proportions.savefig(os.path.join(output_dir, "Proportion_neurones_recrutes_early_late.png"))
# ======================================================================================
recruited_early_idx = set(np.where(neuron_classif_early != 0)[0].tolist())
recruited_late_idx  = set(np.where(neuron_classif_late  != 0)[0].tolist())
common_idx    = recruited_early_idx & recruited_late_idx
only_early_idx = recruited_early_idx - recruited_late_idx
only_late_idx  = recruited_late_idx - recruited_early_idx

print(f"\nNeurones recrutés Early ET Late      : {len(common_idx)}")
print(f"Neurones recrutés Early UNIQUEMENT   : {len(only_early_idx)}")
print(f"Neurones recrutés Late UNIQUEMENT    : {len(only_late_idx)}")
# =========================================================
# AUC PAR ESSAI, séparément Early / Late, avec le recrutement propre à chaque groupe
# =========================================================
auc_early, recruited_idx_early = compute_auc_recruited_social(dff, early_social, neuron_classif_early, sampling_rate)
auc_late, recruited_idx_late = compute_auc_recruited_social(dff, late_social, neuron_classif_late, sampling_rate)
# AUC Z-score
auc_early_z, recruited_idx_early_z = compute_auc_recruited_social_zscore(dff, early_social, neuron_classif_early, sampling_rate)
auc_late_z, recruited_idx_late_z = compute_auc_recruited_social_zscore(dff, late_social, neuron_classif_late, sampling_rate)

# =========================================================
# PLOTS
# =========================================================
def plot_auc_heatmap_single(auc, recruited_idx, neuron_classif, group_label):
    mean_auc = np.nanmean(auc, axis=1)
    types = neuron_classif[recruited_idx]
    exc_rows = np.where(types == 1)[0]
    inh_rows = np.where(types == -1)[0]
    exc_order = exc_rows[np.argsort(mean_auc[exc_rows])[::-1]]
    inh_order = inh_rows[np.argsort(mean_auc[inh_rows])]
    order = np.concatenate([exc_order, inh_order])
    auc_sorted = auc[order]
    neuron_ids_sorted = recruited_idx[order]
    finite_vals = auc_sorted[~np.isnan(auc_sorted)]
    if finite_vals.size == 0:
        print(f"Aucune AUC valide pour {group_label}.")
        return None
    vmax = np.percentile(np.abs(finite_vals), 95)
    n_neurons, n_trials = auc_sorted.shape
    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    sns.heatmap(
        auc_sorted, cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
        mask=np.isnan(auc_sorted),
        xticklabels=False, yticklabels=False,
        cbar_ax=cax, cbar_kws={"label": "AUC"},
        ax=ax,
    )
    ax.axhline(len(exc_order), color="black", linewidth=2)
    step_x = max(1, n_trials // 10)
    ax.set_xticks(np.arange(0, n_trials, step_x) + 0.5)
    ax.set_xticklabels(np.arange(1, n_trials + 1, step_x))
    step_y = max(1, n_neurons // 15)
    ax.set_yticks(np.arange(0, n_neurons, step_y) + 0.5)
    ax.set_yticklabels(neuron_ids_sorted[np.arange(0, n_neurons, step_y)])
    ax.set_xlabel(f"Social touch ({group_label}, n° dans le groupe)")
    ax.set_ylabel(f"Neurones recrutés — Act (n={len(exc_order)}, haut) / Inh (n={len(inh_order)}, bas)")
    ax.set_title(f"{mouse_name} — AUC Social-touch — {group_label}")
    plt.tight_layout()
    return fig


def plot_recruited_vs_not_recruited_single(dff, intervals, neuron_classif, sampling_rate,group_label, vmin=-2, vmax=2):
    recruited_idx = np.where(neuron_classif != 0)[0]
    not_recruited_idx = np.where(neuron_classif == 0)[0]
    activity_recruited = extract_social_zscore_matrix(dff[recruited_idx], intervals, sampling_rate).T
    activity_not_recruited = extract_social_zscore_matrix(dff[not_recruited_idx], intervals, sampling_rate).T
    order_r  = np.argsort(np.mean(activity_recruited, axis=1))
    order_nr = np.argsort(np.mean(activity_not_recruited, axis=1))
    activity_recruited = activity_recruited[order_r]
    activity_not_recruited = activity_not_recruited[order_nr]
    n_trials = activity_recruited.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    im1 = axes[0].imshow(activity_recruited, aspect="auto", cmap="RdPu", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{mouse_name} — Neurones recrutés ({group_label}, n={len(recruited_idx)})")
    axes[0].set_ylabel("Neurones")
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(im1, cax=cax0, label="Z-score")
    im2 = axes[1].imshow(activity_not_recruited, aspect="auto", cmap="RdPu", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{mouse_name} — Neurones non recrutés ({group_label}, n={len(not_recruited_idx)})")
    axes[1].set_ylabel("Neurones")
    step_x = max(1, n_trials // 10)
    axes[1].set_xticks(np.arange(0, n_trials, step_x))
    axes[1].set_xticklabels(np.arange(1, n_trials + 1, step_x))
    axes[1].set_xlabel(f"Social touch ({group_label}, n° dans le groupe)")
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(im2, cax=cax1, label="Z-score")
    plt.tight_layout()
    return fig

# ============================Enregistrement figures====================================
auc_dff_early_20 = plot_auc_heatmap_single(auc_early, recruited_idx_early, neuron_classif_early, "20 premiers")
if auc_dff_early_20 is not None:
    auc_dff_early_20.savefig(os.path.join(output_dir,"AUC_df_f_early_20.png"))
auc_dff_late_20 = plot_auc_heatmap_single(auc_late, recruited_idx_late, neuron_classif_late, "20 derniers")
if auc_dff_late_20 is not None:
    auc_dff_late_20.savefig(os.path.join(output_dir,"AUC_df_f_late_20.png"))
#AUC Z-score
auc_Zscore_early_20 = plot_auc_heatmap_single(auc_early_z, recruited_idx_early_z, neuron_classif_early, "20 premiers (Z-score AUC)")
if auc_Zscore_early_20 is not None:
    auc_Zscore_early_20.savefig(os.path.join(output_dir,"AUC_Z-score_early_20.png"))
auc_Zscore_late_20 = plot_auc_heatmap_single(auc_late_z, recruited_idx_late_z, neuron_classif_late, "20 derniers (Z-score AUC)")
if auc_Zscore_late_20 is not None:
    auc_Zscore_late_20.savefig(os.path.join(output_dir,"AUC_Z-score_late_20.png"))

zscore_early, fig_early = plot_social_touch_zscore(dff, early_social, sampling_rate, title="Z-score — 20 premiers social touch")
if fig_early is not None:
    fig_early.savefig(os.path.join(output_dir,"Z-score_early_20_premiers.png"))
zscore_late, fig_late = plot_social_touch_zscore(dff, late_social, sampling_rate, title="Z-score — 20 derniers social touch")
if fig_late is not None:
    fig_late.savefig(os.path.join(output_dir,"Z-score_late_20_derniers.png"))
zscore, fig = plot_social_touch_zscore(dff, social_intervals, sampling_rate, title="Z-score")
if fig is not None:
    fig.savefig(os.path.join(output_dir,"Z-score.png"))

auc_dff_early_20_recrutes_vs_non_recrutes = plot_recruited_vs_not_recruited_single(dff, early_social, neuron_classif_early, sampling_rate, "20 premiers")
if auc_dff_early_20_recrutes_vs_non_recrutes is not None:
    auc_dff_early_20_recrutes_vs_non_recrutes.savefig(os.path.join(output_dir,"AUC_df_f_early_20_recrutes_vs_non_recrutes.png"))
auc_dff_late_20_recrutes_vs_non_recrutes = plot_recruited_vs_not_recruited_single(dff, late_social, neuron_classif_late, sampling_rate, "20 derniers")
if auc_dff_late_20_recrutes_vs_non_recrutes is not None:
    auc_dff_late_20_recrutes_vs_non_recrutes.savefig(os.path.join(output_dir,"AUC_df_f_late_20_recrutes_vs_non_recrutes.png"))

check_nan_fraction(auc_early, recruited_idx_early, neuron_classif_early)
check_nan_fraction(auc_late, recruited_idx_late, neuron_classif_late)

# ------------------------------------------------
# 20 premiers et 20 derniers sur toutes la séance
# ------------------------------------------------
auc_early_full, recruited_idx_early_full = compute_auc_recruited_social(dff, social_intervals, neuron_classif_early, sampling_rate)
auc_late_full, recruited_idx_late_full = compute_auc_recruited_social(dff, social_intervals, neuron_classif_late, sampling_rate)

auc_dff_early_full = plot_auc_heatmap_single(auc_early_full, recruited_idx_early_full,neuron_classif_early, "Recrutés Early — toute la séance")
if auc_dff_early_full is not None:
    auc_dff_early_full.savefig(os.path.join(output_dir,"AUC_df_f_early_full.png"))
auc_dff_late_full = plot_auc_heatmap_single(auc_late_full, recruited_idx_late_full,neuron_classif_late, "Recrutés Late — toute la séance")
if auc_dff_late_full is not None:
    auc_dff_late_full.savefig(os.path.join(output_dir,"AUC_df_f_late_full.png"))

#AUC Z-score
auc_early_full_z, recruited_idx_early_full_z = compute_auc_recruited_social_zscore(dff, social_intervals, neuron_classif_early, sampling_rate)
auc_late_full_z, recruited_idx_late_full_z = compute_auc_recruited_social_zscore(dff, social_intervals, neuron_classif_late, sampling_rate)

auc_Zscore_early_full = plot_auc_heatmap_single(auc_early_full_z, recruited_idx_early_full_z, neuron_classif_early, "Recrutés Early — toute la séance (Z-score AUC)")
if auc_Zscore_early_full is not None:
    auc_Zscore_early_full.savefig(os.path.join(output_dir,"AUC_Z-score_early_full.png"))
auc_Zscore_late_full = plot_auc_heatmap_single(auc_late_full_z, recruited_idx_late_full_z, neuron_classif_late, "Recrutés Late — toute la séance (Z-score AUC)")
if auc_Zscore_late_full is not None:
    auc_Zscore_late_full.savefig(os.path.join(output_dir,"AUC_Z-score_late_full.png"))
# ======================================================================================

# --- Courbe comparative : AUC moyen +/- SEM par essai, Early-recrutés vs Late-recrutés ---
def add_regression_line(ax, trials, values, color, label_prefix):
    valid = ~np.isnan(values)
    x_valid = trials[valid]
    y_valid = values[valid]
    if x_valid.size < 2:
        return None
    slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)
    x_line = np.array([trials.min(), trials.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=2.5, linestyle="-",
            label=f"{label_prefix} — droite de régression (pente={slope:.4f}, p={p_value:.3g})")
    return slope, intercept, r_value, p_value


def plot_auc_comparison_over_session(auc_early_full, auc_late_full, n_touch_total,ylabel="AUC moyen", title_suffix=""):
    mean_early = np.nanmean(auc_early_full, axis=0)
    sem_early  = np.nanstd(auc_early_full, axis=0) / np.sqrt(np.sum(~np.isnan(auc_early_full), axis=0))
    mean_late  = np.nanmean(auc_late_full, axis=0)
    sem_late   = np.nanstd(auc_late_full, axis=0) / np.sqrt(np.sum(~np.isnan(auc_late_full), axis=0))
    trials = np.arange(1, n_touch_total + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Courbes brutes (données réelles)
    ax.plot(trials, mean_early, color="orange", alpha=0.5, linewidth=1.5,
            label="Neurones recrutés (Early)")
    ax.plot(trials, mean_late, color="violet", alpha=0.5, linewidth=1.5,
            label="Neurones recrutés (Late)")
    # Droites de régression, superposées
    fit_early = add_regression_line(ax, trials, mean_early, color="darkorange", label_prefix="Early")
    fit_late  = add_regression_line(ax, trials, mean_late, color="purple", label_prefix="Late")
    if fit_early is not None:
        print(f"Pente Early : {fit_early[0]:.5f} (p={fit_early[3]:.4g}, r²={fit_early[2]**2:.3f})")
    if fit_late is not None:
        print(f"Pente Late  : {fit_late[0]:.5f} (p={fit_late[3]:.4g}, r²={fit_late[2]**2:.3f})")
    ax.set_xlabel("Social touch (n° dans la séance)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{mouse_name} — Comparaison AUC — neurones recrutés Early vs Late, "
                 f"sur toute la séance{title_suffix}")
    ax.legend()
    plt.tight_layout()
    return fig

# ============================Enregistrement figures====================================
# dF/F
auc_dff_comparaison_late_early = plot_auc_comparison_over_session(auc_early_full, auc_late_full, n_touch_total)
auc_dff_comparaison_late_early.savefig(os.path.join(output_dir,"AUC_df_f_comparaison_late_early.png"))
# Z-score
auc_Zscore_comparaison_late_early = plot_auc_comparison_over_session(auc_early_full_z, auc_late_full_z, n_touch_total,ylabel="AUC moyen (Z-score)", title_suffix=" — Z-score")
auc_Zscore_comparaison_late_early.savefig(os.path.join(output_dir,"AUC_Z-score_comparaison_late_early.png"))
# ======================================================================================

def mean_sem(auc_full, recruited_idx, neuron_classif, response_type):
        types = neuron_classif[recruited_idx]
        rows = np.where(types == response_type)[0]
        sub = auc_full[rows]
        mean = np.nanmean(sub, axis=0)
        sem  = np.nanstd(sub, axis=0) / np.sqrt(np.sum(~np.isnan(sub), axis=0))
        return mean, sem, len(rows)

def plot_auc_comparison_over_session_exc_inh(auc_early_full, recruited_idx_early_full, neuron_classif_early,auc_late_full, recruited_idx_late_full, neuron_classif_late,n_touch_total, ylabel="AUC moyen", title_suffix=""):
    trials = np.arange(1, n_touch_total + 1)
    mean_early_exc, sem_early_exc, n_early_exc = mean_sem(auc_early_full, recruited_idx_early_full, neuron_classif_early, 1)
    mean_early_inh, sem_early_inh, n_early_inh = mean_sem(auc_early_full, recruited_idx_early_full, neuron_classif_early, -1)
    mean_late_exc,  sem_late_exc,  n_late_exc  = mean_sem(auc_late_full,  recruited_idx_late_full,  neuron_classif_late,  1)
    mean_late_inh,  sem_late_inh,  n_late_inh  = mean_sem(auc_late_full,  recruited_idx_late_full,  neuron_classif_late,  -1)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Courbes brutes (donnees reelles)
    ax.plot(trials, mean_early_exc, color="orange", linestyle="-", alpha=0.4, linewidth=1.2,
            label=f"Early — Activés (n={n_early_exc})")
    ax.plot(trials, mean_early_inh, color="orange", linestyle="--", alpha=0.4, linewidth=1.2,
            label=f"Early — Inhibés (n={n_early_inh})")
    ax.plot(trials, mean_late_exc, color="violet", linestyle="-", alpha=0.4, linewidth=1.2,
            label=f"Late — Activés (n={n_late_exc})")
    ax.plot(trials, mean_late_inh, color="violet", linestyle="--", alpha=0.4, linewidth=1.2,
            label=f"Late — Inhibés (n={n_late_inh})")
    # Droites de regression
    fit_early_exc = add_regression_line(ax, trials, mean_early_exc, color="darkorange", label_prefix="Early Act")
    fit_early_inh = add_regression_line(ax, trials, mean_early_inh, color="chocolate", label_prefix="Early Inh")
    fit_late_exc  = add_regression_line(ax, trials, mean_late_exc, color="purple", label_prefix="Late Act")
    fit_late_inh  = add_regression_line(ax, trials, mean_late_inh, color="mediumvioletred", label_prefix="Late Inh")
    for label, fit in [("Early Act", fit_early_exc), ("Early Inh", fit_early_inh),("Late Act", fit_late_exc), ("Late Inh", fit_late_inh)]:
        if fit is not None:
            print(f"Pente {label} : {fit[0]:.5f} (p={fit[3]:.4g}, r²={fit[2]**2:.3f})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Social touch (n° dans la séance)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{mouse_name} — Comparaison AUC — Activés vs Inhibés, Early vs Late, "
                 f"sur toute la séance{title_suffix}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

# ============================Enregistrement figures====================================
# dF/F
auc_dff_comparaison_late_early_exc_inh_separes = plot_auc_comparison_over_session_exc_inh(auc_early_full, recruited_idx_early_full, neuron_classif_early,auc_late_full, recruited_idx_late_full, neuron_classif_late, n_touch_total)
auc_dff_comparaison_late_early_exc_inh_separes.savefig(os.path.join(output_dir,"AUC_df_f_comparaison_late_early.png"))
# Z-score
auc_Zscore_comparaison_late_early_exc_inh_separes = plot_auc_comparison_over_session_exc_inh(auc_early_full_z, recruited_idx_early_full_z, neuron_classif_early,auc_late_full_z, recruited_idx_late_full_z, neuron_classif_late, n_touch_total,ylabel="AUC moyen (Z-score)", title_suffix=" — Z-score")
auc_Zscore_comparaison_late_early_exc_inh_separes.savefig(os.path.join(output_dir,"AUC_Z-score_comparaison_late_early.png"))
# ======================================================================================

def plot_auc_evolution_selected_trials(dff, social_intervals, neuron_classif, sampling_rate,group_label, step=10, trend="mean"):
    auc_full, recruited_idx = compute_auc_recruited_social_zscore(dff, social_intervals, neuron_classif, sampling_rate)
    n_touch_total = len(social_intervals)
    selected_trials = np.arange(0, n_touch_total, step)
    auc_selected = auc_full[:, selected_trials]
    x = selected_trials + 1
    types = neuron_classif[recruited_idx]
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, neuron in enumerate(recruited_idx):
        color = "crimson" if types[i] == 1 else "royalblue"
        ax.plot(x, auc_selected[i], color=color, alpha=0.3, linewidth=1, marker="o", markersize=3)
    for t, color, label in [(1, "crimson", "Activés"), (-1, "royalblue", "Inhibés")]:
        rows = np.where(types == t)[0]
        if len(rows) == 0:
            continue
        mean_t = np.nanmean(auc_selected[rows], axis=0)
        sem_t  = np.nanstd(auc_selected[rows], axis=0) / np.sqrt(len(rows))
        if trend == "smooth" and len(x) >= 4:
            from scipy.interpolate import make_interp_spline
            x_smooth = np.linspace(x.min(), x.max(), 200)
            spline = make_interp_spline(x, mean_t, k=min(3, len(x) - 1))
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=3,
                    label=f"Tendance {label} (lissée, n={len(rows)})")
        else:
            ax.plot(x, mean_t, color=color, linewidth=3,
                    label=f"Tendance {label} (moyenne, n={len(rows)})")
        #ax.fill_between(x, mean_t - sem_t, mean_t + sem_t, color=color, alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.set_xlabel("Social touch n°")
    ax.set_ylabel("AUC")
    ax.set_title(f"{mouse_name} — Évolution de l'AUC Z-score — {group_label} (essais {', '.join(map(str, x))})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    plt.tight_layout()
    return auc_selected, recruited_idx, x, fig

# ============================Enregistrement figures====================================
auc_sel_early, idx_sel_early, trials_sel_early, evolution_early = plot_auc_evolution_selected_trials(dff, social_intervals, neuron_classif_early, sampling_rate,group_label="Recrutés Early - Z-score", step=10, trend="smooth")
evolution_early.savefig(os.path.join(output_dir,"AUC_Z-score_evolution_early.png"))
auc_sel_late, idx_sel_late, trials_sel_late, evolution_late = plot_auc_evolution_selected_trials(dff, social_intervals, neuron_classif_late, sampling_rate,group_label="Recrutés Late - Z-score", step=10, trend="smooth")
evolution_late.savefig(os.path.join(output_dir,"AUC_Z-score_evolution_late.png"))
# ======================================================================================

def plot_recruited_neurons_activity(dff, social_intervals, neuron_classif, sampling_rate,group_label, baseline_duration=1.0, offset=3.0,linewidth=0.8, pad_s=2.0):
    recruited_idx = np.where(neuron_classif != 0)[0]
    n_neurons = len(recruited_idx)
    if n_neurons == 0:
        print(f"Aucun neurone recruté pour {group_label}")
        return None
    t_start_win = max(0, social_intervals[0][0] - pad_s)
    t_end_win   = min(dff.shape[1] / sampling_rate, social_intervals[-1][1] + pad_s)
    i0_win = int(t_start_win * sampling_rate)
    i1_win = int(t_end_win * sampling_rate)
    time = np.arange(i0_win, i1_win) / sampling_rate
    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(14, max(6, n_neurons * 0.6)), sharex=True,
        gridspec_kw={"height_ratios": [0.3, 8]}
    )
    for start, end in social_intervals:
        ax_top.add_patch(Rectangle((start, 0), end - start, 1,facecolor="dodgerblue", edgecolor="none",alpha=0.35, zorder=2))
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([]); ax_top.set_xticks([])
    ax_top.set_ylabel("ST")
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    types = neuron_classif[recruited_idx]
    order = np.argsort(-types)  # 1 (exc) avant -1 (inh)
    yticks, yticklabels = [], []
    for rank, k in enumerate(order):
        neuron = recruited_idx[k]
        response_type = types[k]
        color = "crimson" if response_type == 1 else "royalblue"
        trace = dff[neuron, i0_win:i1_win]
        y0 = rank * offset
        ax.plot(time, trace + y0, color=color, lw=linewidth)
        ax.axhline(y0, color="black", lw=0.5, linestyle=":", alpha=0.6)  # F0 = 0 pour ce neurone
        yticks.append(y0)
        yticklabels.append(f"N{neuron} ({'Act' if response_type == 1 else 'Inh'})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_ylim(-offset, n_neurons * offset)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Neurones recrutés (dF/F)")
    ax.set_title(f"{mouse_name} — Activité des neurones recrutés — {group_label} "f"(n={n_neurons}, ligne pointillée = F0)")
    plt.tight_layout()
    return fig

# ============================Enregistrement figures====================================
activite_early_20 = plot_recruited_neurons_activity(dff, early_social, neuron_classif_early, sampling_rate,group_label="Early (20 premiers social touch)")
if activite_early_20 is not None:
    activite_early_20.savefig(os.path.join(output_dir,"Activite_early_20_premiers.png"))
activite_late_20 = plot_recruited_neurons_activity(dff, late_social, neuron_classif_late, sampling_rate,group_label="Late (20 derniers social touch)")
if activite_late_20 is not None:
    activite_late_20.savefig(os.path.join(output_dir,"Activite_late_20_derniers.png"))
# ======================================================================================

#------------------Z-score early et late tout au long de la séance-------------------------------------
def plot_zscore_recruited_full_session(dff, social_intervals, neuron_classif, sampling_rate,group_label, baseline_duration=1.0,cmap="RdPu", vmin=-2, vmax=2):
    recruited_idx = np.where(neuron_classif != 0)[0]
    if len(recruited_idx) == 0:
        print(f"Aucun neurone recruté pour {group_label}")
        return None
    # Z-score sur toute la séance, pour les neurones recrutés uniquement
    zscore_activity = extract_social_zscore_matrix(dff[recruited_idx], social_intervals, sampling_rate, baseline_duration).T  
    n_trials = zscore_activity.shape[1]
    types = neuron_classif[recruited_idx]
    exc_rows = np.where(types == 1)[0]
    inh_rows = np.where(types == -1)[0]
    mean_z = np.nanmean(zscore_activity, axis=1)
    exc_order = exc_rows[np.argsort(mean_z[exc_rows])[::-1]]
    inh_order = inh_rows[np.argsort(mean_z[inh_rows])]
    order = np.concatenate([exc_order, inh_order])
    zscore_sorted = zscore_activity[order]
    neuron_ids_sorted = recruited_idx[order]
    n_neurons = zscore_sorted.shape[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(zscore_sorted, aspect="auto", cmap=cmap,interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.axhline(len(exc_order) - 0.5, color="black", linewidth=2)
    step_x = max(1, n_trials // 10)
    ax.set_xticks(np.arange(0, n_trials, step_x))
    ax.set_xticklabels(np.arange(1, n_trials + 1, step_x))
    step_y = max(1, n_neurons // 15)
    ax.set_yticks(np.arange(0, n_neurons, step_y))
    ax.set_yticklabels(neuron_ids_sorted[np.arange(0, n_neurons, step_y)])
    ax.set_xlabel("Social touch (n° dans la séance)")
    ax.set_ylabel(f"Neurones recrutés — Act (n={len(exc_order)}, haut) / Inh (n={len(inh_order)}, bas)")
    ax.set_title(f"{mouse_name} — Z-score — Neurones recrutés {group_label}, toute la séance")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Z-score")
    plt.tight_layout()
    return zscore_sorted, fig

# ============================Enregistrement figures====================================
result_early_zsession = plot_zscore_recruited_full_session(dff, social_intervals, neuron_classif_early, sampling_rate, group_label="Early")
if result_early_zsession is not None:
    zscore_early_recruited_full, Zscore_early = result_early_zsession
    Zscore_early.savefig(os.path.join(output_dir,"Z-score_early_toute_la_seance.png"))
result_late_zsession = plot_zscore_recruited_full_session(dff, social_intervals, neuron_classif_late, sampling_rate, group_label="Late")
if result_late_zsession is not None:
    zscore_late_recruited_full, Zscore_late = result_late_zsession
    Zscore_late.savefig(os.path.join(output_dir,"Z-score_late_toute_la_seance.png"))
# ======================================================================================

#------------------------------heatmap df/f-------------------------------------
def plot_mean_dff_recruited_heatmap(dff, social_intervals, neuron_classif, sampling_rate,group_label, cmap="RdPu", vmin=None, vmax=None):
    recruited_idx = np.where(neuron_classif != 0)[0]
    if len(recruited_idx) == 0:
        print(f"Aucun neurone recruté pour {group_label}")
        return None
    activity = extract_social_activity_matrix(dff[recruited_idx], social_intervals, sampling_rate).T
    n_trials = activity.shape[1]
    types = neuron_classif[recruited_idx]
    exc_rows = np.where(types == 1)[0]
    inh_rows = np.where(types == -1)[0]
    mean_activity = np.nanmean(activity, axis=1)
    exc_order = exc_rows[np.argsort(mean_activity[exc_rows])[::-1]]
    inh_order = inh_rows[np.argsort(mean_activity[inh_rows])]
    order = np.concatenate([exc_order, inh_order])
    activity_sorted = activity[order]
    neuron_ids_sorted = recruited_idx[order]
    n_neurons = activity_sorted.shape[0]
    if vmax is None:
        finite_vals = activity_sorted[~np.isnan(activity_sorted)]
        vmax = np.percentile(np.abs(finite_vals), 95) if finite_vals.size else 1
        vmin = -vmax if vmin is None else vmin
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(activity_sorted, aspect="auto", cmap=cmap,interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.axhline(len(exc_order) - 0.5, color="black", linewidth=2)
    step_x = max(1, n_trials // 10)
    ax.set_xticks(np.arange(0, n_trials, step_x))
    ax.set_xticklabels(np.arange(1, n_trials + 1, step_x))
    step_y = max(1, n_neurons // 15)
    ax.set_yticks(np.arange(0, n_neurons, step_y))
    ax.set_yticklabels(neuron_ids_sorted[np.arange(0, n_neurons, step_y)])
    ax.set_xlabel("Social touch (n° dans la séance)")
    ax.set_ylabel(f"Neurones recrutés — Act (n={len(exc_order)}, haut) / Inh (n={len(inh_order)}, bas)")
    ax.set_title(f"{mouse_name} — Moyenne dF/F par social touch — Neurones recrutés {group_label}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("dF/F moyen")
    plt.tight_layout()
    return activity_sorted, neuron_ids_sorted, fig

# ============================Enregistrement figures====================================
result_dff_early = plot_mean_dff_recruited_heatmap(dff, social_intervals, neuron_classif_early, sampling_rate, group_label="Early")
if result_dff_early is not None:
    dff_mean_early, ids_early, mean_dff_early_fig = result_dff_early
    mean_dff_early_fig.savefig(os.path.join(output_dir,"df_f_moyenne_early_toute_la_seance.png"))
result_dff_late = plot_mean_dff_recruited_heatmap(dff, social_intervals, neuron_classif_late, sampling_rate, group_label="Late")
if result_dff_late is not None:
    dff_mean_late, ids_late, mean_dff_late_fig = result_dff_late
    mean_dff_late_fig.savefig(os.path.join(output_dir,"df_f_moyenne_late_toute_la_seance.png"))
# ======================================================================================

# --- df_f en frame ---
def plot_session_activity_frames(dff, social_intervals, sampling_rate, cmap="RdPu", vmin=None, vmax=None):
    n_neurons, n_frames = dff.shape

    fig, (ax_top, ax_heat) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [0.4, 6]}
    )
    for start, end in social_intervals:
        f0 = start * sampling_rate
        f1 = end * sampling_rate
        ax_top.axvspan(f0, f1, facecolor="dodgerblue", edgecolor="none", alpha=0.35)
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([]); ax_top.set_xticks([])
    ax_top.set_ylabel("ST")
    for spine in ("top", "right", "left"):
        ax_top.spines[spine].set_visible(False)
    im = ax_heat.imshow(dff, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_heat.set_xlabel("Frame")
    ax_heat.set_ylabel("Neurones")
    step = 1000
    ax_heat.set_xticks(np.arange(0, n_frames + step, step))
    ax_heat.set_xlim(0, n_frames)
    divider_heat = make_axes_locatable(ax_heat)
    cax = divider_heat.append_axes("right", size="4%", pad=0.15)
    fig.colorbar(im, cax=cax, label="dF/F")
    divider_top = make_axes_locatable(ax_top)
    cax_top = divider_top.append_axes("right", size="4%", pad=0.15)
    cax_top.axis("off")
    ax_top.set_title(f"{mouse_name} — dF/F par frame")
    plt.tight_layout()
    return fig

# ============================Enregistrement figures====================================
dff_frames = plot_session_activity_frames(dff, social_intervals, sampling_rate)
dff_frames.savefig(os.path.join(output_dir,"df_f_frames_Heatmap.png"))
# ======================================================================================

# ===================================================================================
# Peak amplitude
# ===================================================================================
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
    ax.set_ylabel(f"Activés (n={len(exc_order)}, haut) / Inhibés (n={len(inh_order)}, bas)")
    ax.set_title(f"{mouse_name} — Peak amplitude relatif à la baseline — rouge = activé, bleu = inhibé")
    plt.tight_layout()
    return fig
# ============================Enregistrement figures====================================
peak_early, recruited_idx_early_peak = compute_peak_amplitude_recruited_social(dff, early_social, neuron_classif_early, sampling_rate)
fig_peak_early = plot_peak_amplitude_heatmap(peak_early, recruited_idx_early_peak, neuron_classif_early, xlabel="Social touch (20 premiers)")
if fig_peak_early is not None:
    fig_peak_early.savefig(os.path.join(output_dir,"Peak_amplitude_early_20.png"))

peak_late, recruited_idx_late_peak = compute_peak_amplitude_recruited_social(dff, late_social, neuron_classif_late, sampling_rate)
fig_peak_late = plot_peak_amplitude_heatmap(peak_late, recruited_idx_late_peak, neuron_classif_late, xlabel="Social touch (20 derniers)")
if fig_peak_late is not None:
    fig_peak_late.savefig(os.path.join(output_dir,"Peak_amplitude_late_20.png"))
# ======================================================================================

# =========================================================
# MOYENNE PAR NEURONE : dF/F, AUC, PEAK AMPLITUDE
# (neuron_classif_combined a déjà été calculé plus haut, juste après early/late)
# =========================================================
def mean_dff_per_neuron_signed(dff, intervals, neuron_classif, sampling_rate):
    recruited_idx = np.where(neuron_classif != 0)[0]
    means = np.full(len(recruited_idx), np.nan)
    for i, neuron in enumerate(recruited_idx):
        response_type = neuron_classif[neuron]
        vals = []
        for start, end in intervals:
            i0 = int(start * sampling_rate)
            i1 = int(end * sampling_rate)
            if i1 <= i0:
                continue
            signal = dff[neuron, i0:i1].copy()
            if response_type == 1:
                signal = np.maximum(signal, 0)
            else:
                signal = np.minimum(signal, 0)
            vals.append(np.mean(signal))
        if vals:
            means[i] = np.mean(vals)
    return means, recruited_idx

# dF/F moyen par neurone, Early vs Late
mean_dff_early, idx_dff = mean_dff_per_neuron_signed(dff, early_social, neuron_classif_combined, sampling_rate)
mean_dff_late,  _       = mean_dff_per_neuron_signed(dff, late_social,  neuron_classif_combined, sampling_rate)

# AUC moyen par neurone 
auc_early_c, idx_auc = compute_auc_recruited_social(dff, early_social, neuron_classif_combined, sampling_rate)
auc_late_c,  _        = compute_auc_recruited_social(dff, late_social,  neuron_classif_combined, sampling_rate)
mean_auc_early = np.nanmean(auc_early_c, axis=1)
mean_auc_late  = np.nanmean(auc_late_c,  axis=1)

# Peak amplitude moyen par neurone
peak_early_c, idx_peak = compute_peak_amplitude_recruited_social(dff, early_social, neuron_classif_combined, sampling_rate)
peak_late_c,  _        = compute_peak_amplitude_recruited_social(dff, late_social,  neuron_classif_combined, sampling_rate)
mean_peak_early = np.nanmean(peak_early_c, axis=1)
mean_peak_late  = np.nanmean(peak_late_c,  axis=1)

types_combined = neuron_classif_combined[idx_dff]  # même ordre pour idx_dff, idx_auc, idx_peak

# =========================================================
# GRAPHE EARLY -> LATE (une ligne par neurone, exc en rouge, inh en bleu)
# =========================================================
def plot_early_late_comparison(values_early, values_late, types, ylabel, title,xtick_labels=("Early\n(20 premiers)", "Late\n(20 derniers)")):
    fig, ax = plt.subplots(figsize=(6, 8))
    x = [0, 1]
    for i in range(len(values_early)):
        if np.isnan(values_early[i]) or np.isnan(values_late[i]):
            continue
        color = "crimson" if types[i] == 1 else "royalblue"
        ax.plot(x, [values_early[i], values_late[i]], color=color, alpha=0.4,
                 marker="o", markersize=4, linewidth=1)
    exc_mask = types == 1
    inh_mask = types == -1
    if np.any(exc_mask):
        ax.plot(x, [np.nanmean(values_early[exc_mask]), np.nanmean(values_late[exc_mask])],
                 color="crimson", linewidth=3, marker="o", markersize=9,
                 label=f"Activés (n={np.sum(exc_mask)})")
    if np.any(inh_mask):
        ax.plot(x, [np.nanmean(values_early[inh_mask]), np.nanmean(values_late[inh_mask])],
                 color="royalblue", linewidth=3, marker="o", markersize=9,
                 label=f"Inhibés (n={np.sum(inh_mask)})")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{mouse_name} — {title}")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    plt.tight_layout()
    return fig

def compute_mean_dff_early_late_populations(dff, neuron_classif_early, neuron_classif_late,early_social, late_social, sampling_rate):
    # early
    dff_early_first20, idx_early = mean_dff_per_neuron_signed(dff, early_social, neuron_classif_early, sampling_rate)
    dff_early_last20, _ = mean_dff_per_neuron_signed(dff, late_social, neuron_classif_early, sampling_rate)
    # late
    dff_late_first20, idx_late = mean_dff_per_neuron_signed(dff, early_social, neuron_classif_late, sampling_rate)
    dff_late_last20, _ = mean_dff_per_neuron_signed(dff, late_social, neuron_classif_late, sampling_rate)
    return {
        "early_population": {
            "dff_first20": dff_early_first20,
            "dff_last20": dff_early_last20,
            "types": neuron_classif_early[idx_early],
            "neuron_idx": idx_early,
        },
        "late_population": {
            "dff_first20": dff_late_first20,
            "dff_last20": dff_late_last20,
            "types": neuron_classif_late[idx_late],
            "neuron_idx": idx_late,
        },
    }

def compute_mean_auc_early_late_populations(dff, neuron_classif_early, neuron_classif_late,early_social, late_social, sampling_rate):
    # early
    auc_early_first20_raw, idx_early = compute_auc_recruited_social(dff, early_social, neuron_classif_early, sampling_rate)
    auc_early_last20_raw, _ = compute_auc_recruited_social(dff, late_social, neuron_classif_early, sampling_rate)
    auc_early_first20 = np.nanmean(auc_early_first20_raw, axis=1)
    auc_early_last20  = np.nanmean(auc_early_last20_raw, axis=1)
    # late
    auc_late_first20_raw, idx_late = compute_auc_recruited_social(dff, early_social, neuron_classif_late, sampling_rate)
    auc_late_last20_raw, _ = compute_auc_recruited_social(dff, late_social, neuron_classif_late, sampling_rate)
    auc_late_first20 = np.nanmean(auc_late_first20_raw, axis=1)
    auc_late_last20  = np.nanmean(auc_late_last20_raw, axis=1)
    return {
        "early_population": {
            "auc_first20": auc_early_first20,
            "auc_last20": auc_early_last20,
            "types": neuron_classif_early[idx_early],
            "neuron_idx": idx_early,
        },
        "late_population": {
            "auc_first20": auc_late_first20,
            "auc_last20": auc_late_last20,
            "types": neuron_classif_late[idx_late],
            "neuron_idx": idx_late,
        },
    }

def compute_mean_peak_early_late_populations(dff, neuron_classif_early, neuron_classif_late,early_social, late_social, sampling_rate):
    # early
    peak_early_first20_raw, idx_early = compute_peak_amplitude_recruited_social(dff, early_social, neuron_classif_early, sampling_rate)
    peak_early_last20_raw, _ = compute_peak_amplitude_recruited_social(dff, late_social, neuron_classif_early, sampling_rate)
    peak_early_first20 = np.nanmean(peak_early_first20_raw, axis=1)
    peak_early_last20  = np.nanmean(peak_early_last20_raw, axis=1)
    # late
    peak_late_first20_raw, idx_late = compute_peak_amplitude_recruited_social(dff, early_social, neuron_classif_late, sampling_rate)
    peak_late_last20_raw, _ = compute_peak_amplitude_recruited_social(dff, late_social, neuron_classif_late, sampling_rate)
    peak_late_first20 = np.nanmean(peak_late_first20_raw, axis=1)
    peak_late_last20  = np.nanmean(peak_late_last20_raw, axis=1)
    return {
        "early_population": {
            "peak_first20": peak_early_first20,
            "peak_last20": peak_early_last20,
            "types": neuron_classif_early[idx_early],
            "neuron_idx": idx_early,
        },
        "late_population": {
            "peak_first20": peak_late_first20,
            "peak_last20": peak_late_last20,
            "types": neuron_classif_late[idx_late],
            "neuron_idx": idx_late,
        },
    }
# ============================================================
# Enregistrement des figures
# ============================================================
new_labels = ("20 premiers\nsocial touch", "20 derniers\nsocial touch")
# ============================ Moyennes des df/f ====================================
mean_dff_results = compute_mean_dff_early_late_populations(
    dff, neuron_classif_early, neuron_classif_late, early_social, late_social, sampling_rate
)
# Population Early : dF/F moyen sur 20 premiers vs 20 derniers social touch
pop_early = mean_dff_results["early_population"]
fig_dff_pop_early = plot_early_late_comparison(
    pop_early["dff_first20"], pop_early["dff_last20"], pop_early["types"],
    ylabel="dF/F moyen",
    title="Neurones recrutés Early — dF/F moyen",
    xtick_labels=new_labels
)
fig_dff_pop_early.savefig(os.path.join(output_dir, "Comparaison_dff_population_early.png"))
# Population Late : dF/F moyen sur 20 premiers vs 20 derniers social touch
pop_late = mean_dff_results["late_population"]
fig_dff_pop_late = plot_early_late_comparison(
    pop_late["dff_first20"], pop_late["dff_last20"], pop_late["types"],
    ylabel="dF/F moyen",
    title="Neurones recrutés Late — dF/F moyen",
    xtick_labels=new_labels
)
fig_dff_pop_late.savefig(os.path.join(output_dir, "Comparaison_dff_population_late.png"))
# ===================================================================================
# ============================ Moyennes des AUC de df/f ====================================
mean_auc_results = compute_mean_auc_early_late_populations(
    dff, neuron_classif_early, neuron_classif_late, early_social, late_social, sampling_rate
)
# Population Early : AUC moyenne sur 20 premiers vs 20 derniers social touch
pop_early = mean_auc_results["early_population"]
fig_auc_pop_early = plot_early_late_comparison(
    pop_early["auc_first20"], pop_early["auc_last20"], pop_early["types"],
    ylabel="AUC moyenne",
    title="Neurones recrutés Early — AUC moyenne",
    xtick_labels=new_labels
)
fig_auc_pop_early.savefig(os.path.join(output_dir, "Comparaison_AUC_population_early.png"))
# Population Late : AUC moyenne sur 20 premiers vs 20 derniers social touch
pop_late = mean_auc_results["late_population"]
fig_auc_pop_late = plot_early_late_comparison(
    pop_late["auc_first20"], pop_late["auc_last20"], pop_late["types"],
    ylabel="AUC moyenne",
    title="Neurones recrutés Late — AUC moyenne",
    xtick_labels=new_labels
)
fig_auc_pop_late.savefig(os.path.join(output_dir, "Comparaison_AUC_population_late.png"))
# ======================================================================================
# ============================ Moyennes des Peak amplitude de df/f ====================================

mean_peak_results = compute_mean_peak_early_late_populations(
    dff, neuron_classif_early, neuron_classif_late, early_social, late_social, sampling_rate
)

# Population Early : peak amplitude moyen sur 20 premiers vs 20 derniers social touch
pop_early = mean_peak_results["early_population"]
fig_peak_pop_early = plot_early_late_comparison(
    pop_early["peak_first20"], pop_early["peak_last20"], pop_early["types"],
    ylabel="Peak amplitude moyen (dF/F - baseline)",
    title="Neurones recrutés Early — Peak amplitude moyen",
    xtick_labels=new_labels
)
fig_peak_pop_early.savefig(os.path.join(output_dir, "Comparaison_peak_population_early.png"))

# Population Late : peak amplitude moyen sur 20 premiers vs 20 derniers social touch
pop_late = mean_peak_results["late_population"]
fig_peak_pop_late = plot_early_late_comparison(
    pop_late["peak_first20"], pop_late["peak_last20"], pop_late["types"],
    ylabel="Peak amplitude moyen (dF/F - baseline)",
    title="Neurones recrutés Late — Peak amplitude moyen",
    xtick_labels=new_labels
)
fig_peak_pop_late.savefig(os.path.join(output_dir, "Comparaison_peak_population_late.png"))
# ======================================================================================
# # ============================Enregistrement figures====================================
# fig_dff = plot_early_late_comparison(mean_dff_early, mean_dff_late, types_combined,
#                                       ylabel="dF/F moyen", title="dF/F moyen par neurone — Early vs Late")
# fig_dff.savefig(os.path.join(output_dir, "Comparaison_dff_moyen_early_late.png"))

# fig_auc = plot_early_late_comparison(mean_auc_early, mean_auc_late, types_combined,
#                                       ylabel="AUC moyen", title="AUC moyen par neurone — Early vs Late")
# fig_auc.savefig(os.path.join(output_dir, "Comparaison_AUC_moyen_early_late.png"))

# fig_peak = plot_early_late_comparison(mean_peak_early, mean_peak_late, types_combined,
#                                        ylabel="Peak amplitude moyen (dF/F - baseline)",
#                                        title="Peak amplitude moyen par neurone — Early vs Late")
# fig_peak.savefig(os.path.join(output_dir, "Comparaison_peak_moyen_early_late.png"))
# # ======================================================================================

# =========================================================
# CHEVAUCHEMENT EARLY / LATE
# =========================================================
def compute_recruitment_overlap(neuron_classif_early, neuron_classif_late, label=""):
    recruited_early = set(np.where(neuron_classif_early != 0)[0].tolist())
    recruited_late  = set(np.where(neuron_classif_late  != 0)[0].tolist())
    common     = recruited_early & recruited_late
    only_early = recruited_early - recruited_late
    only_late  = recruited_late  - recruited_early
    union      = recruited_early | recruited_late
    n_early = len(recruited_early)
    n_late  = len(recruited_late)
    n_common = len(common)
    n_union  = len(union)
    # Proportions
    prop_of_early_also_late = n_common / n_early if n_early > 0 else np.nan
    prop_of_late_also_early = n_common / n_late if n_late > 0 else np.nan
    jaccard = n_common / n_union if n_union > 0 else np.nan  # indice de chevauchement symétrique
    print(f"\n=== CHEVAUCHEMENT RECRUTEMENT {label} ===")
    print(f"Neurones recrutés Early           : {n_early}")
    print(f"Neurones recrutés Late            : {n_late}")
    print(f"Neurones recrutés Early ET Late    : {n_common}")
    print(f"Neurones recrutés Early UNIQUEMENT : {len(only_early)}")
    print(f"Neurones recrutés Late UNIQUEMENT  : {len(only_late)}")
    print(f"% des recrutés Early aussi recrutés Late : {prop_of_early_also_late*100:.1f}%")
    print(f"% des recrutés Late aussi recrutés Early : {prop_of_late_also_early*100:.1f}%")
    print(f"Indice de Jaccard (chevauchement symétrique) : {jaccard*100:.1f}%")
    return {
        "n_early": n_early,
        "n_late": n_late,
        "n_common": n_common,
        "n_only_early": len(only_early),
        "n_only_late": len(only_late),
        "n_union": n_union,
        "prop_early_also_late": prop_of_early_also_late,
        "prop_late_also_early": prop_of_late_also_early,
        "jaccard": jaccard,
        "common_idx": common,
        "only_early_idx": only_early,
        "only_late_idx": only_late,
    }

overlap_result = compute_recruitment_overlap(neuron_classif_early, neuron_classif_late, label="Habituation")
summary_overlap = pd.DataFrame([{
    "Neurones recrutés Early": overlap_result["n_early"],
    "Neurones recrutés Late": overlap_result["n_late"],
    "Recrutés Early ET Late (communs)": overlap_result["n_common"],
    "Recrutés Early uniquement": overlap_result["n_only_early"],
    "Recrutés Late uniquement": overlap_result["n_only_late"],
    "% Early aussi recrutés Late": overlap_result["prop_early_also_late"] * 100,
    "% Late aussi recrutés Early": overlap_result["prop_late_also_early"] * 100,
    "Indice de Jaccard (%)": overlap_result["jaccard"] * 100,
}])
print(summary_overlap.to_string(index=False))
# ============================Enregistrement figures====================================
summary_overlap.to_excel(os.path.join(output_dir, "recap_chevauchement_early_late.xlsx"), index=False)
# ======================================================================================

plt.show()
