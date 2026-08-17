# ========================================================================================================================================================================================================================================================================================================================================================================
# GO/NOGO
# ========================================================================================================================================================================================================================================================================================================================================================================
Objectif du script

Ce script analyse des données d'imagerie calcique (Suite2p) enregistrées pendant une session de comportement (tâche de discrimination Go/Nogo avec vibrations et social touch), synchronisée avec les logs Bpod (fichier .xls) et les paramètres d'essai (fichier .json). Il identifie les neurones recrutés par le social touch, les classe en excitateurs/inhibiteurs, et étudie leur activité selon le type d'essai comportemental (Hit, Miss, Go-Timeout, False Alarm, Correct Rejection).

Entrées attendues
base_dir/
├── analog.txt              # signal analogique (col 1-2: vibration, col 3-4: social touch)
├── bpod.xls                 # logs Bpod (header ligne 6), colonnes temps/événements
├── params_trial.json        # paramètres par essai (amp, trial_type)
└── suite2p/
    ├── F.npy
    ├── Fneu.npy
    └── iscell.npy

Sortie : base_dir/heatmaps_output/.

Pipeline, étape par étape
1. Prétraitement dF/F

Identique au script Habituation : correction neuropile (0.7×Fneu), lissage gaussien, baseline par fenêtre glissante de variance minimale.

2. Détection et synchronisation des événements

Deux sources indépendantes sont détectées puis recalées l'une sur l'autre :

Vibrations : détectées dans l'analogique (detect_vibrations_analog, pics > seuil) et dans l'Excel (detect_vibrations_excel, séquence Stimulus/StimOff avec amp > 0). Le matching (match_vibrations) apparie chaque vibration Excel à son équivalent analogique le plus proche, ce qui donne un décalage temporel dt réutilisé pour projeter d'autres événements Excel sur l'axe analogique (donc sur l'axe d'imagerie).
Social touch : détectés dans l'analogique (front montant/descendant du signal binaire) et dans l'Excel (marqueur "SocialTouch"), puis appariés dans l'ordre chronologique (match_social_touch).
3. Classification des essais comportementaux

À partir des logs Bpod et du JSON de paramètres (amplitude, type d'essai), le script détecte et projette sur l'axe analogique :

Go-Timeout : essai Go/Go-Touch avec timeout
Miss : essai avec vibration (amp>0) sans reward
Hit : essai avec vibration et reward
False Alarm : essai Nogo-Touch avec timeout
Correct Rejection : essai Nogo-Touch sans timeout

Chaque catégorie est projetée sur l'axe temps du signal analogique via le décalage dt déterminé lors du matching des vibrations.

4. Sélection des social touch par catégorie

get_social_from_trials associe à chaque essai comportemental le(s) social touch qu'il contient (avec le même remove_start=0.3s qu'en Habituation). get_all_social_touch_intervals regroupe l'ensemble (Hit+Miss+Timeout+FA+CR) en une seule liste ordonnée, utilisée pour le recrutement.

5. Recrutement des neurones (méthode par essai / shuffle)

Méthode strictement identique à celle du script Habituation (voir README correspondant) :

Similarité de Dice par essai (compute_per_event_similarity)
Distribution nulle propre à chaque essai, construite en excluant à la fois les vibrations et les social touch de l'espace de tirage (build_valid_shuffle_mask(signal_length, vib_intervals, social_intervals, sampling_rate) — différence avec Habituation qui n'exclut que les social touch, en l'absence de vibrations dans ce protocole)
Seuils low_pct=5, high_pct=95, min_events=2
Diagnostic (diagnose_per_event_significance) avant classification, pour visualiser la distribution du nombre d'essais significatifs par neurone

Une seule classification globale est produite (neuron_classif), sur l'ensemble des social touch de la session (pas de découpage Early/Late comme en Habituation).

6. Quantification de la réponse
AUC dF/F signée par essai (compute_auc_recruited_social)
Peak amplitude relatif à une baseline pré-essai (compute_peak_amplitude_recruited_social)
Ces métriques sont calculées à la fois sur l'ensemble des social touch et en les regroupant par type d'essai comportemental (Hit/Miss/Timeout/FA/CR)
7. Visualisations générées
Heatmaps dF/F et Z-score de toute la session, avec code couleur des événements (vert=vibration, bleu=social touch, rose=Go-Timeout, jaune=Miss, turquoise=Hit, vert clair=Correct Rejection, rouge=False Alarm)
Heatmaps dF/F normalisées localement (replace_local_dff) par événement
Heatmaps de similarité par type d'essai (Go-touch, Nogo-touch, vibrations)
Traces de tous les neurones / des 10 premiers ou d'une sélection aléatoire
AUC des neurones recrutés : heatmap globale, groupée par exc/inh, et par type d'essai
Comparaison neurones recrutés vs non recrutés (Z-score), globale et par type d'essai
Traces des neurones recrutés concaténées sur les social touch, sur les essais complets, et sur les vibrations
Heatmap de peak amplitude
8. Tableau récapitulatif

recap_recrutement_comportement.xlsx : nombre de neurones excitateurs/inhibiteurs/non recrutés sur l'ensemble de la session.

Fichiers de sortie (dans output_dir)
Catégorie	Fichiers
Détection/synchro (console uniquement)	logs de matching vibrations/social touch, comptages Hit/Miss/Timeout/FA/CR
Heatmaps session	dF/F brute, dF/F normalisée localement, avec bandeaux d'événements
Heatmaps par type d'essai	Go-touch, vibrations, Nogo-touch
AUC	heatmap globale, groupée exc/inh, par type d'essai
Traces	tous neurones, neurones recrutés (social touch / essais complets / vibrations)
Peak amplitude	heatmap
Récapitulatif	recap_recrutement_comportement.xlsx
Points de vigilance connus
Beaucoup de code mort en commentaire (anciennes méthodes de recrutement par bootstrap, AUC positive/négative séparée, sélection de 10 neurones aléatoires) — n'affecte pas l'exécution mais alourdit la lecture ; à nettoyer si le script doit être maintenu.
compute_peak_amplitude_recruited_social et get_social_from_trials sont chacune définies deux fois dans le script (la seconde définition écrase la première) — sans conséquence fonctionnelle ici, mais source de confusion en cas de modification future.
Le matching match_social_touch et match_vibrations suppose un ordre chronologique cohérent entre Excel et analogique ; les warnings imprimés en cas de non-match méritent d'être vérifiés avant d'interpréter les résultats (essais perdus silencieusement sinon).
n_shuffles=2000 est correctement fixé ici (contrairement à une version antérieure du script Habituation) — bon niveau pour une estimation fiable des percentiles à 5%/95%.
Le recrutement est calculé une seule fois sur l'ensemble de la session : aucune comparaison temporelle (ex. début vs fin de session) n'est faite ici, contrairement au script Habituation — à ajouter si l'analyse comportementale nécessite une comparaison Early/Late équivalente.
Dépendances

numpy, pandas, matplotlib, scipy, seaborn, openpyxl, ainsi que xlrd ou openpyxl selon le moteur utilisé par pd.read_excel pour lire bpod.xls.

# ========================================================================================================================================================================================================================================================================================================================================================================
# HABITUATION
# ========================================================================================================================================================================================================================================================================================================================================================================

Objectif du script
Ce script analyse des données d'imagerie calcique (Suite2p) enregistrées pendant une session d'habituation à un stimulus de social touch. Il vise à identifier les neurones dont l'activité est significativement liée au social touch ("recrutement"), à les classer en excitateurs/inhibiteurs, et à comparer leur comportement entre le début (20 premiers essais) et la fin (20 derniers essais) de la session, afin d'étudier une éventuelle habituation neuronale.
Entrées attendues

Le script suppose l'arborescence suivante :

base_dir/
├── analog.txt          # signal analogique synchronisé (colonnes: temps/signal social touch en col. 3-4)
└── suite2p/
    ├── F.npy            # fluorescence brute par ROI
    ├── Fneu.npy          # fluorescence neuropile
    └── iscell.npy        # classification cellule/non-cellule de Suite2p

Chemins à adapter en haut du script :

python
base_dir   = r"...\HABITUATION\20260430_M3_testing_synchro"
output_dir = r"...\graphes\HABITUATION\M3"

sampling_rate (30.9609 Hz) doit correspondre à la fréquence d'acquisition réelle du microscope.

Pipeline, étape par étape
1. Prétraitement du signal (dF/F)
Correction de la contamination neuropile : Fcorr = F - 0.7 × Fneu
Lissage gaussien (sigma = 0.1 × sampling_rate)
Estimation de la baseline F0 par fenêtre glissante de plus faible variance (300 frames), pour limiter l'effet des transitoires calciques sur l'estimation
Calcul du dF/F = (Fcorr - F0) / F0
2. Détection des social touch

detect_social_touch_analog détecte les fronts montants/descendants du signal binaire analogique, avec un délai remove_start=0.3s retranché en début d'événement (pour exclure l'artefact de contact initial).

3. Sélection Early / Late

Les 20 premiers et 20 derniers social touch de la session sont extraits (early_social, late_social), filtrés pour exclure les intervalles trop courts (< 50 ms).

4. Recrutement des neurones (méthode par essai / shuffle)

Pour chaque essai social touch, on calcule un coefficient de similarité de Dice entre la trace dF/F du neurone et un masque binaire de l'événement. Cette similarité est comparée à une distribution nulle propre à cet essai, générée en tirant aléatoirement 2000 (ou plus) intervalles de même durée en dehors de tous les social touch de la session.

Un neurone est déclaré :

excitateur (+1) s'il dépasse le seuil haut (percentile high_pct) sur au moins min_events essais,
inhibiteur (-1) s'il est en dessous du seuil bas (percentile low_pct) sur au moins min_events essais,
non recruté (0) sinon.

Cette procédure (compute_recruitment) est appliquée séparément pour Early et Late, donnant deux classifications indépendantes : neuron_classif_early, neuron_classif_late.

Paramètres clés :

n_shuffles : nombre de tirages aléatoires par essai (2000-5000 recommandé ; ⚠️ actuellement fixé à 10 dans le script, ce qui est insuffisant — voir section Points de vigilance)
min_events : nombre minimum d'essais significatifs requis (2 par défaut)
low_pct / high_pct : seuils de significativité (5/95 par défaut)

Une fonction de diagnostic (diagnose_per_event_significance) affiche la distribution du nombre d'essais significatifs par neurone, utile pour ajuster ces seuils.

5. Quantification de la réponse

Pour les neurones recrutés, trois métriques sont calculées par essai :

dF/F moyen signé (partie positive gardée pour les excitateurs, négative pour les inhibiteurs)
AUC (aire sous la courbe, dF/F et version Z-score)
Peak amplitude (maximum/minimum relatif à une baseline pré-essai)
6. Visualisations générées

Le script produit (dans output_dir) :

Heatmap dF/F de toute la session (temps et frames)
Traces de tous les neurones sur la session
Heatmaps de similarité Z-score (globale, Early, Late)
Heatmaps AUC (dF/F et Z-score) pour Early/Late, sur leurs essais propres et sur toute la séance
Comparaison recrutés vs non recrutés (Z-score)
Courbes d'évolution de l'AUC au fil de la séance (Early vs Late, tous neurones et séparé exc/inh)
Évolution de l'AUC Z-score sur essais échantillonnés (pas de 10)
Traces d'activité des neurones recrutés (Early/Late)
Heatmaps de peak amplitude
7. Tableaux récapitulatifs (Excel)
recap_recrutement_habituation.xlsx : nombre de neurones excitateurs/inhibiteurs/non recrutés, Early vs Late
recap_chevauchement_early_late.xlsx : chevauchement du recrutement entre Early et Late (effectifs, % et indice de Jaccard)
8. Comparaison par neurone Early vs Late

Pour les neurones recrutés dans au moins un des deux groupes (union), trois graphes "pente" comparent dF/F moyen, AUC moyen et peak amplitude moyen entre Early et Late, avec code couleur exc (rouge) / inh (bleu).

9. Chevauchement du recrutement

compute_recruitment_overlap calcule :

le nombre de neurones recrutés Early ET Late (communs), Early uniquement, Late uniquement
le % de recouvrement dans chaque sens
l'indice de Jaccard (mesure symétrique du chevauchement, communs / union)
Fichiers de sortie (dans output_dir)
Fichier	Contenu
df_f_time_Heatmap.png, df_f_frames_Heatmap.png	Heatmap dF/F session complète
Activite_tous_neurones_toute_la_session.png	Traces de tous les neurones
Z-score*.png	Heatmaps Z-score (global, Early, Late)
AUC_df_f_*.png, AUC_Z-score_*.png	Heatmaps et courbes AUC
Activite_early/late_20_*.png	Traces des neurones recrutés
Peak_amplitude_early/late_20.png	Heatmaps peak amplitude
Comparaison_*_moyen_early_late.png	Graphes pente par neurone (dF/F, AUC, peak)
recap_recrutement_habituation.xlsx	Tableau récap. recrutement
recap_chevauchement_early_late.xlsx	Tableau récap. chevauchement
Points de vigilance connus
n_shuffles=10 dans les deux appels à compute_recruitment — beaucoup trop faible pour estimer des percentiles à 5%/95% de façon fiable. À remonter à 1000-5000 avant toute analyse définitive (attention au temps de calcul : boucle n_events × n_shuffles × n_neurons).
Le script contient deux fonctions de construction de distribution nulle : build_null_distribution_per_event (utilisée, correcte, par-essai) et build_null_distribution_excluding_events (ancienne version globale, conservée mais non utilisée — peut être supprimée si inutile ailleurs).
En cas de neurone recruté avec un type différent (excitateur en Early, inhibiteur en Late), build_combined_classif donne la priorité au type Early — à adapter selon l'interprétation biologique souhaitée.
Les fonctions de plot retournent None si aucun neurone n'est recruté pour un groupe ; les appels sont protégés (if ... is not None) mais un recrutement nul rend l'essentiel des figures vides.
Dépendances

numpy, pandas, matplotlib, scipy, seaborn, openpyxl (pour l'export .xlsx).