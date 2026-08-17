Inventoried experimental scripts and orchestrated comprehensive documentation framework
Inventoried experimental scripts and orchestrated comprehensive documentation framework
 — Pipeline d'analyse comportementale Go/NoGo (Batch2, Social Anxiety)
Vue d'ensemble
Ce pipeline traite des données comportementales issues d'une tâche de discrimination Go/NoGo (avec ou sans social touch) enregistrées via Bpod, pour plusieurs souris et plusieurs sessions. Il construit un tableau d'essais structuré, ajuste des modèles statistiques (logistiques, mixtes) sur la performance et le comportement de léchage, puis génère des graphiques et des tests statistiques comparant les génotypes WT et SIS.

Le protocole expérimental est décrit dans SocialAnxiety_Batch2_testing.doc (document Word, 2 pages) — à consulter pour le détail du paradigme comportemental.

Structure du pipeline
Tableaux_A.py          →  extraction et structuration des essais (bas niveau)
       │
       ▼
Coefficients_A.py       →  modèles statistiques sur Performance / Choice
Coefficients_ITI2.py    →  modèles statistiques sur l'effet de l'ITI2
       │
       ▼
Program_A.py            →  script principal : parcourt les dossiers, construit
                            les tableaux, ajuste tous les modèles, exporte les
                            fichiers Excel
       │
       ▼
Barplots_trials.py       →  génère tous les graphiques (coefficients, effets
                            marginaux, licking time) à partir des objets créés
                            par Program_A.py
Barplots_ITI2.py         →  variante allégée (entièrement commentée / obsolète)
       │
       ▼
stats.py                 →  résumé statistique détaillé (β, SE, z, p, IC95%)
                            de tous les modèles, + tests WT vs SIS sur le
                            licking time normalisé
Point important : Barplots_trials.py et stats.py font from Program_A import * — ils ne fonctionnent pas seuls, ils réexécutent tout Program_A.py (donc tout le chargement de données et tous les modèles) à l'import. Il n'y a pas de découplage entre calcul et affichage : lancer stats.py relance therefore aussi Program_A.py et Barplots_trials.py en entier.

Détail par fichier
Tableaux_A.py
Fonctions bas niveau de lecture/structuration :

catch_info_xls : lit un fichier .xls Bpod (sans header).
catch_info_json : va chercher, dans le même dossier, le fichier .json contenant les paramètres d'essai (fréquences, amplitudes).
search_nb_newtrial / search_nb_endtrial : indices des lignes "New trial" / "The trial ended" dans le fichier Excel — délimitent chaque essai.
trial_type, touch, reaction, mouse_reaction, outcome, performance, ITI2, choice, licking_time : extraient, essai par essai, le type d'essai (Go/Go-Touch/Nogo/Nogo-Touch), la présence d'un social touch, la réaction du setup (Reward/Timeout), la réaction de la souris (Lick/No Lick), l'issue comportementale (Hit/Miss/Go-Timeout/Correct Rejection/False Alarm), la performance binaire, le choix (0/1), l'ITI2 (durée entre la fin d'une éventuelle perturbation et le stimulus), et le temps de léchage après le social touch.
previous_trial : ajoute les colonnes décalées d'un essai (Prev_*) pour analyser l'effet de l'essai précédent.
genotype_souris : déduit le génotype (WT/SIS) à partir du nom de dossier de la souris (présence de "SIS").
trial_table : assemble tout ça en un DataFrame, un essai par ligne.
Coefficients_A.py
Modèles portant sur l'essai précédent :

logistic_previous_outcome_trial_perf : régression logistique Prev_Perf ~ Performance.
logistic_previous_lick_choice : régression logistique Prev_Lick ~ Choice.
mixed_previous_perf / mixed_previous_lick : modèles mixtes (effet aléatoire souris) équivalents.
Une fonction commentée (logistic_previous_outcome_trial_choice) est laissée de côté — vraisemblablement une tentative abandonnée.
Coefficients_ITI2.py
Modèles portant sur l'effet de l'ITI2 :

logistic_previous_outcome_ITI2 : Performance ~ ITI2.
logistic_iti2_genotype : Hit ~ ITI2 × Genotype + C(Mouse) — teste l'interaction ITI2/génotype avec effet fixe souris.
prepare_iti2_dataframe : construit, pour chaque type d'essai (Go, Go-Touch, Nogo, Nogo-Touch), la variable binaire Success (Hit pour Go/Go-Touch, Correct Rejection pour Nogo/Nogo-Touch).
fit_models_par_souris : ajuste un GLM binomial (Success ~ ITI2) séparément pour chaque souris.
build_all_iti2_models : orchestre l'ajustement par souris pour les 4 types d'essai, retourne les DataFrames et modèles associés.
Program_A.py (script principal)
Parcourt racine (chemin local à adapter — r"C:\Users\Asus\Stage 1A\SocialAnxiety_Batch2_testing"), puis chaque sous-dossier souris/session, cherche le .xls.
Pour chaque session : construit le tableau d'essais (trial_table), ajoute Génotype/Session/Mouse, calcule les colonnes Prev_*.
Exporte par session : Trial_table_{session}.xlsx (tableau complet) et Analysis_by_trial_type_{session}.xlsx (4 feuilles Go/GoTouch/NoGo/NoGoTouch, colonnes sélectionnées).
Calcule un résumé Performance Touch vs NoTouch par souris/session → Performance_Touch_vs_NoTouch.xlsx.
Concatène toutes les sessions (df_total), sépare par type d'essai et par génotype (WT/SIS).
Ajuste, pour chaque combinaison (type d'essai × génotype) :
modèles Performance ~ Prev_Perf → Coefficients_WT_trials_perf.xlsx, Coefficients_SIS_trials_perf.xlsx
modèles Choice ~ Prev_Lick → Coefficients_WT_trials_choice.xlsx, Coefficients_SIS_trials_choice.xlsx
modèles Performance ~ ITI2, toutes souris confondues → Coefficients_ITI2.xlsx
Prépare un sous-jeu df_go_touch_hitmiss (essais Go-Touch Hit/Miss uniquement, sans Timeout), ajuste un modèle Hit ~ ITI2, calcule des corrélations de Pearson ITI2/Performance (globale, première session, par génotype).
Ajuste logistic_iti2_genotype (interaction ITI2 × Genotype) → ITI2_Genotype_MixedModel.xlsx.
Ajuste un GLM par souris pour chaque type d'essai (build_all_iti2_models).
Tous les objets modèles (model_go_wt_perf, modeli_go, models_iti2_souris, df_total, etc.) restent en mémoire au niveau module et sont réutilisés tels quels par Barplots_trials.py et stats.py via import *.

Barplots_trials.py
Génère l'ensemble des figures, sauvegardées dans ../Graphes/Graphes_Batch2/ (chemin relatif au script) :

plot_coeff : forest plot (coefficient ± IC95%) d'une variable, un point par type d'essai — utilisé pour Prev_Perf→Performance et Prev_Lick→Choice, séparément WT/SIS.
plot_marginal_continu : effet marginal prédit d'une variable continue (ITI2 sur P(Hit), Go-Touch Hit/Miss uniquement).
plot_marginal_binaire : effet marginal d'une variable binaire (Prev_Perf, Prev_Lick) — un point à 0 et un à 1, pour chaque combinaison type d'essai × génotype (16 figures).
plot_iti2_genotype : courbes prédites Hit~ITI2 séparées WT/SIS (modèle d'interaction).
plot_une_souris / plot_iti2_par_souris : pour chaque souris et chaque type d'essai, courbe logistique ITI2→Performance avec tangentes en début/milieu/fin de plage ITI2 et rapport de pentes βdébut/βfin (indicateur d'un effet non linéaire/de saturation).
plot_session1_iti2_logistique : même analyse mais restreinte à la session 1, avec tangentes en 3 points (début/milieu/fin) et βmilieu en plus.
Section "Licking Time" : nombreuses fonctions de visualisation du temps de léchage par souris/session/type d'essai (plot_licktime_moyennes, plot_licking_time_par_socialtouch, plot_licktime_par_session_genotype, plot_licktime_by_session_genotype, plot_session_means, plot_licktime_touch_regroupe, plot_moyenne_licktime_touch_regroupe), avec code couleur WT=bleu/SIS=rouge, et export final licking_time_summary.xlsx (moyennes par souris/session, feuilles Go-Touch/Nogo-Touch/Touch regroupé).
Barplots_ITI2.py
Fichier entièrement commenté — ancienne version simplifiée de plot_coeff limitée à la variable ITI2 et aux 4 modèles modeli_*. Conservé pour référence mais non exécuté (aucun code actif).

stats.py
Réexécute Program_A.py et Barplots_trials.py (via import), puis :

Imprime, pour chaque modèle Performance et chaque modèle Choice (WT/SIS × Go/GoTouch/Nogo/NogoTouch), le détail statistique complet de chaque coefficient (β, SE, z, p, IC95%).
Compare le licking time normalisé (par souris, ramené à sa propre moyenne) entre WT et SIS, session par session et type d'essai par type d'essai (comparaison_WT_SIS_normalise, analyse_WT_SIS_normalise) : test de normalité (Shapiro) puis test paramétrique (t-test) ou non paramétrique (Mann-Whitney) selon le résultat.
Même comparaison mais sur le licking time normalisé toutes conditions touch confondues, par session uniquement (comparaison_WT_SIS_normalise_social_touch, analyse_WT_SIS_normalise_social_touch).
Contient un grand bloc de fonctions commentées (anciennes versions des comparaisons WT/SIS, tables intermédiaires) — code mort conservé, sans effet à l'exécution.
Fichiers de sortie principaux
Fichier	Origine	Contenu
Trial_table_{session}.xlsx	Program_A	Tableau complet des essais, par session
Analysis_by_trial_type_{session}.xlsx	Program_A	Essais filtrés par type, colonnes clés
Performance_Touch_vs_NoTouch.xlsx	Program_A	Performance moyenne Touch vs NoTouch
Coefficients_WT/SIS_trials_perf.xlsx	Program_A	Coefficients Prev_Perf→Performance
Coefficients_WT/SIS_trials_choice.xlsx	Program_A	Coefficients Prev_Lick→Choice
Coefficients_ITI2.xlsx	Program_A	Coefficients Performance~ITI2
ITI2_Genotype_MixedModel.xlsx	Program_A	Coefficients interaction ITI2×Genotype
licking_time_summary.xlsx	Barplots_trials	Moyennes de licking time par souris/session
Figures .png	Barplots_trials	Voir détail par fonction ci-dessus, dans ../Graphes/Graphes_Batch2/

Comment lancer le pipeline

Barplots_trials.py : génère les graphiques ET tous les fichiers Excel (relance Program_A.py par import).
Program_A.py : génère uniquement les fichiers Excel, sans les graphiques.
stats.py : statistiques supplémentaires (résumé détaillé des modèles + tests WT/SIS), relance tout le reste par import.
Deux chemins locaux sont à adapter avant exécution :

racine dans Program_A.py (ligne 10) : chemin vers le dossier racine des données (SocialAnxiety_Batch2_testing).
Le chemin de sortie des graphiques dans Barplots_trials.py (DOSSIER_GRAPHS, calculé relativement à l'emplacement du script — vérifier qu'il pointe bien où souhaité).
Points de vigilance connus
Aucun découplage calcul/affichage : Barplots_trials.py et stats.py réexécutent l'intégralité du chargement de données et de l'ajustement des modèles à chaque lancement (pas de sauvegarde intermédiaire des modèles/objets) — coûteux si les données sont volumineuses.
Chemins codés en dur (racine dans Program_A.py) — à adapter pour chaque poste/utilisateur.
Fichiers .pyc fournis en plus des .py (compilés pour CPython 3.14) — probablement des artefacts de cache (__pycache__), non nécessaires si les .py sont présents ; à ne pas committer/partager habituellement.
Beaucoup de code mort en commentaire dans Coefficients_A.py, Program_A.py, Barplots_trials.py, stats.py, et Barplots_ITI2.py (intégralement commenté) — anciennes approches conservées à titre de référence, sans impact sur l'exécution mais alourdissant la lecture.
Program_A.py, ligne 39-44 : la boucle sur les fichiers du dossier session ne garde que la dernière valeur de ma_path trouvée se terminant par .xls — si un dossier de session contient plusieurs fichiers .xls, seul le dernier de la liste os.listdir (ordre non garanti) est traité.
Le fichier SocialAnxiety_Batch2_testing.doc est un document Word (ancien format .doc), vraisemblablement une description du protocole expérimental — à ouvrir avec Word ou un lecteur compatible pour le contexte scientifique complet, non lu automatiquement par le pipeline.
Dépendances
pandas, numpy, matplotlib, statsmodels, scipy (stats.pearsonr, stats.shapiro, stats.ttest_ind, stats.mannwhitneyu, integrate.trapezoid), json, os, ast. Lecture des .xls : moteur xlrd ou équivalent selon la version de pandas.