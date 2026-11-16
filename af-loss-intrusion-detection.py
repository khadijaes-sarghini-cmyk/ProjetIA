# ============================================================
# Projet IA : AF-Loss pour la detection d'intrusion (UNSW-NB15)
#
# Ce script fait TOUT, dans l'ordre :
#   1. Collecte des donnees
#   2. Nettoyage (colonnes, doublons, valeurs aberrantes, valeurs manquantes)
#   3. Transformation (encodage, separation train/validation, normalisation)
#   4. Choix de beta sur la VALIDATION (pas sur le test)
#   5. Entrainement de 5 modeles avec 5 graines
#   6. Explication avec SHAP + modele allege (top-k attributs)
#   7. Resultats : Table 2, rappel par classe, figures, fichiers CSV
#
# Utilisation : mettre ce fichier dans le meme dossier que
#   UNSW_NB15_training-set.csv  et  UNSW_NB15_testing-set.csv
# puis lancer :  py projet_afloss_complet.py
#
# Bibliotheques : py -m pip install pandas numpy scikit-learn shap matplotlib
#                 py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# ============================================================
import copy
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # on enregistre les figures en PNG (pas de fenetre)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
T_DEBUT = time.time()

# ============================================================
# ETAPE 1 : COLLECTE
# ============================================================
print("=" * 60)
print("ETAPE 1 : COLLECTE")
print("=" * 60)
SEED = 42
np.random.seed(SEED)

train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

print("dim avant pretrait train :", train.shape)
print("dim avant pretrait test  :", test.shape)
n_train_brut = train.shape[0]
n_test = test.shape[0]

# ============================================================
# ETAPE 2 : NETTOYAGE
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 2 : NETTOYAGE")
print("=" * 60)

# 2.1 Suppression des colonnes inutiles
print("\n--- sous etape 2.1 : suppression des colonnes inutiles ---")
print("id    : simple numero de ligne (aucune information)")
print("label : version 0/1 de attack_cat (on garde attack_cat, 10 classes)")
train = train.drop(columns=["id", "label"])
test = test.drop(columns=["id", "label"])
print("dim apres suppression des colonnes :", train.shape)

# 2.2 Suppression des doublons (train seulement)
print("\n--- sous etape 2.2 : suppression des doublons (train seulement) ---")
nb_doublons = int(train.duplicated().sum())
print("nombre de doublons dans le train :", nb_doublons,
      f"({100 * nb_doublons / len(train):.1f} %)")
avant = train["attack_cat"].value_counts()
train = train.drop_duplicates()
apres = train["attack_cat"].value_counts()
print("dim apres suppression des doublons :", train.shape)
repartition = pd.DataFrame({
    "avant": avant,
    "apres": apres,
    "pourcentage_apres": (100 * apres / apres.sum()).round(2),
}).sort_values("apres", ascending=False)
print(repartition.to_string())
print("Le test (officiel) n'est pas modifie : on compare ainsi avec la litterature.")

# 2.3 Valeurs aberrantes (methode IQR, exemple sur sbytes)
print("\n--- sous etape 2.3 : valeurs aberrantes (methode IQR sur sbytes) ---")
Q1 = train["sbytes"].quantile(0.25)
Q3 = train["sbytes"].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
print(f"Q1 = {Q1} | Q3 = {Q3} | IQR = {IQR}")
print(f"intervalle normal : [{lim_inf}, {lim_sup}]")
VA = (train["sbytes"] < lim_inf) | (train["sbytes"] > lim_sup)
nb_va = int(VA.sum())
print(f"nombre de valeurs aberrantes dans sbytes : {nb_va} ({100 * nb_va / len(train):.1f} %)")
va_par_classe = train.loc[VA, "attack_cat"].value_counts()
va_pct = (100 * va_par_classe / train["attack_cat"].value_counts()).round(1)
print("repartition par classe (nombre | % de la classe) :")
print(pd.DataFrame({"VA": va_par_classe, "pourcentage_de_la_classe": va_pct}).to_string())
print("=> DECISION : on NE supprime PAS ces valeurs. Ce sont des valeurs reelles du trafic")
print("   (gros transferts), pas des erreurs de mesure. Les supprimer ferait perdre")
print(f"   {100 * nb_va / len(train):.1f} % des donnees, dont des exemples de classes rares.")

# 2.4 Valeurs manquantes
print("\n--- sous etape 2.4 : valeurs manquantes (imputation) ---")
cat_cols = ["proto", "service", "state"]
num_cols = [c for c in train.columns if c not in cat_cols + ["attack_cat"]]
print("valeurs manquantes avant : train =", int(train.isna().sum().sum()),
      "| test =", int(test.isna().sum().sum()))

imputer_mean = SimpleImputer(strategy="mean")          # numeriques : moyenne
train[num_cols] = imputer_mean.fit_transform(train[num_cols])
test[num_cols] = imputer_mean.transform(test[num_cols])
imputer_mode = SimpleImputer(strategy="most_frequent")  # categorielles : mode
train[cat_cols] = imputer_mode.fit_transform(train[cat_cols])
test[cat_cols] = imputer_mode.transform(test[cat_cols])
print("valeurs manquantes apres imputation :",
      int(train.isna().sum().sum() + test.isna().sum().sum()))

# ============================================================
# ETAPE 3 : TRANSFORMATION
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 3 : TRANSFORMATION")
print("=" * 60)

# 3.1 Encodage
print("\n--- sous etape 3.1 : encodage ---")
le = LabelEncoder()
y_train_full = le.fit_transform(train["attack_cat"])
y_test = le.transform(test["attack_cat"])
classes = list(le.classes_)
print("cible (attack_cat) -> entiers 0 a", len(classes) - 1, ":", classes)

X_train_full = train.drop(columns=["attack_cat"])
X_test = test.drop(columns=["attack_cat"])
d_avant_encodage = X_train_full.shape[1]
X_train_full = pd.get_dummies(X_train_full, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)
feature_names = list(X_train_full.columns)
print(f"one-hot sur {cat_cols} : {d_avant_encodage} attributs -> {len(feature_names)} attributs")

# 3.2 Separation train / validation
print("\n--- sous etape 3.2 : separation train / validation (85 % / 15 %) ---")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15,
    stratify=y_train_full, random_state=SEED)
X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_test.copy()
print("separation stratifiee : chaque classe garde la meme proportion dans train et validation")

# 3.3 Normalisation (MinMax), ajustee sur le train seulement
print("\n--- sous etape 3.3 : normalisation MinMax (ajustee sur le train seulement) ---")
scaler = MinMaxScaler()
X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_te[num_cols] = scaler.transform(X_te[num_cols])
print("valeurs ramenees entre 0 et 1 ; min/max calcules sur le train (pas de fuite de donnees)")

X_tr = np.ascontiguousarray(X_tr.astype("float32").values)
X_val = np.ascontiguousarray(X_val.astype("float32").values)
X_te = np.ascontiguousarray(X_te.astype("float32").values)

C = len(classes)
d = X_tr.shape[1]
counts = np.bincount(y_tr, minlength=C)
print("\nn_c (exemples par classe dans le train) :")
print(pd.Series(dict(zip(classes, counts))).to_string())
print("\ndim apres transformation : X_tr", X_tr.shape, "| X_val", X_val.shape, "| X_te", X_te.shape)

print("\n----- RESUME POUR LA SECTION 4.1 DE L'ARTICLE -----")
print(f"train brut                : {n_train_brut}")
print(f"train apres doublons      : {len(y_train_full)}")
print(f"entrainement / validation : {len(y_tr)} / {len(y_val)}")
print(f"test officiel             : {n_test}")
print(f"attributs                 : {d_avant_encodage} -> {d} apres one-hot")
print(f"classes                   : {C}")
repartition.to_csv("repartition_classes.csv")

# ============================================================
# MODELES ET FONCTIONS
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nmateriel utilise (device) :", device)

SEEDS = [42, 43, 44, 45, 46]
EPOCHS, BATCH, LR, WD = 20, 512, 1e-3, 1e-5   # WD = lambda (regularisation L2)
G_MIN, G_MAX = 1.0, 3.0
K_TOP = 30                                    # nombre d'attributs gardes apres SHAP


class FocalLoss(nn.Module):
    """Focal loss classique : un seul gamma pour toutes les classes."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        pt = logp.exp()
        return -((1 - pt) ** self.gamma * logp).mean()


class AFLossBeta(nn.Module):
    """AF-Loss : alpha_c et gamma_c propres a chaque classe (alpha_c ** beta)."""

    def __init__(self, counts, beta=1.0, g_min=1.0, g_max=3.0):
        super().__init__()
        n = torch.as_tensor(counts, dtype=torch.float32)
        alpha = (n.sum() / (len(n) * n)) ** beta
        self.register_buffer("alpha", alpha)
        self.register_buffer("gamma", g_min + (g_max - g_min) * (1 - n / n.max()))

    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        pt = logp.exp()
        return -(self.alpha[y] * (1 - pt) ** self.gamma[y] * logp).mean()


def make_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, C)).to(device)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb = torch.from_numpy(X[i:i + 8192]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_mlp(loss_fn, seed, Xtr=None, Xv=None):
    Xtr = X_tr if Xtr is None else Xtr
    Xv = X_val if Xv is None else Xv
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_mlp(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y_tr).long())
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen)
    best_f1, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        f1v = f1_score(y_val, predict(model, Xv), average="macro", zero_division=0)
        if f1v > best_f1:  # on garde la meilleure epoch (validation)
            best_f1, best_state = f1v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def scores(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred) * 100,
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "rec_cls": recall_score(y_true, y_pred, average=None, labels=list(range(C)), zero_division=0),
    }


# ============================================================
# ETAPE 4 : CHOIX DE BETA SUR LA VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 4 : CHOIX DE BETA (sur la validation, graine 42)")
print("=" * 60)
print("beta = exposant applique a alpha_c : 0 = pas de poids alpha, 1 = poids complet.")
val_f1 = {}
for b in [0.0, 0.5, 1.0]:
    m = train_mlp(AFLossBeta(counts, b).to(device), 42)
    val_f1[b] = f1_score(y_val, predict(m, X_val), average="macro", zero_division=0)
    print(f"beta = {b} | F1 macro (validation) = {val_f1[b]:.3f}", flush=True)
BETA = max(val_f1, key=val_f1.get)
print("=> beta retenu :", BETA)

# ============================================================
# ETAPE 5 : ENTRAINEMENT (5 graines)
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 5 : ENTRAINEMENT DES MODELES (5 graines)")
print("=" * 60)
NAME_AF = f"MLP + AF-Loss (beta={BETA})"
noms = ["Foret aleatoire", "MLP + cross-entropy", "MLP + focal loss"]
if BETA != 1.0:
    noms.append("MLP + AF-Loss (beta=1)")
noms.append(NAME_AF)
results = {n: [] for n in noms}
model_af = None  # modele AF-Loss (graine 42), explique ensuite avec SHAP

for seed in SEEDS:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    rf.fit(X_tr, y_tr)
    results["Foret aleatoire"].append(scores(y_test, rf.predict(X_te)))

    losses = {
        "MLP + cross-entropy": nn.CrossEntropyLoss(),
        "MLP + focal loss": FocalLoss(gamma=2.0),
    }
    if BETA != 1.0:
        losses["MLP + AF-Loss (beta=1)"] = AFLossBeta(counts, 1.0).to(device)
    losses[NAME_AF] = AFLossBeta(counts, BETA).to(device)

    for name, lf in losses.items():
        model = train_mlp(lf, seed)
        results[name].append(scores(y_test, predict(model, X_te)))
        if name == NAME_AF and seed == SEEDS[0]:
            model_af = model
    print(f"graine {seed} terminee en {time.time() - t0:.0f} s", flush=True)

# ============================================================
# ETAPE 6 : EXPLICATION AVEC SHAP + MODELE ALLEGE
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 6 : EXPLICATION AVEC SHAP")
print("=" * 60)
print("SHAP donne, pour chaque decision du modele, la contribution de chaque attribut.")
print("On l'applique au modele AF-Loss (graine 42), sur des exemples de la VALIDATION")
print("(10 par classe) afin de ne pas utiliser le test pour choisir les attributs.")

rng = np.random.RandomState(SEED)
idx_expl = np.concatenate([
    rng.choice(np.where(y_val == c)[0], size=min(10, int((y_val == c).sum())), replace=False)
    for c in range(C)])
X_explain = X_val[idx_expl]


def f_proba(Z):
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(device)
        return torch.softmax(model_af(t), dim=1).cpu().numpy()


t0 = time.time()
background = shap.kmeans(X_tr, 30)
explainer = shap.KernelExplainer(f_proba, background)
sv = explainer.shap_values(X_explain, nsamples=1000)
if isinstance(sv, list):
    sv = np.stack(sv, axis=-1)          # (exemples, attributs, classes)
sv = np.asarray(sv)
if sv.ndim == 3 and sv.shape[0] == C and sv.shape[2] != C:
    sv = np.transpose(sv, (1, 2, 0))
print(f"valeurs SHAP calculees en {time.time() - t0:.0f} s | forme : {sv.shape}")

imp_global = np.abs(sv).mean(axis=(0, 2))   # importance moyenne sur toutes les classes
imp_classe = np.abs(sv).mean(axis=0)        # attributs x classes
imp_df = pd.DataFrame(imp_classe, index=feature_names, columns=classes)
imp_df["GLOBAL"] = imp_global
imp_df = imp_df.sort_values("GLOBAL", ascending=False)
imp_df.to_csv("shap_importance.csv")

print("\nTop 15 attributs les plus importants (global) :")
print(imp_df["GLOBAL"].head(15).round(4).to_string())
for cl in ["Worms", "Shellcode", "Backdoor"]:
    if cl in imp_df.columns:
        print(f"\nTop 5 attributs pour la classe {cl} :")
        print(imp_df[cl].sort_values(ascending=False).head(5).round(4).to_string())

# Figure 1 : importance globale (top 15)
top15 = imp_df["GLOBAL"].head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top15.index, top15.values)
plt.xlabel("Importance moyenne |SHAP|")
plt.title("Attributs les plus importants (modele AF-Loss)")
plt.tight_layout()
plt.savefig("shap_importance_globale.png", dpi=200)
plt.close()

# Figure 2 : importance par classe (top 10, barres empilees)
top10 = imp_df.head(10)[classes][::-1]
top10.plot(kind="barh", stacked=True, figsize=(9, 6))
plt.xlabel("Importance moyenne |SHAP|")
plt.title("Contribution des attributs par classe (top 10)")
plt.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("shap_importance_par_classe.png", dpi=200)
plt.close()
print("\nFigures enregistrees : shap_importance_globale.png, shap_importance_par_classe.png")

# Modele allege : on ne garde que les K_TOP attributs les plus importants
top_idx = [feature_names.index(n) for n in imp_df.index[:K_TOP]]
Xtr_k = np.ascontiguousarray(X_tr[:, top_idx])
Xval_k = np.ascontiguousarray(X_val[:, top_idx])
Xte_k = np.ascontiguousarray(X_te[:, top_idx])
NAME_K = f"MLP + AF-Loss + SHAP top-{K_TOP}"
results[NAME_K] = []
print(f"\nModele allege : {d} attributs -> {K_TOP} attributs")
for seed in SEEDS:
    mk = train_mlp(AFLossBeta(counts, BETA).to(device), seed, Xtr_k, Xval_k)
    results[NAME_K].append(scores(y_test, predict(mk, Xte_k)))
    print(f"graine {seed} (modele allege) terminee", flush=True)

# ============================================================
# ETAPE 7 : RESULTATS
# ============================================================
print("\n" + "=" * 60)
print("ETAPE 7 : RESULTATS (moyenne +- ecart-type sur 5 graines, jeu de test)")
print("=" * 60)
rows = []
for name, lst in results.items():
    row = {"Methode": name}
    for k, lab in [("acc", "Accuracy (%)"), ("prec", "Precision"),
                   ("rec", "Rappel"), ("f1", "F1 macro")]:
        v = np.array([r[k] for r in lst])
        row[lab] = f"{v.mean():.2f} +- {v.std():.2f}" if k == "acc" else f"{v.mean():.3f} +- {v.std():.3f}"
    rows.append(row)

print("\n===== TABLE 2 =====")
print(pd.DataFrame(rows).to_string(index=False))

print("\n===== RAPPEL PAR CLASSE =====")
rec = pd.DataFrame(
    {name: np.mean([r["rec_cls"] for r in lst], axis=0) for name, lst in results.items()},
    index=classes).round(3)
print(rec.to_string())

pd.DataFrame(rows).to_csv("table2.csv", index=False)
rec.to_csv("rappel_par_classe.csv")
print("\nFichiers enregistres dans le dossier du script :")
print("  table2.csv, rappel_par_classe.csv, repartition_classes.csv, shap_importance.csv,")
print("  shap_importance_globale.png, shap_importance_par_classe.png")
print(f"\nDuree totale : {(time.time() - T_DEBUT) / 60:.1f} minutes")
