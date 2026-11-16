# AF-Loss : focal loss adaptative par classe pour la détection d'intrusion réseau explicable

Projet du module Intelligence Artificielle, EST Kénitra (DUT IDRS), 2026-2027.

## Idée
AF-Loss associe à chaque classe son propre poids (alpha_c) et son propre
exposant de focalisation (gamma_c), selon la rareté de la classe.
SHAP explique les décisions du modèle.

## Données
UNSW-NB15 (partition officielle). Télécharger le jeu de données depuis le site
officiel, puis le placer dans le dossier `data/`.

## Méthodes comparées
- MLP + cross-entropy
- MLP + focal loss (gamma = 2)
- MLP + AF-Loss (beta = 0,5 et beta = 1)
- Forêt aléatoire (100 arbres)

## Résultats (moyenne sur 5 graines)
| Méthode | Rappel macro | F1 macro |
|---|---|---|
| Cross-entropy | 0,441 | 0,396 |
| Focal loss | 0,458 | 0,406 |
| AF-Loss (beta = 0,5) | 0,504 | 0,403 |
| AF-Loss (beta = 1) | 0,549 | 0,356 |
| Forêt aléatoire | 0,510 | 0,461 |

AF-Loss augmente le rappel macro et détecte la classe Worms (rappel 0,78),
mais la F1 macro n'est pas améliorée.

## Utilisation
pip install -r requirements.txt

## Auteure
Khadija ESSARGHINI, EST Kénitra
