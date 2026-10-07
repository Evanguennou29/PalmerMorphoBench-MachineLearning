# PalmerMorphoBench

**Benchmark de classification d'espèces à partir de mesures morphométriques.** Ce projet Python reproductible compare quatre classifieurs pour prédire l'espèce d'un manchot à partir de quatre mesures physiques. Le [notebook](notebooks/comparaison_modeles.ipynb) contient l'exploration, les métriques, une matrice de confusion et les conclusions.

## Données et droits

Le fichier [`data/penguins.csv`](data/penguins.csv) contient 344 observations issues du projet [palmerpenguins](https://github.com/allisonhorst/palmerpenguins/blob/main/inst/extdata/penguins.csv). Les données ont été collectées par Kristen Gorman et le programme Palmer Station LTER et sont [diffusées sous CC0](https://github.com/allisonhorst/palmerpenguins#license). SHA-256 du CSV inclus : `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`.

Référence scientifique : Gorman KB, Williams TD, Fraser WR (2014), *Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis)*, PLOS ONE 9(3): e90081. https://doi.org/10.1371/journal.pone.0090081

## Méthode

- Cible : `species` (Adelie, Chinstrap, Gentoo).
- Variables : longueur et profondeur du bec, longueur de nageoire, masse corporelle. L'île, le sexe et l'année sont exclus pour limiter les raccourcis liés au lieu ou à la période de collecte et garder un cas d'usage centré sur la morphologie.
- Séparation stratifiée : 80 % entraînement, 20 % test, graine 42.
- Sélection : validation croisée stratifiée à cinq plis **sur l'entraînement seulement**, selon le F1 macro moyen. Les imputations et normalisations sont ajustées à l'intérieur de chaque pli via des pipelines.
- Modèles : classe majoritaire, régression logistique, 7 plus proches voisins, forêt aléatoire.
- Rapport final : accuracy, balanced accuracy, F1 macro, rapport par classe et matrice de confusion sur le test séparé.

Le test est présenté pour chaque modèle à titre descriptif. Le choix du modèle est fixé par la validation croisée avant l'inspection du test.

## Résultat observé

Les 7 plus proches voisins sont sélectionnés : F1 macro de **0,976 ± 0,037** en validation croisée sur 275 observations d'entraînement, contre **0,205** pour la baseline majoritaire. Sur les 69 observations de test, leur F1 macro, balanced accuracy et accuracy valent **1,000**. Ce test parfait reste une estimation sur un petit ensemble ; la validation croisée est la référence pour le choix du modèle. Les résultats détaillés figurent dans [`results/metrics.json`](results/metrics.json) et dans le notebook.

## Exécution

Python 3.12 recommandé :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m penguins_ml.experiment
python -m unittest discover -s tests -v
```

Ouvrir ensuite `notebooks/comparaison_modeles.ipynb` avec Jupyter ou VS Code et exécuter toutes les cellules depuis la racine du projet ou depuis le dossier `notebooks`.

## Limites

L'échantillon est petit et provient de trois îles et des années 2007 à 2009. Une bonne métrique sur cette séparation aléatoire ne prouve pas une généralisation à d'autres lieux, périodes ou méthodes de mesure. Les catégories sont déséquilibrées ; c'est pourquoi le F1 macro et la balanced accuracy accompagnent l'accuracy. Le notebook montre les résultats observés, sans prétendre à une validation en production.
