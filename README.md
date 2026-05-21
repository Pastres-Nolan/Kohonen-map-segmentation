# Cartes de Kohonen (Self-Organizing Maps) – Segmentation Client

## Présentation

Ce projet implémente une **carte de Kohonen** (*Self-Organizing Map / SOM*) en Python afin de réaliser une segmentation de clients à partir de données comportementales.

Les cartes de Kohonen sont des réseaux de neurones non supervisés capables de :
- détecter des structures cachées dans des données,
- regrouper automatiquement des profils similaires,
- projeter des données complexes dans un espace plus simple à visualiser.

Dans ce projet, chaque client est décrit par plusieurs caractéristiques :
- âge,
- revenu annuel,
- score de dépense,
- fréquence d’achat,
- panier moyen.

Ces données sont donc représentées dans un espace **5 dimensions (5D)**.

---

## Pourquoi passer de 5D à 2D ?

Les données réelles possèdent souvent beaucoup de dimensions et sont difficiles à visualiser directement.

La carte de Kohonen permet de :
- conserver les similarités entre les clients,
- rapprocher les profils similaires,
- organiser automatiquement les groupes de données.

Pour rendre le résultat compréhensible visuellement, les neurones du SOM sont projetés en **2D** grâce à une **ACP (PCA – Principal Component Analysis)**.

Cette réduction de dimension permet :
- d’observer l’organisation des clusters,
- de visualiser l’apprentissage du réseau,
- de suivre l’évolution des neurones pendant l’entraînement.

---

## Fonctionnement du projet

Le programme :
1. charge les données clients,
2. normalise les variables,
3. entraîne une grille SOM 10x10,
4. recherche le neurone le plus proche (*BMU – Best Matching Unit*),
5. met à jour les neurones voisins avec une fonction gaussienne,
6. projette les neurones en 2D avec PCA,
7. génère un GIF montrant l’évolution de l’apprentissage.

---

## Technologies utilisées

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- ImageIO
- Pillow


