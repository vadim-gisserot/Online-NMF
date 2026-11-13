# Projet d'apprentissage pour les séries temporelles

Étude du papier suivant : “COVID-19 Time Series Prediction by Joint Dictionary Learning and Online NMF” (https://arxiv.org/abs/2004.09112).

## Description du projet :
Le papeir décrit une nouvelle approche pour la moélisation et la prédiction de séries temporelles : dans le papier, les auteurs se concentrent sur les nouveaux cas, les nouveaux décès et les nouveaux rétablissements quotidiens de COVID-19, au début de l'année 2020. Ce repo contient le code associé au papier, dont nous entendons étudier les forces et les faiblesses.

La démarche ne décrit aucun modèle baseline auquel comparer les résultats de l'ONMF. C'est ce que nous proposons dans un premier temps, avec la RMSE :
1. Approche naïve
2. Prédiction par moyenne mobile
3. Modélisation ARIMA
4. Modélisation VAR

Puis nous chercherons à déterminer les méthodes les plus robustes pour la prévision de moyen terme grâce aux données ultérieures aujourdh'ui disponibles.

## File description
(From origianl repository)

  1. **ontf.py** : Online Nonnegative Tensor Factorization algorithm (generalization of onmf to the tensor setting by folding/unfolding operation)
  2. **time_series_ONMF_COVID19.py** : Main file implementing ONMF to COVID-19 time-series data  
  3. **main.py** : Tune hyperparameters and execute main files

## Usage
  1. git clone this repository
  2. git clone https://github.com/CSSEGISandData/COVID-19 inside ONMF-COVID19
  3. run main.py 
