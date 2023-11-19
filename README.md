# Bet prediction web interface
Flask application to replace TkInter GUI from ProphitBet application.

# League Statistics

For each league, the application computes several statistics (features) about the teams, including their form, the performance of the last N matches, etc. The stats are computed for both the home team and the away team.

🚧 Adding more parameters

# Leagues

Provides download of match history for some leagues from https://www.football-data.co.uk/.


# Analysis
## Feature Correlation Analysis

Heatmap for the computed stats, which shows the correlations
netween columns. The correlation is described by an arithmetic value ${r \in[-1.0, 1.0]}$. The closer $r$ is to zero, the weaker the correlation is between 2 columns. The closer to 1.0 or -1.0, the stronger the correlation will be. Ideally, a feature is good if the correlation with other features is close to zero ($r=0$).

## Feature Importance Analysis

There are 4 methods provided:

1. Ensemble Learning (https://www.knowledgehut.com/blog/data-science/bagging-and-random-forest-in-machine-learning)
2. Variance Analysis (https://corporatefinanceinstitute.com/resources/knowledge/accounting/variance-analysis/)
3. Univariate Analysis (https://link.springer.com/referenceworkentry/10.1007/978-94-007-0753-5_3110)
4. Recursive Feature Elimination (https://bookdown.org/max/FES/recursive-feature-elimination.html)

## Class Distribution Analysis


# Training
## Training Deep Neural Networks

A detailed description of neural networks can be found in the link below:
https://www.investopedia.com/terms/n/neuralnetwork.asp


## Training Random Forests

A detailed description of random forests can be found in the link below:
https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

🚧 Adding more models


# Evaluating Models

🚧 Ensure proper train-test-validation datasets

# Fixture Parsing
Auto parse footystats fixtures to predict whole "match days".

# Requirements
Python 3.11

# Instructions

Create env, install requirements.txt, run
```
python main.py
```
open shown url in browser