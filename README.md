# ProphitBet - Soccer Bets Predictor
ProphitBet is a Machine Learning Soccer Bet prediction application. It analyzes the form of teams, computes statistics from previous matches of a selected league and predicts the outcomes of a match using Machine Learning (ML) methods. The supported algorithms in this application are Neural Networks and Random Forests. Additionally, the users may analyze the features of the models and adjust the models accordingly. The model extracts soccer data for multiple leagues from *football-data*(https://www.football-data.co.uk/). Additionally, the application can parse upcoming fixtures from *Footystats*(https://footystats.org/) and predict the upcoming matches for a league. There is also an auto-save feature, which saves the training of the models, so that users can re-load them on the next run. Finally, the application requires **Internet Connection**, in order to download the league data.

# Simple Graphical Interface

The user interface is pretty simple: Every action can be done via a menu-bar on the top of the application.

![gui](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/stats.png)

# League Statistics

For each league, the application computes several statistics (features) about the teams, including their form, the performance of the last N matches, etc. The stats are computed for both the home team and the away team. More specifically:

1. **Home Wins (HW)**: Last N wins of the home team in its home
2. **Home Losses (HL)**: Last N losses of the home team in its home
3. **Home Goal Forward (HGF)**: Sum of goals that the home team scored in the last N matches in its home
4. **Home G-Goal Difference Wins (HGD-W)** Last N wins of the home team with G difference in the final score in its home (${HG - AG \geq 2}$)
5. **Home G-Goal Difference Losses (HGD-L)** Last N losses of the home team with G difference in the final score in its home (${HG - AG \geq 2}$)
6. **Home Win Rate (HW%)** Total win rate of the home team from the start of the league in its home
7. **Home Loss Rate (HL%)** Total loss rate of the home team from the start of the league in its home
8. **Away Wins (AW)**: Last N wins of the away team away its home
9. **Away Losses (AL)**: Last N losses of the away team away its home
10. **Away Goal Forward (AGF)**: Sum of goals that the away team scored in the last N matches away its home
11. **Away G-Goal Difference Wins (AGD-W)** Last N wins of the away team with G difference in the final score away its home(${HG - AG \geq 2}$)
12. **Away G-Goal Difference Losses (AGD-L)** Last N losses of the away team with G difference in the final score away its home (${HG - AG \geq 2}$)
13. **Away Win Rate (AW%)** Total win rate from the start of the league away its home
14. **Away Loss Rate (AL%)** Total loss rate from the start of the league away its home

# Leagues

# Feature Correlation Analysis

# Feature Importance Analysis

# Class Distribution Analysis

# Training Deep Neural Networks

# Training Random Forests

# Evaluating Models

# Outcome Predictions

# Fixture Parsing

# Requirements

| Library/Module  | Download Url |
| ------------- | ------------- |
| Python Language | https://www.python.org/ |
| Numpy  | https://numpy.org/ |
| Pandas  | https://pandas.pydata.org/ |
| Matplotlib  | https://matplotlib.org/ |
| Seaborn  | https://seaborn.pydata.org/ |
| Scikit-Learn  | https://scikit-learn.org/stable/ |
| XGBoost  | https://xgboost.readthedocs.io/en/stable/ |
| Tensorflow  | https://www.tensorflow.org/ |
| Tensorflow-Addons  | https://www.tensorflow.org/addons |
| TKinter  | https://docs.python.org/3/library/tkinter.html |

# Instructions (How to Run)


