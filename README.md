# New Version (Release 13-11-2025)

The old version of ProphitBet is deprecated and has been moved to branch `ProphitBet-v1`. The new version, ProphitBet-v2 has now been released! The new version features:

* Easier/Faster Installation.
* New, Prettier User Interface (UI).
* Better Documentation.
* Cleaner & Faster Code.
* More Statistics.
* More Analysis Tools.
* More Training/Evaluation Tools.
* More Training Options
* Explainable Models.

# ProphitBet-v2

ProphitBet (Prophet/Profit + Bet) is an **Open Source** Machine Learning (ML) Soccer Bet prediction application, which allows you to download historical soccer data, analyze the form of teams using advanced ML methods, compute several team statistics, create statistical graph visualizations, and predict the outcomes of a matches. This app extracts soccer data (requires Internet Connection) for **every** league included in *football-data*(https://www.football-data.co.uk/). Additionally, it can parse upcoming fixtures from *Footystats*(https://footystats.org/) and predict the upcoming matches of a league, which can be organized and saved into an Excel file. Finally, it utilizes advanced model training & validation techniques, such as Cross-Validation and Holdout, which are automatically employed during the model's training, to ensure robust training of the models with low probability of generating over-fitted models. 

ML is a sub-field of Artificial Intelligence (AI), which allows users to construct models that automatically extract patterns between multiple variables and predict the outcome of an event. You can learn more about ML methods here: 

* https://www.ibm.com/think/topics/machine-learning
* https://www.geeksforgeeks.org/machine-learning/machine-learning/

# Improved Graphical User Interface (GUI)

The new GUI is more lightweight and prettier. It comes with two themes:

1. Light
2. Dark

Additionally, it supports several key shortcuts to create, open and delete leagues to enable faster workflow.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/gui.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/gui.png)

# Creating Leagues

To create a league, press `CTRL+N` or navigate to `File/New League`. Then select a league and provide a unique identifier (id). The id is used to save the league and restore it when you re-open the application. Then, you can choose a variety of statistics that you would like to import, along with the historical data range and create the league.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/newleague.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/newleague.png)

You created leagues are automatically stored in your computer and you can restore them anytime using the open window (or by pressign `CTRL+O`). 

# Excel-Like Table

The new match table looks like the excel table and includes several advanced functions, such as:

1. Hide Matches (hides matches with missing values that are ommited during training).
2. Search/Find (searches and selects all rows that contain a specific keyword).
3. Copy (copies matches to clipboard using a table-like format. The copied matches can be pasted directly into excel as well).

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/table.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/table.png)

## Tips

* Select a recent historical change (e.g. 2015 or after). Although old seasons can be downloaded, they do not often yield the best training results.
* Use the filters to select a desired odd range.
* Use the analysis tools to eliminate bad statistical features (variables).

# Extended Data Analysis

The app includes a variety of data analysis tools that you can work with. Use them mainly to analyze the collected matches, filter noisy statistical features and visualize the quality of each variable.

## Descriptive Statistics

Calculate basic statistics for each variable, such as mean, variance, standard deviation, median, min and vax values.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_descriptive.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_descriptive.png)

## Distributions

Visualize the distribution of each variable to extract useful information about the quality of each variable. For instance:

* Visualize the target distribution to detect imbalanced target classes (Imbalanced classes can lead to extreme bias of the model towards specific classes. In such cases, you can employ data sampling techniques and class weights.)
* Normal distributions are usually desirable.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_destribution.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_destribution.png)

## Variance

Visualize the variance of each variable. Typically, low-variance variables contain little to no information and can be eliminated.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_variance.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_variance.png)

## Correlation

Use the correlation analysis tool as follows:

1. Detect variables that are highly correlated to the target variables (such variables help the models to make better predictions).
2. Detect correlations between two variables. If your goal is to create explainable/interpretable models, then only one of the two variables should be kept. 

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_variance.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_variance.png)

## Boruta Algorithm (New)

The boruta algorithm is a feature (variable) selection algorithm. It trains multiple times (iterations) a Random Forest to predict the outcome of the matches, but at each iteration, it replaces a feature with random values and examines the performance of the model. Finally, it ranks each feature based on the Random Forest's performance. 

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_boruta.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_boruta.png)

## Logistic Regression Coefficients

This tool trains a logistic regression model to predict the outcomes of the matches and analyzes its coefficients. The coefficients can display how much a statistical variable affects the outcome of a match. Be aware that the logistic regression assumes linear relationship between the target variables and the input.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_coefficients.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_coefficients.png)

## Impurity Analyzer

Impurity is a statistical metric of tree-based models. This tools trains a decision tree and displays the impurity metric for each variable. It is important to note that a decision tree does not assume linear relationship between inputs and outputs, so the importances of each variable could be different than the logistic regression ones.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_impurity.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_impurity.png)

## Rules Extractor (New)

You can now extract rules by training and explaining a decision tree. Specifically, the decision tree forms "IF-ELSE" rules.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_rules.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/analysis_rules.png)

# Training Models

The new training window allows you to train/tune a model more easily. Also, it supports both sliding cross-validation and k-fold cross validation methods to evaluate the trained models. Below is a list of the supported models:

1. 

## Tips:

* If you are not sure about the model's parameters, you can choose to tune those parameters instead.
* Tuning is typically a slow procedure, but usually finds the best possible parameters.
* Use the sliding cross-validation and k-fold cross validation method to evaluate your models. This might increase the training time, but produces more reliable models.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/training.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/training.png)

# Contribution

If you liked the app and would like to contribute, You are allowed to make changes to the code and make a pull request! Usually, it takes 1-3 days for me to
review the changes and accept them or reply to you if there is something wrong.

# Donation

If you liked the app, you can buy me a coffee via Paypal:

![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=AK3SEFDGVAWFE)

or you can scan the QR-Code below:

![Donation](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/ProphitBet-v1/screenshots/QR%20Code.png)
# Citation

If you are writing an academic paper, please cite us!

```
@software{prophitBet2024,
  author = {Vasileos Kochliaridis},
  orcid = {0000-0001-9431-6679},
  month = {1},
  title = {{ProphitBet - An Open Source Soccer Prediction App}},
  url = {https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor},
  version = {2.0.0},
  year = {2024}
}
```
