# New Version (Release 13-11-2025)

The old version of ProphitBet is deprecated and has been moved to branch `ProphitBet-v1` (You can access it by selecting it from top-left combobox). The new version, ProphitBet-v2 has now been released! The new version features:

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

# Installation (New)

## Step 1 - Python Installation

If you are a new user to Python, it is recommended that you previously uninstall everything **(including previous Python versions)!**

The first step to install ProphitBet-v2 is to install **Python 3.11**. Although other Python versions can also be used (Python 3.10, 3.12, 3.13, 3.14, etc.), it has only been tested in 3.10 and 3.11. Download it from here: https://www.python.org/downloads/release/python-3119/

During installation, make sure the *"Add Python to Path"* is selected. If you are not sure how to install it, watch this tutorial here: (https://www.youtube.com/watch?v=yivyNCtVVDk)

[![YouTube Video](https://img.youtube.com/vi/yivyNCtVVDk/0.jpg)](https://www.youtube.com/watch?v=yivyNCtVVDk)

## Step 2 - Download Code

Download the code from the github repository other by clicking on the green "Code" button above selecting Download zip or by copying this address: https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/archive/refs/heads/main.zip
After the download, unzip the files into your preferred directory (e.g. *Downloads/ProphitBet*)

## Step 3 - Install Python Libraries

ProphitBet requires several Python libraries **(with specific versions)** to run, which are analytically described in `requirements.txt` file. If you are a new Python user, skip ignore the following instruction. If you are an advanced Python user, you can use this file to manually install the requirements in whichever environment you like or by typing `pip install -r requirements.txt`.

If you are a new user, you can use the `install.py` script to automatically download the required libraries. Open the Command Line (CMD) (or Trminal in Linux). In windows, you can open the cmd by typing *cmd* or *Command Prompt* in the windows search bar or by pressing the keys *Win+R* and typing *cmd* there. Then, nagivate on the created folder (e.g., *cd Downloads/ProphitBet*. Finally, type: `python install.py` and press ENTER to initiate the installation. These libraries will be automatically installed to the default Python version. If you are an advanced user and would like to install the libraries in a specific envrionment, you can also use *python install.py --venv "C:\Users\You\python\envs\myenv"*. If everything runs perfectly, you will notice an *Installation complete!* message and the installation details below, otherwise it will display an error.

## Step 4 - Run Application

To open ProphitBet, you can navigate the CMD to the folder and type `python app.py`. You can also run the application by double-clicking the *app.bat* file (In linux, you can use the *app.sh* respectively). If you have an anti-virus enabled, it is possible that it slows down the initialization of the app.

# Installation Help

If you would like an additional installation help or an error occurs, please open a github issue, so that me or any user can further assist you.

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

1. **Logistic Regression**: Linear model taht assumes linear relationship between inputs $x$ and targets $y$.
2. **Discriminant Analysis Classifier (LDA/QDA) (New)**: Encodes the inputs $x$ into a 2D space and then computes targets in a linear/quadratic manner.
3. **Decision Tree**: Tree classifier that forms "if-else" rules.
4. **Random Forest**: Multiple decision trees trained in different subsets of data. The decisions of all trees are then averaged to calculate $y$.
5. **Extreme Boosting (XGBoost)**: Classifier that trains consecutive decision trees that rectify the mistakes of the previous trees.
6. **K-Nearest Neighbors (KNN)**: Laze classifier that computes y based on the $k$ closest (most identical) samples (matches).
7. **Naive Bayes**: Statistical learning algorithm that uses Bayes rules to compute the probability of each target.
8. **Support Vector Machine (SVM)**: Advanced statistical learning algorithm that utilizes support vectors to separate targets in the best possible manner.
9. **Deep Neural Network (DNN) (New)**: Current State-Of-The-Art algorithm (used by Chat-GPT). Added support for *Attention* mechanism and *Variable Selection (VNS)*.

**Two training modes are currently available**:

1. Result (1/X/2)
2. (U/O) 2.5

Other result types can be added in the future.

## Tips:

* Tuning is typically a slow procedure, but usually finds the best possible parameters.
* You don't have to tune all parameters, but if you are not sure about specific model's parameters, you can choose to tune them instead.
* Use the sliding cross-validation and k-fold cross validation methods to evaluate your models. This might increase the training time, but produces more reliable models.
* When the targets are imbalanced, use data-sampling techniques and class weight option (if available)!
* The standard scaling algorithm usually works well with most of the algorithms.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/training.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/training.png)

# Evaluating Models

The main purpose of model evaluation is to understand the strengths and weaknesses of your models (to understand in which cases/matches your model performs best). To achieve this, several tools are provided:

1. Dataset Evaluation: Εvaluate how your model fits in the train (known/seen) data, as well as how your model performs in (unknown/unseen) data that are used for evaluation purposes. The correct matches are automatically highlighted.
2. Range filters: Filter the performance of the models based on the specified odd range (.e.g., odd 1 from 1.0 to 1.30).
3. Percentiles: Filter the performance of the models based on their output probabilities (e.g., $prob(1) > 0.3, prob(X) > 0.5, prob(2) > 0.4$, etc.).

You also have the option to store the filters and utilize them during the prediction of the fixtures. Keep in mind that the filters are calculated on the specified dataset separately (All/Train/Eval).

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/evaluate.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/evaluate.png)

# Predict Offline

You can predict any possible combination of matches you can think of (without requiring internet connection). Simply specify Home, Away teams and their odds, and select a model to predict the outcome.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/predict.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/predict.png)

# Predict Fixtures

This mode enables you to predict the outcomes of the upcoming league fixture (**requires internet connection and the Google Chrome installed**). You just have to specify the fixture date and the parsing will start automatically. 
Then, you can select a model and generate the predictions. By default, all predictions are selected (highlighted) for export. However, you can also specify the evaluation filters to filter specific odd ranges and probabilities.

![https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/fixtures.png](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/fixtures.png)

# Contribution

If you liked the app and would like to contribute, You are allowed to make changes to the code and make a pull request! Usually, it takes 1-3 days for me to
review the changes and accept them or reply to you if there is something wrong. Keep in mind that i am only a single person working on my free time on this project, so please be patient!

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
