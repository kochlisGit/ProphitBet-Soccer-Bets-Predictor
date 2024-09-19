# ProphitBet - Soccer Bets Predictor
ProphitBet is an **Open Source** Machine Learning (ML) Soccer Bet prediction application. The name comes from a combination of "Profit" & "Prophet". With profitbet, You can analyze the form of teams using advanced machine learning methods and stunning visualizations techniques, compute several statistics from previous matches of a selected league and predict the outcomes of a matches. The supported algorithms in this application are Deep Neural Networks, Random Forests, XG-Boost, KNN, SVM, Decision Tree, Logistic Regression, Naive Bayes & Ensemble models. Several feature prrprocessing method are also included, such as Data Normalization and Imbalanced-Learning techniques, which further boost the performance of ML models. The app extracts soccer data for **every** league included in *football-data*(https://www.football-data.co.uk/). Additionally, the application can parse upcoming fixtures from *Footystats*(https://footystats.org/) and predict the upcoming matches of a league. Moreover, advanced validation techniques, such as Cross-Validation and Holdout are automatically employed during the model's training, to ensure smaller chances of the trained models being over-fitted. Finally, the application requires **Internet Connection**, in order to download the league data.

# Stunning Graphical Interface

The user interface is pretty simple: Every action can be done via a menu-bar on the top of the application. There are 5 available menus:

* Application: Create/Load/Delete Leagues
* Analysis: Data Analysis & Feature Importance
* Model: Train/Evaluate Models & Predict Matches
* Theme: Select a Theme for the Application Window
* Help: Additional Resources to Read about Machine Learning Topics

Also, 4 custom themes have been added and can be selected via "Theme" menu. The themes are:

1. Breeze-Light
1. Breeze-Dark
1. Forest-Light
1. Forest-Dark

![gui](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/create_league.png)

![gui](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/loaded_league.png)

# League Statistics

For each league, the application computes several statistics (features) about the teams, including their form, the performance of the last N matches, etc. The stats are computed for both the home team and the away team. More specifically:

1. **Home Wins (HW)**: Last N wins of the home team in its home
2. **Home Losses (HL)**: Last N losses of the home team in its home
3. **Home Goal Forward (HGF)**: Sum of goals that the home team scored in the last N matches in its home
4. **Home Goal Against (HGA)**: Sum of goals that the away teams scored in the last N matches.
5. **Home G-Goal Difference Wins (HGD-W)** Last N wins of the home team with G difference in the final score in its home (${HG - AG \geq 2}$)
6. **Home G-Goal Difference Losses (HGD-L)** Last N losses of the home team with G difference in the final score in its home (${HG - AG \geq 2}$)
7. **Home Win Rate (HW%)** Total win rate of the home team from the start of the league in its home
8. **Home Loss Rate (HL%)** Total loss rate of the home team from the start of the league in its home
9. **Away Wins (AW)**: Last N wins of the away team away its home
10. **Away Losses (AL)**: Last N losses of the away team away its home
11. **Away Goal Forward (AGF)**: Sum of goals that the away team scored in the last N matches away its home
12. **Away Goal Against (AGA)**: Sum of goals that the home teams scored in the last N matches.
13. **Away G-Goal Difference Wins (AGD-W)** Last N wins of the away team with G difference in the final score away its home(${HG - AG \geq 2}$)
14. **Away G-Goal Difference Losses (AGD-L)** Last N losses of the away team with G difference in the final score away its home (${HG - AG \geq 2}$)
15. **Away Win Rate (AW%)** Total win rate from the start of the league away its home
16. **Away Loss Rate (AL%)** Total loss rate from the start of the league away its home

Each column can be added or removed from a league during the creating phase. 

# Leagues

ProphitBet provides 11 main soccer leagues and 2 extras, which are downloaded by https://www.football-data.co.uk/. More specifically, these leagues are:
* 'Argentina': [PrimeraDivision]
* 'Belgium': [JupilerLeague]
* 'Brazil': [BrazilSerieA]
* 'China': [ChinaSuperLeague]
* 'Denmark': [SuperLiga]
* 'England': [PremierLeague, Championshio, League1, League2]
* 'Finland': [VeikkausLiiga]
* 'France': [Ligue1, Ligue2]
* 'Germany': [Bundesliga1, Bundesliga2]
* 'Greece': [SuperLeague]
* 'Ireland': [IrelandPremierDivision]
* 'Italy': [SerieA, SerieB]
* 'Japan': [J1]
* 'Mexico': [LigaMX]
* 'Netherlands': [Eredivisie]
* 'Norgway': [Eliteserien]
* 'Poland': [Ekstraklasa]
* 'Portugal': [Liga1]
* 'Romania': [RomaniaLiga1]
* 'Russia': [RussiaPremierLeague]
* 'Scotland': [Premiership]
* 'Spain': [LaLiga, SegundaDivision]
* 'Sweden': [Allsvenskan]
* 'Switzerland': [SwitzerlandSuperLeague]
* 'USA': [MLS]
* 'Turkey': [SuperLig]


You can add additional leagues by modifying the `database/leagues.csv` configuration file. In order to add a new league, you need to specify:
1. Country (The country of the league, e.g. Russia)
2. League Name (The name of the league e.g. Premier League)
3. League ID: You can create multiple leagues, but with different ID.
4. The statistical odds that will be used to train the models.

# Feature Correlation Analysis

This is particulary useful, when analyzing the quality of the training data. ProphitBet provides a headmap for the correlation matrix between the features, which shows the correlations 
between 2 features (columns). The correlation is described by an arithmetic value ${r \in[-1.0, 1.0]}$. The closer $r$ is to zero, the weaker the correlation is between 2 columns. The closer to 1.0 or -1.0, the stronger the correlation will be. Ideally, a feature is good if its correlation with the rest of the features is close to zero ($r=0$).

![correlation heatmap analysis](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/correlations.png)

# Feature Importance Analysis

ProphitBet also comes with a built-in module for "**interpretability**". In case you are wondering which stats are the most important, there are 3 methods included:

2. Variance Analysis (https://corporatefinanceinstitute.com/resources/knowledge/accounting/variance-analysis/)
3. Recursive Feature Elimination (https://bookdown.org/max/FES/recursive-feature-elimination.html)
4. Random Forest importance scores

![feature-importance-analysis](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/importance.png)

# Class (Target) Distribution Analysis

It is noticed that the training dataset of several leagues contains imbalanced classes, which means that the number of matches that ended in a win for the home team 
is a lot larger than the number of the matches that ended in a win for the away team. This often leads models to overestimate their prediction probabilities and tend to have a bias towards the home team. ProphitBet provides a plot to detect such leagues, using the **Target Distrubution Plot**, as well as several tools to deal with 
that, including:

1. Noise Injection (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2771718/)
2. Output Probability Calibration (https://davidrosenberg.github.io/ttml2021/calibration/2.calibration.pdf)
3. Resampling techniques (SMOTE, SMOTE-NN, SVM-SMOTE, NearMiss, Radnom Resampling)

![class distribution](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/targets.png)

# Training Deep Neural Networks

A detailed description of neural networks can be found in the link below:
https://www.investopedia.com/terms/n/neuralnetwork.asp

![deep neural networks](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/neuralnetwork.png)

# Machihe Learning Models

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Naive Bayes
4. Decision Tree
5. Random Forest
6. XG-Boost
7. Support Vector Machine (SVM)
8. Deep Neural Networks

# Training Random Forests

A detailed description of random forests can be found in the link below:
https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

![random forests](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/randomforest.png)

# The Ensemble Model

This type of combines the predictions of several machine learning models. Typically, a well tuned Random Forest could generate similar predictions with a Neural Network or any other ML model. However, there are some cases where 2 models could output different output probabilities (e.g. Random Forest might give higher probability that an outcome is Home). In that case, the ensemble model (Voting Model) can be used, which averages the output probabilities of several models and decides on the predicted outcome. The idea is that each model makes unique predictions, so their predictions are combined to form a stronger model.

# Evaluating Models

Before using a trained model, it is wise to first evaluate the model on unseen matches. This should reveal the quality of the model training, as well as its output probabilities. You can compare the probabilities of random forest with the neural network's probabilities and choose the most confident and well-trained model. Additionally, you can request an analytical report of the accuracy of the classifiers for specific odd intervals (e.g. the accuracy between 1.0 and 1.3, 1.3, and 1.6, etc., for the home or away team).

![model evaluation](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/evaluate.png)

# Outcome Predictions

In order to request a prediction for a match, You need to select the home/away team, as well as the book odds. You should use both models to make a prediction. If both models agree, then the prediction should probably be good. If the models disagree, then it's best to avoid betting on that match. The outcome prediction includes:

1. Home, Draw or Away
2. Under (2.5) or Over (2.5)

![match predictions](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/predictions.png)

# Fixture Parsing

An alternative way to predict multiple matches at once is to use the "**Fixture Parsing**" option. You may now automatically parse the fixtures using your browser. Once the fixture window pops-up, select your **browser and the fixture date** and the application will automatically download the page & parse the upcoming fixtures of the specified data. This is a new feature, so please report any bugs in the issues page.
![fixture parsing & upcoming match prediction](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/fixtures.png)

# Requirements & Installation

Below are the steps of installing this application to your machine. First, download this code and extract it into a directory. Then, follow the steps below:

1. Download & Install python. During the installation, you should choose  **add to "Path"**. It is recommended to download **python 3.9.** or higher version.
2. After you download & install python, you can Download the above libraries using pip module (e.g. `pip install numpy==VERSION`). The version can be found in *requirements.txt* file. These modules can be installed via the cmd (in windows) or terminal (in linux). **IMPORTANT**: To download the correct versions, just add "==" after pip install to specify version, as described on requirements.txt file. For example, to install `tensorlfow 2.9.1`, you can use: `pip install tensorflow==2.9.1`.
3. On windows, you can double click the main.py file. Alternatively (Both Windows & Linux), You can open the cmd on the project directory and run: `python main.py`. 

**A `requirements.txt` file has been added to the project directory. The table below presents the required libraries, however, you should check the `requirements.txt` file for the required library versions.**

| Library/Module  | Download Url | Installation |
| ------------- | ------------- | -------------
| Python Language | https://www.python.org/ | Download from website |
| Numpy  | https://numpy.org/ | `pip install numpy` |
| Pandas  | https://pandas.pydata.org/ | `pip install pandas` |
| Matplotlib  | https://matplotlib.org/ | `pip install matplotlib` |
| Seaborn  | https://seaborn.pydata.org/ | `pip install seaborn` |
| Scikit-Learn  | https://scikit-learn.org/stable/ | `pip install scikit-learn` |
| Imbalanced-Learn  | https://imbalanced-learn.org/stable/ | `pip install imbalanced-learn` |
| XGBoost  | https://xgboost.readthedocs.io/en/stable/ | `pip install xgboost` |
| Tensorflow  | https://www.tensorflow.org/ | `pip install tensorflow` |
| Tensorflow-Addons  | https://www.tensorflow.org/addons | `pip install tensorflow_addons` |
| TKinter  | https://docs.python.org/3/library/tkinter.html | `pip install tk ` |
| Optuna | https://optuna.org/ | `pip install optuna` |
| Fuzzy-Wuzzy | https://pypi.org/project/py-stringmatching | `pip install fuzzywuzzy` |
| Python-Levenshtein | https://pypi.org/project/python-Levenshtein/ | `pip install python-Levenshtein` |
| Tabulate | https://pypi.org/project/tabulate/ | `pip install tabulate` |
| Selenium | https://pypi.org/project/selenium/ | `pip install selenium` |
| LXML | https://pypi.org/project/lxml/ | `pip install lxml` |

To run `pip` commands, open CMD (windows) using Window Key + R or by typing cmd on the search. In linux, You can use the linux terminal. You can also install multiple libraries at once (e.g. `pip install numpy==1.22.4 pandas==1.4.3 ...`

# Common Errors
1. `Cannot install tensorflow.` Sometimes, it requires visual studio to be installed. Download the community edition which is free here:  [https://pypi.org/project/py-stringmatching](https://visualstudio.microsoft.com/downloads/)
2. `pip command was not found` in terminal. In this case, you forgot to choose **add to Path** option during the the installation of python. Delete python and repeat download instructions 1-3.
3. `File main.py was not found`. This is because when you open command line (cmd) tool on windows, or terminal on linux, the default directory that cmd is looking at is the home directory, not prophitbet directory. You need to navigate to prophitbet directory, where the main.py file exists. To do that, you can use the `cd` command. e.g. if prophitbit is downloaded on "Downloads" folder, then type `cd Downloads/ProphitBet-Soccer-Bets-Predictor` and then type `python main.py`
4. `python command not found` on linux. This is because python command is `python3` on linux systems
5. `Parsing date is wrong` when trying to parse fixtures from the html file. The html file has many fixtures. Each fixture has a date. You need to specify the correct date of the fixture you are requesting, so the parser identifies the fixtures from the given date and grab the matches. You need to specify the date before importing the fixture file into program.
6. `<<library>> module was not found` This means that a library has been installed, but it is not included in the documentation or requirements.txt file. Try to install it via `pip` command or open an issue so that i can update the documentation.

# Supported Platforms
1. Windows
2. Linux
3. Mac

# Open An Issue
In case there is an error with the application, open a Github Issue so that I can get informed and (resolve the issue if required).

# Known Issues

1. **Neural Network's Training Dialog Height is too large and as a result, "Train" button cannot be displayed.**

Solution: You can press "ENTER" button to start training. The same applies to Random Forest Training Dialog, as well as the tuning dialogs.

# Release (2024/09/18)
* Updated statistics: The average odds (1,X,2) are not replaced with "close" average odds, which are the average odds at the time that a match starts.
* Updated documentation: installation instructions and common errors.
* Simplified filters and fixed a bug in percentiles.
* Deleted deprecated directory "network".
* Fixed a bug where no normalization was applied after loading a pre-trained model.
* Fixed a bug where several leagues could not be loaded. This is because football-data website changed the column names".

# Release (2024/01/29)

* Fixed Download/Update bugs. All leagues should now be properly downloaded and updated.
* Added several url links in the help menu about the Machine Learning methods.
* Added every league from football-data website. There is no more need to manually add a new league.
* Added 2 Prediction tasks: H/D/A and U/0 (2.5).
* Added 9 Machine Learning models, including KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, XG-Boost, Deep Neural Networks, SVM, Voting Model.
* Several parameters are now provided for each model during the training process.
* Cross-Validation is now employed during training, which enchances the model's reliability.
* Tuning process is now more simple. You can automatically select which parameters you wish to search and manually select the rest of them.
* Fixed fixture parser bugs.
* Fixtures can now be automatically parsed from Footystats by selecting a browser and the fixture date.
* Updated documentation and menus.

# Release (2023/01/19)

* Improved Graphical User Interface (GUI)
* Added Custom Themes
* Added "Ensemble" Model
* Training can now start by pressing "ENTER" button
* Added option for SVM-Smote resampling method (https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SVMSMOTE.html). It requires imbalanced-learn to be installed
* Replaced `py_stringmatching` library, which was a bit confusing to install, with `fuzzywuzzy` (Check requirements.txt) 
* Fixtures are now imported, even if odds are missing. You can also manually add them or edit them
* Fixed Bugs (Leagues not updating, Fixtures not being imported, etc.)
* Added `Weighting` method to Random Forest.
* Neural Networks may now have different activation, regularization or batch normalization option on each layer separately. 
* Added more metrics (F1-Score, Precision, Recall)
* Tuning may now focus on maximizing F1-Score, Precision and Recall of a specified target (Home, Draw or Away).
* Updated Documentation!

# Release (2022/08/30)

* Fixed a bug in Evaluation Filters
* Fixed Fixture Parser
* Added 2 new statistic features (columns): HGA, AGA
* Neural Network now supports different noise ranges for each odd (1/x/2)
* Neural Network may now add noise only to favorite teams (teams with odd < 2.0)

# Release (2022/09/19)

* Fixed a bug where several leagues would not be updated
* Fixed a bug in evaluation filters

# Release (2022/11/05)

* Improved Model's Training
* Added more training parameters, including, Dropout layers, Batch Normalization, Optimizers, Learning Rate, Regularizers
* Model may now achieve higher accuracies
* Added option to automatically search for best parameters, using OPTUNA package (Requires the installation of optuna, see instructions)
* Updated Documentation: Added more detailed instruction + common errors and how they are dealt

![Training Parameters](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/parameters.png)

# Release (2022/11/12)

* Fixed a bug where leagues wouldn't be updated up to last day

# Release (2023/02/19)

* Smaller windows sizes
* Better parameter selection for neural network tuning
* Train Dialogs may now initiate training by hitting "ENTER" button
* Small bug fixes

# Release (2023/04/01)

* Fixed a bug where model could not be saved during training
* Fixed a bug where validation accuracy was not properly monitored during tuning
* Increased number of available Trials to 2000
* Added more options, including layers of neural network during training
* Updated documentation

# Contribution

If you liked the app and would like to contribute, You are allowed to make changes to the code and make a pull request! Usually, it takes 1-3 days for me to
review the changes and accept them or reply to you if there is something wrong.

# Donation

A lot of people request more football training data (e.g. corners, shots, cards, injuries, etc.). I found a football API that does provide such data https://footystats.org/api/ . However, such data are not available for free. I would like your help to improve the quality of the training data that the algorithms use. Higher quality data should increase the performance of the algorithms and get better predictions in every league. Addiotioanlly, more options could be supported (e.g. Under 2.5 or Over 2.5, Num Corners, etc.). I made it available for everyone who liked to app and would like to contribute to donate any amount. The money will be spent on monthly subscription on footystats API, where I will be able to download a whole more data. 

If you liked the app, earned some prophits (profits) and want to contribute to more advanced betting-helper app, you can donate using the following link:

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=AK3SEFDGVAWFE)

or via QR-Code:

![Donation](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/QR%20Code.png)

# Currently Donated Money

80€ **(Next Goal: 100€)**

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
