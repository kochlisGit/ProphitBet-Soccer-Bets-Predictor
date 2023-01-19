# ProphitBet - Soccer Bets Predictor
ProphitBet is a Machine Learning Soccer Bet prediction application. The name is a combination of "Profit" & "Prophet". It analyzes the form of teams with stunning visualizations, computes statistics from previous matches of a selected league and predicts the outcomes of a match using Advanced Machine Learning (ML) methods. The supported algorithms in this application are Neural Networks, Random Forests & Ensemble models. Additionally, the users may analyze the features of the models and adjust the models accordingly. The model extracts soccer data for multiple leagues from *football-data*(https://www.football-data.co.uk/). Additionally, the application can parse upcoming fixtures from *Footystats*(https://footystats.org/) and predict the upcoming matches for a league. There is also an auto-save feature, which saves the training of the models, so that users can re-load them on the next run. Finally, the application requires **Internet Connection**, in order to download the league data.

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
* Premier League (England)
* Premiership (Scotland)
* Bundesliga I (Germany)
* Serie A (Italy)
* La Liga (Spain)
* Ligue I (Franch)
* Eredivisie (Netherlands)
* Jupiler Pro League (Belgium)
* Liga I (Portugal)
* Super Lig (Turkey)
* Super League (Greece)
* Serie A (Brazil)
* Allsvenskan (Sweden)

You can add additional leagues by modifying the `database/leagues.csv` configuration file. In order to add a new league, you need to specify:
1. Country (The country of the league, e.g. Russia)
2. League Name (The name of the league e.g. Premier League)
3. Base Url (The link to the .csv file from *football-data*, e.g. https://www.football-data.co.uk/new/RUS.csv)
4. Year Start (The year that ProphitBet will stop collecting data, e.g. 2015)
5. League Type (Since it's an extra league, it always has to be "`extra`")
6. Fixtures Url (The fixture's url from *footystats, which will be used to parse upcoming matches*, e.g. https://footystats.org/russia/russian-premier-league)

# Feature Correlation Analysis

This is particulary useful, when analyzing the quality of the training data). ProphitBet provides a headmap for the computed stats, which shows the correlations 
between the columns. The correlation is described by an arithmetic value ${r \in[-1.0, 1.0]}$. The closer $r$ is to zero, the weaker the correlation is between 2 columns. The closer to 1.0 or -1.0, the stronger the correlation will be. Ideally, a feature is good if the correlation with other features is close to zero ($r=0$).

![correlation heatmap analysis](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/correlations.png)

# Feature Importance Analysis

ProphitBet also comes with a built-in module for "**interpretability**". In case you are wondering which stats are the most important, there are 4 methods provided:

1. Ensemble Learning (https://www.knowledgehut.com/blog/data-science/bagging-and-random-forest-in-machine-learning)
2. Variance Analysis (https://corporatefinanceinstitute.com/resources/knowledge/accounting/variance-analysis/)
3. Univariate Analysis (https://link.springer.com/referenceworkentry/10.1007/978-94-007-0753-5_3110)
4. Recursive Feature Elimination (https://bookdown.org/max/FES/recursive-feature-elimination.html)

![feature-importance-analysis](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/importance.png)

# Class Distribution Analysis

It is noticed that the training dataset of several leagues contains imbalanced classes, which means that the number of matches that ended in a win for the home team 
is a lot larger than the number of the matches that ended in a win for the away team. This often leads models to overestimate their prediction probabilities and tend to have a bias towards the home team. ProphitBet provides a plot to detect such leagues, using the **Target Distrubution Plot**, as well as several tools to deal with 
that, including:

1. Noise Injection (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2771718/)
2. Output Probability Calibration (https://davidrosenberg.github.io/ttml2021/calibration/2.calibration.pdf)

![class distribution](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/targets.png)

# Training Deep Neural Networks

A detailed description of neural networks can be found in the link below:
https://www.investopedia.com/terms/n/neuralnetwork.asp

![deep neural networks](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/neuralnetwork.png)

# Training Random Forests

A detailed description of random forests can be found in the link below:
https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

![random forests](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/randomforest.png)

# The Ensemble Model

This type of combines the predictions of a Neural Network & Random Forest. Typically, a well tuned Random Forest makes similar predictions with a Neural Network. However, there are some cases where these 2 model output different output probabilities (e.g. Random Forest might give higher probability that an outcome is Home). In that case, the ensemble model can be used which averages the output probabilities of both models and decides on the predicted outcome.

# Evaluating Models

Before using a trained model, it is wise to first evaluate the model on unseen matches. This should reveal the quality of the model training, as well as its output probabilities. You can compare the probabilities of random forest with the neural network's probabilities and choose the most confident and well-trained model. Additionally, you can request an analytical report of the accuracy of the classifiers for specific odd intervals (e.g. the accuracy between 1.0 and 1.3, 1.3, and 1.6, etc., for the home or away team).

![model evaluation](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/evaluate.png)

# Outcome Predictions

In order to request a prediction for a match, You need to select the home/away team, as well as the book odds. You should use both models to make a prediction. If both models agree, then the prediction should probably be good. If the models disagree, then it's best to avoid betting on that match.

![match predictions](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/predictions.png)

# Fixture Parsing

An alternative way to predict multiple matches at once is to use the "**Fixture Parsing**" option. When you click on that option, it will open the browser and ask you
to download the specified fixture from *footystats.org*. This can be done by pressing *Ctrl + S* or right click and "Save As" option. Then, You will need to specify the filepath of the downloaded fixture and the application will automatically parse and predict the upcoming matches for you. You may also choose to export these predictions to a csv file, which you can open with Microsoft Excel.

![fixture parsing & upcoming match prediction](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/fixtures.png)

# Requirements

A `requirements.txt` file has been added to the project directory. However, the following table also presents the required libraries. Check the `requirements.txt` file for library versions.

| Library/Module  | Download Url | Installation |
| ------------- | ------------- | -------------
| Python Language | https://www.python.org/ | Download from website |
| Numpy  | https://numpy.org/ | `pip install numpy` |
| Pandas  | https://pandas.pydata.org/ | `pip install pandas` |
| Matplotlib  | https://matplotlib.org/ | `pip install matplotlib` |
| Seaborn  | https://seaborn.pydata.org/ | `pip install seaborn` |
| Scikit-Learn  | https://scikit-learn.org/stable/ | `pip install scikit-learn` |
| XGBoost  | https://xgboost.readthedocs.io/en/stable/ | `pip install xgboost` |
| Tensorflow  | https://www.tensorflow.org/ | `pip install tensorflow` |
| Tensorflow-Addons  | https://www.tensorflow.org/addons | `pip install tensorflow_addons` |
| TKinter  | https://docs.python.org/3/library/tkinter.html | `pip install tk ` |
| Optuna | https://optuna.org/ | `pip install optuna` |
| Fuzzy-Wuzzy | https://pypi.org/project/py-stringmatching (https://pypi.org/project/fuzzywuzzy/) | `pip install fuzzywuzzy` |

To run `pip` commands, open CMD (windows) using Window Key + R or by typing cmd on the search. In linux, You can use the linux terminal.

# Instructions (How to Run)

1. Download & Install python. During the installation, you should choose  **add to "Path"** It is recommended to download python 3.9.
2. After you download & install python, you can Download the above libraries using pip module (e.g. `pip install numpy`). These modules can be installed via the cmd (in windows) or terminal (in linux). 
3. On windows, you can double click the main.py file. Alternatively (Both Windows & Linux), You can open the cmd on the project directory and run: `python main.py`. 

# Supported Platforms
1. Windows
2. Linux
3. Mac

# Contact Me
For further questions or to report any bugs, don't hesitate to email me at: "kohliaridis97@gmail.com"

# Release (2023/01/19)

* Improved Graphical User Interface (GUI)
* Added Custom Themes
* Added "Ensemble" Model
* Fixtures are now imported, even if odds are missing. You can also manually add them or edit them.
* Fixed Bugs (Leagues not updating, Fixtures not being imported, etc.)

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
* Updated Documentation

![Training Parameters](https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/blob/main/screenshots/parameters.png)

# Release (2022/11/12)

* Fixed a bug where leagues wouldn't be updated up to last day
