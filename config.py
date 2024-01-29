from database.entities.leagues import *
from models import estimators

# Database Config
leagues_directory = 'database/storage/leagues'
leagues_index_filepath = 'database/storage/leagues/index.pkl'
models_directory = 'database/storage/models'
models_index_filepath = 'database/storage/models/index.pkl'

all_leagues_dict = {
    'Argentina': [PrimeraDivision()],
    'Belgium': [JupilerLeague()],
    'Brazil': [BrazilSerieA()],
    'China': [ChinaSuperLeague()],
    'Denmark': [SuperLiga()],
    'England': [PremierLeague(), Championship(), League1(), League2()],
    'Finland': [VeikkausLiiga()],
    'France': [Ligue1(), Ligue2()],
    'Germany': [Bundesliga1(), Bundesliga2()],
    'Greece': [SuperLeague()],
    'Ireland': [IrelandPremierDivision()],
    'Italy': [SerieA(), SerieB()],
    'Japan': [J1()],
    'Mexico': [LigaMX()],
    'Netherlands': [Eredivisie()],
    'Norgway': [Eliteserien()],
    'Poland': [Ekstraklasa()],
    'Portugal': [Liga1()],
    'Romania': [RomaniaLiga1()],
    'Russia': [RussiaPremierLeague()],
    'Scotland': [Premiership()],
    'Spain': [LaLiga(), SegundaDivision()],
    'Sweden': [Allsvenskan()],
    'Switzerland': [SwitzerlandSuperLeague()],
    'USA': [MLS()],
    'Turkey': [SuperLig()]
}

# Models config
normalizers = ['None', 'Min-Max', 'Max-Abs', 'Standard', 'Robust']
samplers = ['None', 'Random-UnderSampling', 'Near-Miss', 'Random-OverSampling', 'SVM-SMOTE', 'SMOTE-NN']
fit_test_size = 100

# App Config
app_title = 'Prophit-Bet (v2)'
themes_dict = {
    'Default': 'winnative',
    'forest-light': 'database/storage/themes/forest/forest-light.tcl',
    'forest-dark': 'database/storage/themes/forest/forest-dark.tcl',
    'breeze': 'database/storage/themes/breeze/breeze/breeze.tcl',
    'breeze-dark': 'database/storage/themes/breeze/breeze-dark/breeze-dark.tcl'
}
help_url_links = {
    'About': {
        'About Me': 'https://kochlisgit.github.io/aboutme/',
        'More Applications': 'https://github.com/kochlisGit?tab=repositories'
    },
    'Machine Learning': {
        'ML vs SM': 'https://www.turintech.ai/machine-learning-vs-statistical-modelling-which-one-is-right-for-your-business-problem/',
        'Decision Tree': 'https://www.ibm.com/topics/decision-trees',
        'KNN': 'https://www.ibm.com/topics/knn',
        'Logistic Regression': 'https://www.datacamp.com/tutorial/understanding-logistic-regression-python',
        'Naive Bayes': 'https://www.ibm.com/topics/naive-bayes',
        'Neural Network': 'https://www.ibm.com/topics/neural-networks',
        'XGBoost': 'https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html',
        'Random Forest': 'https://builtin.com/data-science/random-forest-algorithm',
        'SVM': 'https://www.linkedin.com/pulse/support-vector-machine-algorithm-svms-dishant-kharkar/'
    },
    'Evaluation Metrics': {
        'Evaluation Metrics': 'https://www.kdnuggets.com/2020/04/performance-evaluation-metrics-classification.html',
        'Model Evaluation': 'https://www.linkedin.com/pulse/train-test-validation-three-pillars-accurate-machine-jagarlapoodi/',
        'Cross Validation': 'https://www.linkedin.com/pulse/cross-validation-machine-learning-ishan-shah/',
        'Percentiles': 'https://www.w3schools.com/datascience/ds_stat_percentiles.asp'
    },
    'Feature Preprocessing': {
        'Imbalanced Classes': 'https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data',
        'Sampling Methods': 'https://www.turintech.ai/what-is-imbalanced-data-and-how-to-handle-it/',
        'Correlation Analysis': 'https://www.questionpro.com/features/correlation-analysis.html',
        'Variance': 'https://stats.stackexchange.com/questions/488989/in-feature-selection-what-is-the-reason-for-considering-removing-low-variance-f',
        'Normalization': 'https://www.kdnuggets.com/2020/04/data-transformation-standardization-normalization.html'
    }
}

# Fixtures
browsers = ['chrome', 'firefox', 'edge']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
days = [f'{i}' for i in range(1, 32)]

fixtures_class = 'full-matches-table mt1e '
fixture_date_class = 'fs11e lh14e'
team_name_class = 'hover-modal-parent hover-modal-ajax-team'
odd_row_class = 'stat odds dark-gray bbox'
odd_values_class = 'col-lg-4 col-sm-4 ac '


# Training Config
random_seed = 0
