from flask import Flask, render_template, request, g, Response
from league import CreateLeagueForm, LoadLeagueForm, DeleteLeagueForm
from plots import CorrelationPlotter, ClassDistributionPlotter, ImportancePlotter
from model.tuning import TuningRFForm
from flask_wtf.csrf import CSRFProtect
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
import variables
import secrets
import matplotlib
import time

matplotlib.use('agg')


app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key
csrf = CSRFProtect()
csrf.init_app(app)

MODEL_REPO = None
LEAGUE_REPO = None
CURRENT_CONTEXT = None


def get_model_repo():
    global MODEL_REPO
    if not MODEL_REPO:
        MODEL_REPO = ModelRepository(models_checkpoint_directory=variables.models_checkpoint_directory)
    return MODEL_REPO

def get_league_repo():
    global LEAGUE_REPO
    if not LEAGUE_REPO:
        LEAGUE_REPO = LeagueRepository(
            available_leagues_filepath=variables.available_leagues_filepath,
            saved_leagues_directory=variables.saved_leagues_directory
        )
    return LEAGUE_REPO

def get_global_context():
    global CURRENT_CONTEXT
    if CURRENT_CONTEXT is not None:
        return CURRENT_CONTEXT
    return None

def store_global_context(context):
    global CURRENT_CONTEXT
    CURRENT_CONTEXT = context

# Define your Flask routes
@app.route('/')
def index():
    context = get_global_context()
    if context:
        return render_template('index.html', context=context)
    return render_template('index.html')


@app.route('/create_league', methods=['GET', 'POST'])
def create_league():
    LEAGUE_REPO = get_league_repo()
    form = CreateLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        context = {'matches': matches_df, 'league_name': league_name}
        store_global_context(context)
        return render_template('index.html', context=context)

    return render_template('create_league.html', form=form)


@app.route('/load_league', methods=['GET', 'POST'])
def load_league():
    LEAGUE_REPO = get_league_repo()
    form = LoadLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        context = {'matches': matches_df, 'league_name': league_name}
        store_global_context(context)
        return render_template('index.html', context=context)
    # Handle the 'Load League' action here
    return render_template('load_league.html', form=form)


@app.route('/delete_league', methods=['GET', 'POST'])
def delete_league():
    LEAGUE_REPO = get_league_repo()
    form = DeleteLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        form.submit()
        return render_template('delete_league.html', form=form)
    # Handle the 'Load League' action here
    return render_template('delete_league.html', form=form)

@app.route('/plot_correlations', methods=['GET', 'POST'])
def plot_correlations():
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        form = CorrelationPlotter(loaded_df)
        if request.method == 'POST' :
            img = form.generate_image()
            return render_template('plot_correlation.html', image_data=img, form=form)
        return render_template('plot_correlation.html', form=form)

    return render_template('index.html')

@app.route('/plot_importance', methods=['GET', 'POST'])
def plot_importance():
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        form = ImportancePlotter(loaded_df)
        if request.method == 'POST' :
            img = form.generate_image()
            return render_template('plot_importance.html', image_data=img, form=form)
        return render_template('plot_importance.html', form=form)

    return render_template('index.html')

@app.route('/plot_target_distribution')
def plot_target_distribution():
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        form = ClassDistributionPlotter(loaded_df)
        img = form.generate_image()
        return render_template('plot_classes.html', image_data=img)
    return render_template('index.html')

    # Handle the 'Target Distribution' action here
@app.route('/tune_nn')
def tune_nn():
    return "Neural Network (Auto Tuning) page"
    # Handle the 'Neural Network (Auto Tuning)' action here

@app.route('/train_custom_nn')
def train_custom_nn():
    # Handle the 'Neural Network (Custom)' action here
    return "Neural Network (Custom) page"

@app.route('/tune_rf', methods=['GET', 'POST'])
def tune_rf():
    # Handle the 'Random Forest (Auto Tuning)' action here
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        league_name = context["league_name"]
        form = TuningRFForm(get_model_repo(), league_name, 0, loaded_df)
        if request.method == 'POST' :
            img = form.submit_tuning()
            return render_template('tuning_model.html', image_data=img, form=form)
        return render_template('tuning_model.html', form=form)

    return render_template('index.html')

@app.route('/train_custom_rf')
def train_custom_rf():
    # Handle the 'Random Forest (Custom)' action here
    return "Random Forest (Custom) page"

@app.route('/evaluate_models')
def evaluate_models():
    # Handle the 'Evaluate' action here

    return "Evaluate Models page"
@app.route('/predict_matches')
def predict_matches():
    # Handle the 'Predict Matches' action here
    return "Predict Matches page"

@app.route('/predict_fixture')
def predict_fixture():
    # Handle the 'Predict Fixture' action here
    return "Predict Fixture page"

if __name__ == '__main__':
    app.run(debug=True)
