from flask import Flask, render_template, request, g
from league import CreateLeagueForm
from flask_wtf.csrf import CSRFProtect
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
import variables
import secrets

app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key
csrf = CSRFProtect()
csrf.init_app(app)


def get_model_repo():
    if not hasattr(g, 'model_repo'):
        g.model_repo = ModelRepository(models_checkpoint_directory=variables.models_checkpoint_directory)
    return g.model_repo

def get_league_repo():
    if not hasattr(g, 'league_repo'):
        g.league_repo = LeagueRepository(
            available_leagues_filepath=variables.available_leagues_filepath,
            saved_leagues_directory=variables.saved_leagues_directory
        )
    return g.league_repo


# Define your Flask routes
@app.route('/')
def index():
    league_repo = get_league_repo()
    return render_template('index.html')

@app.route('/create_league', methods=['GET', 'POST'])
def create_league():
    league_repo = get_league_repo()
    form = CreateLeagueForm(league_repository=league_repo)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        context = {'matches': matches_df.to_html(classes='table table-bordered', escape=False), 'league_name': league_name}
        return render_template('index.html', context=context)

        # Handle form submission, e.g., save the form data to your database
        # Access form data using form.selected_league.data, form.league_name.data, etc.

    return render_template('create_league.html', form=form)

@app.route('/load_league')
def load_league():
    # Handle the 'Load League' action here
    return "Load League page"

@app.route('/delete_league')
def delete_league():
    # Handle the 'Delete League' action here
    return "Delete League page"

@app.route('/plot_correlations')
def plot_correlations():
    # Handle the 'Correlations' action here
    return "Correlations page"

@app.route('/plot_importance')
def plot_importance():
    # Handle the 'Feature Importance' action here
    return "Feature Importance page"

@app.route('/plot_target_distribution')
def plot_target_distribution():
    return "Target Distribution page"

    # Handle the 'Target Distribution' action here
@app.route('/tune_nn')
def tune_nn():
    return "Neural Network (Auto Tuning) page"
    # Handle the 'Neural Network (Auto Tuning)' action here

@app.route('/train_custom_nn')
def train_custom_nn():
    # Handle the 'Neural Network (Custom)' action here
    return "Neural Network (Custom) page"

@app.route('/tune_rf')
def tune_rf():
    # Handle the 'Random Forest (Auto Tuning)' action here
    return "Random Forest (Auto Tuning) page"

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
