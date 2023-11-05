from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json

from sqlalchemy import text
from flask import render_template, request, session
from league import CreateLeagueForm, LoadLeagueForm, DeleteLeagueForm
from plots import CorrelationPlotter, ClassDistributionPlotter, ImportancePlotter
from model.tuning import TuningRFForm
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
import variables
from pandas import DataFrame
import matplotlib
matplotlib.use('agg')

views = Blueprint('views', __name__)


@views.route('/delete-note', methods=['POST'])
def delete_note():  
    note = json.loads(request.data) # this function expects a JSON from the INDEX.js file 
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()

    return jsonify({})

def get_league_matches(league_name):
    query = text(f"SELECT * FROM '{league_name}'")
    result = db.session.execute(query).fetchall()
    return DataFrame(list(result))

MODEL_REPO = None
LEAGUE_REPO = None


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

# Define your Flask routes
@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if session:
        matches = get_league_matches(session['league_name'])

    return render_template("home.html", user=current_user, matches=matches)


@views.route('/create_league', methods=['GET', 'POST'])
def create_league():
    LEAGUE_REPO = get_league_repo()
    form = CreateLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        session['league_name'] = league_name
        return render_template('home.html', session=session, user=current_user, matches=matches_df)

    return render_template('create_league.html', form=form, user=current_user)


@views.route('/load_league', methods=['GET', 'POST'])
def load_league():
    form = LoadLeagueForm()
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        session['league_name'] = league_name
        return render_template('home.html', session=session, user=current_user, matches=matches_df)
    return render_template('load_league.html', form=form, user=current_user)


@views.route('/delete_league', methods=['GET', 'POST'])
def delete_league():
    LEAGUE_REPO = get_league_repo()
    form = DeleteLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        form.submit()
        return render_template('delete_league.html', form=form, user=current_user)
    return render_template('delete_league.html', form=form, user=current_user)

@views.route('/plot_correlations', methods=['GET', 'POST'])
def plot_correlations():
    if session:
        matches = get_league_matches(session['league_name'])
        form = CorrelationPlotter(matches)
        if request.method == 'POST' :
            img = form.generate_image()
            return render_template('plot_correlation.html', image_data=img, form=form, user=current_user)
        return render_template('plot_correlation.html', form=form, user=current_user)

    return render_template('home.html')

@views.route('/plot_importance', methods=['GET', 'POST'])
def plot_importance():
    if session:
        matches = get_league_matches(session['league_name'])
        form = ImportancePlotter(matches)
        if request.method == 'POST' :
            img = form.generate_image()
            return render_template('plot_importance.html', image_data=img, form=form, user=current_user)
        return render_template('plot_importance.html', form=form, user=current_user)

    return render_template('home.html')

@views.route('/plot_target_distribution')
def plot_target_distribution():
    if session:
        matches = get_league_matches(session['league_name'])
        form = ClassDistributionPlotter(matches)
        img = form.generate_image()
        return render_template('plot_classes.html', image_data=img, user=current_user)
    return render_template('home.html', user=current_user)

    # Handle the 'Target Distribution' action here
@views.route('/tune_nn')
def tune_nn():
    return "Neural Network (Auto Tuning) page"
    # Handle the 'Neural Network (Auto Tuning)' action here

@views.route('/train_custom_nn')
def train_custom_nn():
    # Handle the 'Neural Network (Custom)' action here
    return "Neural Network (Custom) page"

@views.route('/tune_rf', methods=['GET', 'POST'])
def tune_rf():
    # Handle the 'Random Forest (Auto Tuning)' action here
    if session:
        matches = get_league_matches(session['league_name'])
        league_name = session["league_name"]
        form = TuningRFForm(get_model_repo(), league_name, 0, matches)
        if request.method == 'POST' :
            img = form.submit_tuning()
            return render_template('tuning_model.html', image_data=img, form=form, user=current_user)
        return render_template('tuning_model.html', form=form, user=current_user)

    return render_template('home.html')

@views.route('/train_custom_rf')
def train_custom_rf():
    # Handle the 'Random Forest (Custom)' action here
    return "Random Forest (Custom) page"

@views.route('/evaluate_models')
def evaluate_models():
    # Handle the 'Evaluate' action here

    return "Evaluate Models page"
@views.route('/predict_matches')
def predict_matches():
    # Handle the 'Predict Matches' action here
    return "Predict Matches page"

@views.route('/predict_fixture')
def predict_fixture():
    # Handle the 'Predict Fixture' action here
    return "Predict Fixture page"
