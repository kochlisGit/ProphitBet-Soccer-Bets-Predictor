from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json

from flask import Flask, render_template, request, session
from league import CreateLeagueForm, LoadLeagueForm, DeleteLeagueForm
from plots import CorrelationPlotter, ClassDistributionPlotter, ImportancePlotter
from model.tuning import TuningRFForm
from database.repositories.league import LeagueRepository
from database.repositories.model import ModelRepository
import variables

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

# Define your Flask routes
@views.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        note = request.form.get('note')#Gets the note from the HTML

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note
            db.session.add(new_note) #adding the note to the database
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)


@views.route('/create_league', methods=['GET', 'POST'])
def create_league():
    LEAGUE_REPO = get_league_repo()
    form = CreateLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        #session['matches'] = matches_df.to_json()
        session['league_name'] = league_name
        return render_template('index.html', session=session, user=current_user)

    return render_template('create_league.html', form=form, user=current_user)


@views.route('/load_league', methods=['GET', 'POST'])
def load_league():
    LEAGUE_REPO = get_league_repo()
    form = LoadLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        league_name, matches_df = form.submit()
        context = {'matches': matches_df, 'league_name': league_name}
        store_global_context(context)
        return render_template('index.html', context=context, user=current_user)
    # Handle the 'Load League' action here
    return render_template('load_league.html', form=form, user=current_user)


@views.route('/delete_league', methods=['GET', 'POST'])
def delete_league():
    LEAGUE_REPO = get_league_repo()
    form = DeleteLeagueForm(league_repository=LEAGUE_REPO)
    if request.method == 'POST' and form.validate():
        form.submit()
        return render_template('delete_league.html', form=form, user=current_user)
    # Handle the 'Load League' action here
    return render_template('delete_league.html', form=form, user=current_user)

@views.route('/plot_correlations', methods=['GET', 'POST'])
def plot_correlations():
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        form = CorrelationPlotter(loaded_df)
        if request.method == 'POST' :
            img = form.generate_image()
            return render_template('plot_correlation.html', image_data=img, form=form, user=current_user)
        return render_template('plot_correlation.html', form=form, user=current_user)

    return render_template('index.html')

@views.route('/plot_importance', methods=['GET', 'POST'])
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

@views.route('/plot_target_distribution')
def plot_target_distribution():
    context = get_global_context()
    if context:
        loaded_df = context["matches"]
        form = ClassDistributionPlotter(loaded_df)
        img = form.generate_image()
        return render_template('plot_classes.html', image_data=img)
    return render_template('index.html')

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
