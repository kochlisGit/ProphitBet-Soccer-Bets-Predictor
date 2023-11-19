import matplotlib
from flask import Blueprint, flash, render_template, request, session
from flask_login import current_user, login_required

import variables
from database.repositories.model import ModelRepository
from website.evaluation import EvaluationForm
from website.fixtures import FixturesForm
from website.league import CreateLeagueForm, DeleteLeagueForm, LoadLeagueForm
from website.plots import (
    ClassDistributionPlotter,
    CorrelationPlotter,
    ImportancePlotter,
)
from website.training import CustomTrainNNForm, CustomTrainRFForm
from website.tuning import TuningNNForm, TuningRFForm

from .dbwrapper import DBWrapper

matplotlib.use("agg")

views = Blueprint("views", __name__)

db = DBWrapper()
MODEL_REPO = None

def get_model_repo():
    global MODEL_REPO
    if not MODEL_REPO:
        MODEL_REPO = ModelRepository(
            models_checkpoint_directory=variables.models_checkpoint_directory
        )
    return MODEL_REPO


@views.route("/", methods=["GET", "POST"])
@login_required
def home():
    if session and db.league_exists(session.get("league_name", None)):
        matches = db.get_league_matches(session["league_name"])
        return render_template("home.html", user=current_user, matches=matches)
    flash("Load or create league please", "error")
    return render_template("home.html", user=current_user)


@views.route("/create_league", methods=["GET", "POST"])
def create_league():
    form = CreateLeagueForm()
    if request.method == "POST" and form.validate():
        league_name, matches_df = form.submit()
        session["league_name"] = league_name
        if matches_df is not None:
            return render_template(
                "home.html", session=session, user=current_user, matches=matches_df
            )
        else:
            flash("Dataset not created, no connection to internet")
    return render_template("create_league.html", form=form, user=current_user)


@views.route("/load_league", methods=["GET", "POST"])
def load_league():
    form = LoadLeagueForm()
    if request.method == "POST" and form.validate():
        league_name, matches_df = form.submit()
        session["league_name"] = league_name
        return render_template(
            "home.html", session=session, user=current_user, matches=matches_df
        )
    return render_template("load_league.html", form=form, user=current_user)


@views.route("/delete_league", methods=["GET", "POST"])
def delete_league():
    form = DeleteLeagueForm()
    if request.method == "POST" and form.validate():
        form.submit()
        session["league_name"] = None
        return render_template("delete_league.html", form=form, user=current_user)
    return render_template("delete_league.html", form=form, user=current_user)


@views.route("/plot_correlations", methods=["GET", "POST"])
def plot_correlations():
    if session:
        matches = db.get_league_matches(session["league_name"])
        form = CorrelationPlotter(matches)
        if request.method == "POST":
            img = form.generate_image()
            return render_template(
                "plot_correlation.html", image_data=img, form=form, user=current_user
            )
        return render_template("plot_correlation.html", form=form, user=current_user)

    return render_template("home.html")


@views.route("/plot_importance", methods=["GET", "POST"])
def plot_importance():
    if session:
        matches = db.get_league_matches(session["league_name"])
        form = ImportancePlotter(matches)
        if request.method == "POST":
            img = form.generate_image()
            return render_template(
                "plot_importance.html", image_data=img, form=form, user=current_user
            )
        return render_template("plot_importance.html", form=form, user=current_user)

    return render_template("home.html")


@views.route("/plot_target_distribution")
def plot_target_distribution():
    if session:
        matches = db.get_league_matches(session["league_name"])
        form = ClassDistributionPlotter(matches)
        img = form.generate_image()
        return render_template("plot_classes.html", image_data=img, user=current_user)
    return render_template("home.html", user=current_user)


def train_custom(form):
    if session:
        matches = db.get_league_matches(session["league_name"])
        league_name = session["league_name"]
        form = form(get_model_repo(), league_name, matches, 0)
        if request.method == "POST":
            form_validation = form.submit_training()
            flash(form_validation)
            return render_template(
                "training_model.html", form_validation=form_validation, form=form, user=current_user
            )
        return render_template("training_model.html", form=form, user=current_user)

    return render_template("home.html")


@views.route("/train_custom_nn", methods=["GET", "POST"])
def train_custom_nn():
    return train_custom(CustomTrainNNForm)


@views.route("/train_custom_rf", methods=["GET", "POST"])
def train_custom_rf():
    return train_custom(CustomTrainRFForm)


def tune(form):
    if session:
        matches = db.get_league_matches(session["league_name"])
        league_name = session["league_name"]
        form = form(get_model_repo(), league_name, 0, matches)
        if request.method == "POST":
            form.submit_tuning()
            flash("Check terminal")
            return render_template(
                "tuning_model.html", form=form, user=current_user
            )
        return render_template("tuning_model.html", form=form, user=current_user)

    return render_template("home.html")


@views.route("/tune_rf", methods=["GET", "POST"])
def tune_rf():
    return tune(TuningRFForm)


@views.route("/tune_nn", methods=["GET", "POST"])
def tune_nn():
    return tune(TuningNNForm)


@views.route("/evaluate_models", methods=["GET", "POST"])
def evaluate_models():
    if session:
        matches = db.get_league_matches(session["league_name"])
        league_name = session["league_name"]
        form = EvaluationForm(get_model_repo(), league_name, matches)
        if request.method == "POST":
            matches_df, metrics = form.submit_evaluation_task()
            return render_template(
                "evaluation.html", matches=matches_df, metrics=metrics, form=form, user=current_user
            )
        return render_template("evaluation.html", form=form, user=current_user)

    return render_template("home.html")


@views.route("/predict_fixture", methods=["GET", "POST"])
def predict_fixture():
    if session:
        matches_df = db.get_league_matches(session["league_name"])
        fixture_url = db.get_fixture_url_from_league_name(session["league_name"])
        league_name = session["league_name"]
        form = FixturesForm(matches_df, get_model_repo(), league_name, fixture_url)
        if request.method == "POST":
            button_pressed = request.form['button']
            matches = form.import_fixture()
            if button_pressed == 'Predict':
                matches = form.predict_fixture(matches)
            return render_template(
                "fixtures.html", matches=matches, fixture_url=fixture_url, form=form, user=current_user
            )
        return render_template("fixtures.html", form=form, fixture_url=fixture_url, user=current_user)

    return render_template("home.html")
