from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path, mkdir
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"
MODELS_DIR = "models"


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "hjshjhdjah kjshkjdhjs"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_NAME}"
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(auth, url_prefix="/")

    from .models import User

    with app.app_context():
        db.create_all()

    create_storage_folders()

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_storage_folders():
    if not path.exists("instance/" + MODELS_DIR):
        mkdir("instance/" + MODELS_DIR)
