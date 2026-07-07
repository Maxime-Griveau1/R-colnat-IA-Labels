from flask import Flask, logging
from flask_sqlalchemy import SQLAlchemy
from .config import Config
from flask_login import LoginManager


app = Flask(
    __name__, 
    template_folder='templates',
    static_folder='statics')
app.config.from_object(Config)

from .routes import generales, erreurs, annotation, fields, htr, correct, resultats, demo