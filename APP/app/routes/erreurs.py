from ..app import app
from flask import render_template, abort
from werkzeug.exceptions import BadRequest, BadHost, BadGateway, BadRequestKeyError
import json 


@app.errorhandler(400)
@app.errorhandler(404)
def page_not_found(e):
	"""
	Route permettant de gérer les erreurs 404 (ressource non trouvée)
	
	Parameters
	----------
	error : exception
		L'objet correspondant à l'erreur
	
	Returns
	-------
	template
		Retourne le template '404.html' ainsi que le code d'erreur 404
	"""
	return render_template('pages/erreurs/404.html'), 404

@app.errorhandler(403)
def forbiden(e):
	"""
	Route permettant de gérer les erreurs 403 (accès interdit)
	
	Parameters
	----------
	error : exception
		L'objet correspondant à l'erreur
	
	Returns
	-------
	template
		Retourne le template '403.html' ainsi que le code d'erreur 403
	"""
	return render_template('pages/erreurs/403.html'), 403

@app.errorhandler(500)
@app.errorhandler(503)
def service_unavailable(e):
	"""
	Route permettant de gérer les erreurs de type 500
		500 : erreur interne de serveur
		503 : service indisponible
	
	Parameters
	----------
	error : exception
		L'objet correspondant à l'erreur
	
	Returns
	-------
	template
		Retourne le template '500.html' ainsi que le code d'erreur 500
	"""
	db.session.rollback()
	return render_template('pages/erreurs/500.html'), 500



