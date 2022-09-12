__doc__ = """Ce module contient la définition des variables de chemins de destination ainsi que l'importation des librairies nécessaires à la définition des 2 fonctions suivantes:
	- fetch_food_data() : qui sert à télécharger dans un sous-dossier du dossier de travail, 'datasets', le fichier .csv contenant 	le jeu de données complet issu de https://fr.openfoodfacts.org/data
	- load_food_data() : qui sert à charger ce jeu de données dans un DataFrame
"""


import os
import urllib
import zipfile
import pandas as pd

DOWNLOAD_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-scientist/P2/fr.openfoodfacts.org.products.csv.zip"
ZIP_NAME = "fr.openfoodfacts.org.products.csv.zip"
FILE_NAME = "fr.openfoodfacts.org.products.csv"
FOOD_PATH = "datasets"
FILE_PATH = os.path.join(FOOD_PATH, FILE_NAME)
ZIP_PATH = os.path.join(FOOD_PATH, ZIP_NAME)

def fetch_food_data(food_path=FOOD_PATH, dwnld_url=DOWNLOAD_URL, zip_path=ZIP_PATH, file_name=FILE_NAME):

	"""fonction d'extraction des données depuis https://world.openfoodfacts.org/data"""

	if not os.path.isdir(food_path):
    		os.makedirs(food_path)
	urllib.request.urlretrieve(dwnld_url, zip_path)

		with zipfile.ZipFile(zip_path, mode="r") as archive:
    		archive.extract(file_name, food_path)

	os.remove(zip_path)

def load_food_data(file_path=FILE_PATH):

	"""fonction de chargement des données extraites dans un dataframe"""

	return pd.read_csv(file_path, delimiter ="\t", on_bad_lines='skip')
