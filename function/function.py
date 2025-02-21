import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_pickle("data_preprocess/df_resultat_expanded.pkl")
# Sélection des colonnes pertinentes
df = df[["Num Court", "Joueur", "Coéquipier", "Adversaires", "Résultat"]].copy()

# Convertir les adversaires en deux colonnes distinctes
df[["Adversaire_1", "Adversaire_2"]] = df["Adversaires"].str.split(", ", expand=True)
df.drop(columns=["Adversaires"], inplace=True)

# Convertir "Num Court" en entier
df["Num Court"] = pd.to_numeric(df["Num Court"], errors='coerce')
df.dropna(subset=["Num Court"], inplace=True)  # Supprimer les lignes avec des valeurs non valides
df["Num Court"] = df["Num Court"].astype(int)  # Assurer un type entier

# Calcul du terrain moyen pour chaque joueur
terrain_moyen = df.groupby("Joueur")["Num Court"].mean()

# Calcul du taux de victoire moyen pour chaque joueur
taux_victoire = df.groupby("Joueur")["Résultat"].apply(lambda x: (x == "Gagné").mean())

def get_terrain_moyen(joueur):
    return terrain_moyen.get(joueur, df["Num Court"].mean())  # Valeur moyenne si joueur inconnu

def get_taux_victoire(joueur):
    return taux_victoire.get(joueur, 0.5)  # 50% par défaut si inconnu


# Fonction pour prédire un match
def predire_match(joueur1, coequipier, adversaire1, adversaire2):
    """Prédit la probabilité de victoire en utilisant les terrains moyens et taux de victoire des équipes."""
    
    # Charger le modèle et les données
    model = joblib.load("models/match_prediction_model.pkl")
    terrain_moyen, taux_victoire = joblib.load("models/terrain_taux_victoire.pkl")
    
    # Récupérer les terrains moyens et taux de victoire
    terrain_equipe = np.mean([get_terrain_moyen(joueur1), get_terrain_moyen(coequipier)])
    terrain_adversaires = np.mean([get_terrain_moyen(adversaire1), get_terrain_moyen(adversaire2)])
    taux_victoire_equipe = np.mean([get_taux_victoire(joueur1), get_taux_victoire(coequipier)])
    taux_victoire_adversaires = np.mean([get_taux_victoire(adversaire1), get_taux_victoire(adversaire2)])
    
    # Créer l'entrée
    input_data = np.array([[terrain_equipe, terrain_adversaires, taux_victoire_equipe, taux_victoire_adversaires]])
    
    # Prédiction
    prob = model.predict_proba(input_data)[0][1]  # Probabilité de victoire
    
    return f"Probabilité de victoire de {joueur1} et {coequipier} contre {adversaire1} et {adversaire2} : {prob * 100:.2f}%",prob