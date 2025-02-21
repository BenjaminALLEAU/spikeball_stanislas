import pandas as pd
import streamlit as st
import locale
import matplotlib.pyplot as plt
from function.function import *

# Configuration de la locale pour utiliser les espaces comme séparateurs de milliers
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
st.set_page_config(layout="wide", page_title="Analyse Ligue Stanistas", page_icon="dart", initial_sidebar_state="auto")

# Charger le fichier transformé
df_expanded = pd.read_pickle("data_preprocess/df_resultat_expanded.pkl")

# Convertir "Num Court" et "week" en types numériques
df_expanded["Num Court"] = pd.to_numeric(df_expanded["Num Court"], errors='coerce')
df_expanded["week"] = pd.to_numeric(df_expanded["week"], errors='coerce')

# Interface Streamlit
st.title("Stats Ligue Stanislas")

# Création des onglets
onglets = ["🔮 Prédiction de match", "📊 Analyse d'un joueur"]
tabs = st.tabs(onglets)
joueurs = sorted(df_expanded["Joueur"].unique())
# Onglet Prédiction de match
with tabs[0]:
    st.subheader("Prédiction d'un match")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Équipe A")
        joueur1 = st.selectbox("Sélectionner le premier joueur de l'équipe A", joueurs, key="joueur1")
        coequipier = st.selectbox("Sélectionner le coéquipier de l'équipe A", joueurs, key="coequipier")

    with col2:
        st.subheader("Équipe B")
        adversaire1 = st.selectbox("Sélectionner le premier joueur de l'équipe B", joueurs, key="adversaire1")
        adversaire2 = st.selectbox("Sélectionner le coéquipier de l'équipe B", joueurs, key="adversaire2")

    if st.button("Prédire le résultat"):
        resultat, proba = predire_match(joueur1, coequipier, adversaire1, adversaire2)
        # Affichage du score sous forme graphique
        st.markdown(f"""
        <style>
            .scoreboard {{
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 24px;
                font-weight: bold;
                background: linear-gradient(to right, #3b82f6, #1e40af);
                color: white;
                padding: 10px;
                border-radius: 10px;
                width: 60%;
                margin: auto;
                text-align: center;
            }}
            .team {{
                flex: 1;
                text-align: center;
            }}
            .score {{
                font-size: 40px;
                margin: 0 15px;
            }}
        </style>
        <div class="scoreboard">
            <div class="team">{joueur1}/{coequipier}</div>
            <div class="score">{proba * 100:.0f}% - {(1 - proba) * 100:.0f}%</div>
            <div class="team">{adversaire1}/{adversaire2}</div>
        </div>
        """, unsafe_allow_html=True)


# Onglet Analyse d'un joueur
with tabs[1]:
    # Sélection d'un joueur
    joueur_selectionne = st.selectbox("Sélectionner un joueur", joueurs)

    # Filtrer les données pour le joueur sélectionné
    df_joueur = df_expanded[df_expanded["Joueur"] == joueur_selectionne].copy()

    # Correction des résultats erronés
    df_joueur["Résultat"] = df_joueur["Résultat"].str.strip()

    # Ajout des indicateurs
    st.subheader("Statistiques du joueur")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        terrain_moyen = df_joueur["Num Court"].mean()
        st.metric(label="Court moyen joué", value=round(terrain_moyen, 2))

    with col2:
        tx_victoire = (df_joueur["Résultat"] == "Gagné").mean() * 100
        st.metric(label="Taux de victoires", value=f"{round(tx_victoire, 2)}%")

    with col3:
        diff_moyenne = df_joueur["Diff Score"].mean()
        st.metric(label="Différence moyenne de score", value=round(diff_moyenne, 2))

    with col4:
        nb_matchs = df_joueur.shape[0]
        st.metric(label="Nombre total de matchs", value=nb_matchs)

    # Sélection du nombre minimum de matchs
    st.subheader("🏆 Classements des performances")
    n_matchs_min = st.slider("Nombre minimum de matchs pris en compte", 1, 10, 3)

    # Filtrer les coéquipiers et adversaires avec au moins n_matchs_min matchs
    coequipiers_valides = df_joueur["Coéquipier"].value_counts()
    adversaires_valides = df_joueur["Adversaires"].str.split(', ').explode().value_counts()

    coequipiers_valides = coequipiers_valides[coequipiers_valides >= n_matchs_min]
    adversaires_valides = adversaires_valides[adversaires_valides >= n_matchs_min]

    # Affichage des podiums côte à côte
    col1, col2, col3 = st.columns(3)

    # Calcul du % de victoire avec chaque coéquipier
    if not coequipiers_valides.empty:
        win_rates_coequipiers = (df_joueur[df_joueur["Résultat"] == "Gagné"]["Coéquipier"].value_counts() / 
                                df_joueur["Coéquipier"].value_counts() * 100).dropna()
        win_rates_coequipiers = win_rates_coequipiers[win_rates_coequipiers.index.isin(coequipiers_valides.index)]
        top3_coequipiers = win_rates_coequipiers.nlargest(3)
        
        with col1:
            st.subheader("🤝 Mes meilleurs teammates")
            st.text("(% de victoires avec)")
            for i, (name, value) in enumerate(top3_coequipiers.items(), start=1):
                total_matches = df_joueur[df_joueur["Coéquipier"] == name].shape[0]
                st.write(f"🥇🥈🥉"[i-1] + f" {name}: {round(value, 2)}% de victoires ({total_matches} matchs)")

    # Calcul du % de victoire contre chaque adversaire
    if not adversaires_valides.empty:
        win_rates_adversaires = (df_joueur[df_joueur["Résultat"] == "Gagné"]["Adversaires"].str.split(', ').explode().value_counts() / 
                                df_joueur["Adversaires"].str.split(', ').explode().value_counts() * 100).dropna()
        win_rates_adversaires = win_rates_adversaires[win_rates_adversaires.index.isin(adversaires_valides.index)]
        top3_adversaires = win_rates_adversaires.nlargest(3)
        
        with col2:
            st.subheader("⚔️ Mes victimes")
            st.text("(% de victoires contre)")
            for i, (name, value) in enumerate(top3_adversaires.items(), start=1):
                total_matches = df_joueur[df_joueur["Adversaires"].str.contains(name, regex=False)].shape[0]
                st.write(f"🥇🥈🥉"[i-1] + f" {name}: {round(value, 2)}% de victoires ({total_matches} matchs)")

    # Calcul du % de défaites contre chaque adversaire
    if not adversaires_valides.empty:
        loss_rates_adversaires = (df_joueur[df_joueur["Résultat"] == "Perdu"]["Adversaires"].str.split(', ').explode().value_counts() / 
                                df_joueur["Adversaires"].str.split(', ').explode().value_counts() * 100).dropna()
        loss_rates_adversaires = loss_rates_adversaires[loss_rates_adversaires.index.isin(adversaires_valides.index)]
        top3_adversaires_perdu = loss_rates_adversaires.nlargest(3)
        
        with col3:
            st.subheader("🏆 Mes bourreaux")
            st.text("(% de défaites contre)")
            for i, (name, value) in enumerate(top3_adversaires_perdu.items(), start=1):
                total_matches = df_joueur[df_joueur["Adversaires"].str.contains(name, regex=False)].shape[0]
                st.write(f"🥇🥈🥉"[i-1] + f" {name}: {round(value, 2)}% de défaites ({total_matches} matchs)")