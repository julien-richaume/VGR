import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

def build_hybrid_engine(df):
    # 1. PRÉPARATION DES DONNÉES (Tags)
    # Création de la matrice de tags (One-Hot Encoding)
    tags_split = df['tags'].str.split(', ', expand=True).stack()
    tags_dummies = pd.get_dummies(tags_split).groupby(level=0).sum()
    
    # 2. PRÉPARATION DU COMPORTEMENT (Playtime)
    matrix = df.pivot_table(index='user_id', columns='title', values='playtime').fillna(0)
    
    # 3. FILTRAGE COLLABORATIF (SVD)
    matrix_values = matrix.values.astype(float)
    U, sigma, Vt = svds(matrix_values, k=50)
    sigma = np.diag(sigma)
    collaborative_preds = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(collaborative_preds, columns=matrix.columns, index=matrix.index)
    
    return matrix, preds_df, tags_dummies

def get_hybrid_recommendations(user_id, matrix, preds_df, tags_dummies, success_ratios, alpha=1, beta=1):
    # Intersection stricte des jeux
    common_games = matrix.columns.intersection(tags_dummies.index).intersection(preds_df.columns)
    
    # 2. Alignement explicite et sécurisé
    tags_aligned = tags_dummies.loc[common_games]
    #print(tags_aligned)
    matrix_aligned = matrix[common_games]
    preds_aligned = preds_df[common_games] # Maintenant, common_games est garanti d'exister ici
    ratios_aligned = success_ratios.loc[common_games]    
    # Scores
    collab_scores = preds_aligned.loc[user_id]
    user_playtime = matrix_aligned.loc[user_id]
    user_profile = user_playtime.values.dot(tags_aligned.values)
    
    if np.all(user_profile == 0):
        final_scores = collab_scores
    else:
        content_scores = cosine_similarity(user_profile.reshape(1, -1), tags_aligned.values).flatten()
        content_scores = pd.Series(content_scores, index=common_games)
        final_scores = (alpha * collab_scores) + ((1 - alpha) * content_scores)
    
    # Filtrage des jeux déjà possédés
    played = matrix.columns[matrix.loc[user_id] > 0]
    final_scores = final_scores.drop(played, errors='ignore')
    
    # Application du multiplicateur
    final_scores = final_scores * (1 + (beta * ratios_aligned.loc[final_scores.index]))
    
    return final_scores.sort_values(ascending=False).head(5)
    


# LOAD DATA 
steam200k = pd.read_csv("data/steam200k.csv")

steam200k.rename(
    columns ={
        steam200k.columns[0] : "user_id",
        steam200k.columns[1] : "title",
        steam200k.columns[2] : "status",
        steam200k.columns[3] : "playtime",
        steam200k.columns[4] : "osef"
    },
    inplace=True
)

steam200k = steam200k[steam200k["status"] == "play"]

steam200k.drop(
    columns=[
        "status",
        "osef"
    ],
    inplace=True
)

games = pd.read_csv("data/games.csv").sort_values(by="app_id", ascending=True)

games_metadata = pd.read_json("data/games_metadata.json", lines=True)
games = pd.merge(games, games_metadata, on="app_id")

games_and_user_playtime = pd.merge(games, steam200k, on="title")
games_and_user_playtime


# --- WORKFLOW ---
# 1. Merger les données
games_and_user_playtime = pd.merge(games, steam200k, on="title")
# 2. Construire le moteur
success_ratios = games_and_user_playtime[['title', 'positive_ratio']].drop_duplicates('title').set_index('title')['positive_ratio']

matrix, preds, tags = build_hybrid_engine(games_and_user_playtime)
recos = get_hybrid_recommendations(
    games_and_user_playtime["user_id"].value_counts().index[150], 
    matrix, preds, tags, success_ratios
)

import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="Steam Recommender", layout="wide")

st.title("🎮 Steam AI Recommender")
st.markdown("---")

# --- SIDEBAR : Profil Nouvel Utilisateur ---
st.sidebar.header("👤 Profil utilisateur")
selected_games = st.sidebar.multiselect(
    "Quels jeux avez-vous aimés ?", 
    options=matrix.columns.tolist(),
    default=["Portal 2", "Half-Life 2"]
)

# --- LOGIQUE DE RECOMMANDATION ADAPTÉE ---
def get_recommendations_for_new_user(liked_games, matrix, preds_df, tags_dummies, success_ratios):
    # Création d'un profil fictif (moyenne des jeux aimés)
    user_profile_vec = matrix[liked_games].mean(axis=1)
    
    # ... (le reste de ta logique de calcul hybride) ...
    # Ici, tu appelles get_hybrid_recommendations en adaptant l'ID
    return recos

# --- INTERFACE PRINCIPALE ---
if st.sidebar.button("Générer mes recommandations"):
    # Calcul des recommandations (remplace par ton appel)
    recos = get_hybrid_recommendations(..., matrix, preds, tags, success_ratios)
    
    st.subheader(f"✨ Suggestions basées sur : {', '.join(selected_games)}")
    
    # Affichage en grille (3 colonnes)
    cols = st.columns(3)
    for i, (game, score) in enumerate(recos.head(6).items()):
        with cols[i % 3]:
            # Utilisation de st.metric pour un rendu moderne
            st.metric(label=game, value=f"{score*100:.0f}%")
            
            # Affichage stylisé des tags
            game_tags = tags.loc[game]
            active_tags = game_tags[game_tags > 0].index.tolist()
            st.caption(f"Tags: {', '.join(active_tags[:3])}")
            
            if st.button(f"Voir détails {game}", key=i):
                st.info(f"Pourquoi ce choix ? Basé sur votre intérêt pour les jeux {active_tags[0]}")
else:
    st.info("👈 Ajoutez des jeux dans la barre latérale pour obtenir vos recommandations.")