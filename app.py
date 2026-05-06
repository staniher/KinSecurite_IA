"""
APPLICATION PRINCIPALE KINSÉCURITÉ IA
Version complète avec YOLO, upload d'images, carte interactive, et Firebase
Version optimisée pour Streamlit Cloud avec tous les commentaires détaillés
"""

# ============================================================================
# IMPORTATIONS DES BIBLIOTHÈQUES
# ============================================================================

# Importation de Streamlit pour créer l'interface web interactive
import streamlit as st

# Importation de Pandas pour la manipulation des données tabulaires (DataFrames)
import pandas as pd

# Importation de NumPy pour les calculs numériques et les tableaux multidimensionnels
import numpy as np

# Importation de Folium pour créer des cartes interactives (visualisation géospatiale)
import folium

# Importation des plugins Folium pour les heatmaps (carte de chaleur) et clusters de marqueurs
from folium.plugins import HeatMap, MarkerCluster

# Importation de la fonction pour afficher les cartes Folium dans Streamlit
from streamlit_folium import folium_static

# Importation de Plotly Express pour créer des graphiques interactifs facilement
import plotly.express as px

# Importation de Plotly Graph Objects pour des graphiques plus personnalisés
import plotly.graph_objects as go

# Importation de datetime pour manipuler les dates et heures
from datetime import datetime

# Importation du module os pour les opérations système (chemins de fichiers, suppression)
import os

# Importation de tempfile pour créer des fichiers temporaires (pour l'upload d'images)
import tempfile

# Importation de PIL (Python Imaging Library) pour ouvrir et manipuler les images
from PIL import Image

# Importation du module random pour générer des décalages aléatoires de coordonnées
import random


# ============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ============================================================================

# Configuration des paramètres généraux de l'application Streamlit
st.set_page_config(
    page_title="KinSécurité IA - Surveillance Urbaine Kinshasa",  # Titre affiché dans l'onglet du navigateur
    page_icon="🚔",  # Icône affichée dans l'onglet et dans la barre latérale
    layout="wide",  # Mise en page large (utilise toute la largeur de l'écran)
    initial_sidebar_state="expanded"  # Barre latérale déployée par défaut
)


# ============================================================================
# COMMUNES DE KINSHASA (DONNÉES GÉOGRAPHIQUES)
# ============================================================================

# Dictionnaire contenant les 24 communes de Kinshasa avec leurs coordonnées GPS
# Format: "Nom_Commune": {"lat": latitude, "lon": longitude}
COMMUNES = {
    "Bandalungwa": {"lat": -4.341848, "lon": 15.283361},  # Commune de Bandalungwa
    "Barumbu": {"lat": -4.318979, "lon": 15.325618},      # Commune de Barumbu
    "Bumbu": {"lat": -4.370135, "lon": 15.294240},        # Commune de Bumbu
    "Gombe": {"lat": -4.303056, "lon": 15.303333},        # Commune de Gombe (centre-ville/quartier des affaires)
    "Kalamu": {"lat": -4.341800, "lon": 15.318700},       # Commune de Kalamu
    "Kasa-Vubu": {"lat": -4.338800, "lon": 15.303200},    # Commune de Kasa-Vubu
    "Kimbanseke": {"lat": -4.441940, "lon": 15.395000},   # Commune de Kimbanseke (périphérie Est)
    "Kinshasa": {"lat": -4.323330, "lon": 15.308060},     # Commune de Kinshasa (commune éponyme)
    "Kintambo": {"lat": -4.326983, "lon": 15.272884},     # Commune de Kintambo
    "Kisenso": {"lat": -4.409440, "lon": 15.342500},      # Commune de Kisenso
    "Lemba": {"lat": -4.405769, "lon": 15.316123},        # Commune de Lemba
    "Limete": {"lat": -4.374389, "lon": 15.345417},       # Commune de Limete
    "Lingwala": {"lat": -4.320280, "lon": 15.298330},     # Commune de Lingwala
    "Makala": {"lat": -4.379788, "lon": 15.309706},       # Commune de Makala
    "Maluku": {"lat": -4.073060, "lon": 15.537500},       # Commune de Maluku (périphérie Est lointaine)
    "Masina": {"lat": -4.383610, "lon": 15.391390},       # Commune de Masina (proche aéroport)
    "Matete": {"lat": -4.388890, "lon": 15.351670},       # Commune de Matete
    "Mont-Ngafula": {"lat": -4.455893, "lon": 15.228310}, # Commune de Mont-Ngafula (sud)
    "N'Djili": {"lat": -4.385750, "lon": 15.444569},      # Commune de N'Djili (aéroport international)
    "Ngaba": {"lat": -4.376113, "lon": 15.319617},        # Commune de Ngaba
    "Ngaliema": {"lat": -4.369733, "lon": 15.256448},     # Commune de Ngaliema
    "Ngiri-Ngiri": {"lat": -4.357500, "lon": 15.298330},  # Commune de Ngiri-Ngiri
    "N'Sele": {"lat": -4.420400, "lon": 15.494700},       # Commune de N'Sele
    "Selembao": {"lat": -4.371540, "lon": 15.284530}      # Commune de Selembao
}


# ============================================================================
# IMPORTATION DES MODULES PERSONNALISÉS
# ============================================================================

# Module de clustering DBSCAN pour détecter les zones à risque par regroupement spatial
from clustering_module import ClusteringModule

# Module de vision par ordinateur (YOLOv8 pour la détection d'objets dans les images)
from vision_module import VisionModule

# Gestionnaire Firebase pour la synchronisation des incidents avec le cloud
from firebase_manager import FirebaseManager

# Module de prétraitement des données (chargement CSV, nettoyage, normalisation)
from data_preprocessing import load_and_preprocess


# ============================================================================
# INITIALISATION DES MODULES AVEC MISE EN CACHE
# ============================================================================

@st.cache_resource  # Décorateur pour mettre en cache les ressources lourdes (évite de recharger à chaque interaction)
def init_modules():
    """
    Initialise tous les modules de l'application avec mise en cache.
    Cette fonction est appelée une seule fois au démarrage.
    
    Returns:
        tuple: (ClusteringModule, VisionModule, FirebaseManager)
    """
    # Création du module de clustering avec paramètres optimisés pour Kinshasa
    # eps (epsilon) = 0.008 : rayon de recherche en degrés (~890 mètres)
    # min_samples = 5 : nombre minimum de points pour former un cluster
    cm = ClusteringModule(eps=0.008, min_samples=5)
    
    # Initialisation du module vision avec détection réelle activée
    # use_real_detection=True utilise YOLOv8 s'il est installé
    vm = VisionModule(use_real_detection=True)
    
    # Initialisation du gestionnaire Firebase
    # mock_mode=False : tente une connexion réelle à Firebase
    fm = FirebaseManager(mock_mode=False)
    
    return cm, vm, fm

# Appel de la fonction d'initialisation et stockage des modules dans des variables globales
cm, vm, fm = init_modules()


# ============================================================================
# INITIALISATION DE L'ÉTAT DE SESSION STREAMLIT
# ============================================================================
# L'état de session (st.session_state) permet de persister des données entre les rechargements de page

# Flag indiquant si une analyse d'image a été effectuée (pour la page Vision IA)
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

# Liste des détections issues de l'analyse YOLO (objets identifiés dans l'image)
if 'detections' not in st.session_state:
    st.session_state['detections'] = []

# Commune sélectionnée par l'utilisateur lors de l'analyse d'image (valeur par défaut: Gombe)
if 'commune_choice' not in st.session_state:
    st.session_state['commune_choice'] = 'Gombe'


# ============================================================================
# AFFICHAGE DU STATUT DE CONNEXION FIREBASE
# ============================================================================

# Vérification du mode de fonctionnement (mock ou réel)
if not fm.mock_mode:
    try:
        # Tentative de récupération des incidents depuis Firebase
        fb_incidents_list = fm.get_incidents(limit=500)  # Limite à 500 incidents pour éviter la surcharge
        fb_count = len(fb_incidents_list)  # Compte le nombre d'incidents récupérés
        
        # Construction du message de statut avec le nombre d'incidents synchronisés
        firebase_status = f"☁️ Cloud: {fb_count} incidents synchronisés"
        firebase_color = "success"  # Couleur verte pour le succès
        
        # Debug - affichage des 3 premiers incidents dans la console du serveur
        print(f"Firebase: {fb_count} incidents trouvés")
        for inc in fb_incidents_list[:3]:
            print(f"   - {inc.get('commune')}: {inc.get('type_incident')}")
            
    except Exception as e:
        # En cas d'erreur de connexion (clés Firebase manquantes, réseau, etc.)
        firebase_status = f"☁️ Cloud: Erreur de connexion"
        firebase_color = "error"  # Couleur rouge pour l'erreur
        fb_incidents_list = []  # Liste vide en cas d'échec
        fb_count = 0
else:
    # Mode mock activé (utilisation de données locales simulées, pas de connexion Firebase)
    firebase_status = "☁️ Mode MOCK - Données locales uniquement"
    firebase_color = "warning"  # Couleur orange pour l'avertissement
    fb_incidents_list = []
    fb_count = 0


# ============================================================================
# CHARGEMENT DES DONNÉES (CSV + FIREBASE)
# ============================================================================

@st.cache_data  # Décorateur pour mettre en cache les données chargées (évite de recharger le CSV à chaque interaction)
def load_data():
    """
    Charge les données depuis le fichier CSV et les synchronise avec Firebase.
    Les données sont mises en cache pour optimiser les performances.
    
    Returns:
        tuple: (DataFrame contenant tous les incidents, liste des incidents Firebase)
    """
    # Chargement et prétraitement du fichier CSV contenant 30000 incidents historiques
    df = load_and_preprocess("incidents_kinshasa_30000.csv")
    
    # Renommage de la colonne 'date_incident' en 'date' pour standardisation dans tout le code
    if 'date_incident' in df.columns:
        df = df.rename(columns={'date_incident': 'date'})
    
    # Conversion de la colonne 'date' en objet datetime pour faciliter les analyses temporelles
    if 'date' in df.columns:
        # format: '28/04/2023 19:00' (jour/mois/année heure:minute)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
        # Extraction de l'heure (0-23) pour l'analyse horaire des incidents
        df['heure'] = df['date'].dt.hour
    
    # Récupération des incidents depuis Firebase (limite de 500 pour performance)
    fb_incidents = fm.get_incidents(limit=500)
    
    # Si des incidents Firebase existent, les fusionner avec les données CSV
    if fb_incidents:
        df_fb = pd.DataFrame(fb_incidents)  # Conversion en DataFrame Pandas
        print(f"{len(df_fb)} incidents chargés depuis Firebase")  # Debug dans la console
        
        # Concaténation verticale (ajout des lignes Firebase aux lignes CSV)
        if not df.empty:
            df = pd.concat([df, df_fb], ignore_index=True)
        else:
            df = df_fb  # Cas où le CSV est vide
    
    # Si aucune donnée n'est disponible (cas extrême ou première installation)
    if df.empty:
        from data_preprocessing import generate_demo_data
        df = generate_demo_data(2000)  # Génération de 2000 incidents de démonstration
    
    return df, fb_incidents

# Appel de la fonction de chargement des données
df_total, firebase_incidents = load_data()

# Vérification de sécurité : si le DataFrame est vide après tous les chargements, arrêter l'application
if df_total.empty:
    st.error("Aucune donnée disponible")  # Message d'erreur affiché à l'utilisateur
    st.stop()  # Arrêt immédiat de l'exécution


# ============================================================================
# CRÉATION DE LA BARRE LATÉRALE (SIDEBAR) - MENU DE NAVIGATION PRINCIPAL
# ============================================================================

# Titre de l'application dans la barre latérale
st.sidebar.title("🚔 KinSécurité IA")

# Affichage du nombre total d'incidents chargés (formaté avec séparateurs de milliers)
st.sidebar.markdown(f"📊 **{len(df_total):,} incidents**")

# Vérification de la disponibilité de YOLOv8 pour la détection réelle
if vm.yolo_available:
    st.sidebar.success("✅ YOLOv8 actif - Détection réelle")  # Mode réel activé
else:
    st.sidebar.warning("⚠️ YOLO non disponible - Mode simulation")  # Mode dégradé
    st.sidebar.code("pip install ultralytics", language="bash")  # Commande d'installation

# Séparateur visuel dans la barre latérale
st.sidebar.markdown("---")

# Affichage du statut Firebase avec la couleur appropriée
if firebase_color == "success":
    st.sidebar.success(firebase_status)  # Vert pour le succès
elif firebase_color == "warning":
    st.sidebar.warning(firebase_status)  # Orange pour l'avertissement
else:
    st.sidebar.error(firebase_status)  # Rouge pour l'erreur

# Séparateur visuel
st.sidebar.markdown("---")

# Widget radio pour la navigation entre les 4 pages principales de l'application
menu = st.sidebar.radio(
    "📱 Navigation",  # Titre du widget
    ["🏠 Tableau de bord", "🗺️ Carte interactive", "👁️ Vision IA", "📊 Analyses"],  # Options du menu
    key="main_menu_radio"  # Clé unique pour la persistance de l'état
)

# Séparateur visuel
st.sidebar.markdown("---")

# Texte de copyright en bas de la barre latérale
st.sidebar.caption("© 2026 KinSécurité IA")

# Date de dernière mise à jour (affichée dynamiquement)
st.sidebar.caption(f"Dernière MAJ: {datetime.now().strftime('%d/%m/%Y')}")


# ============================================================================
# FILTRES DE DONNÉES (UNIQUEMENT POUR LE TABLEAU DE BORD ET LA CARTE)
# ============================================================================

# Les filtres ne sont affichés que sur les pages Tableau de bord et Carte interactive
if menu in ["🏠 Tableau de bord", "🗺️ Carte interactive"]:
    st.sidebar.markdown("### 🔍 Filtres")  # Sous-titre pour la section filtres
    
    # ------------------------------------------------------------------------
    # FILTRE PAR COMMUNE
    # ------------------------------------------------------------------------
    if 'commune' in df_total.columns:
        # Création de la liste des communes avec l'option "Toutes" en première position
        communes_list = ["Toutes"] + sorted(df_total['commune'].dropna().unique().tolist())
        # Widget selectbox (menu déroulant) pour choisir une commune
        selected_commune = st.sidebar.selectbox("📍 Commune", communes_list, key="filter_commune")
    else:
        selected_commune = "Toutes"  # Valeur par défaut si la colonne n'existe pas
    
    # ------------------------------------------------------------------------
    # FILTRE PAR TYPE D'INCIDENT
    # ------------------------------------------------------------------------
    if 'type_incident' in df_total.columns:
        # Liste unique des types d'incidents présents dans les données
        types_list = df_total['type_incident'].dropna().unique().tolist()
        # Widget multiselect (sélection multiple) pour choisir plusieurs types
        selected_types = st.sidebar.multiselect("📋 Type d'incident", types_list, default=[], key="filter_types")
    else:
        selected_types = []  # Liste vide par défaut
    
    # ------------------------------------------------------------------------
    # FILTRE PAR GRAVITÉ
    # ------------------------------------------------------------------------
    if 'gravite' in df_total.columns:
        # Liste unique des niveaux de gravité
        gravite_list = df_total['gravite'].dropna().unique().tolist()
        # Widget multiselect pour sélectionner plusieurs niveaux de gravité
        selected_gravite = st.sidebar.multiselect("⚠️ Gravité", gravite_list, default=[], key="filter_gravite")
    else:
        selected_gravite = []  # Liste vide par défaut
    
    # ------------------------------------------------------------------------
    # APPLICATION DES FILTRES
    # ------------------------------------------------------------------------
    df_filtered = df_total.copy()  # Copie du DataFrame original pour ne pas le modifier
    
    # Application du filtre par commune (si différent de "Toutes")
    if selected_commune != "Toutes" and 'commune' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['commune'] == selected_commune]
    
    # Application du filtre par type d'incident (si des types sont sélectionnés)
    if selected_types and 'type_incident' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['type_incident'].isin(selected_types)]
    
    # Application du filtre par gravité (si des gravités sont sélectionnées)
    if selected_gravite and 'gravite' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['gravite'].isin(selected_gravite)]
        
else:
    # Pour les autres pages (Vision IA, Analyses), aucun filtre n'est appliqué
    df_filtered = df_total


# ============================================================================
# CLUSTERING DES INCIDENTS (DÉTECTION DES ZONES À RISQUE AVEC DBSCAN)
# ============================================================================

# Vérification que le DataFrame filtré n'est pas vide avant d'effectuer le clustering
if not df_filtered.empty:
    # Application de l'algorithme DBSCAN pour détecter les clusters géographiques
    # Les incidents proches spatialement sont regroupés en "zones à risque"
    df_clustered = cm.run_clustering(df_filtered)
    
    # Calcul des statistiques descriptives des clusters (taille, centre, rayon, etc.)
    cluster_stats = cm.get_cluster_statistics(df_clustered)
    
    # Évaluation de la qualité du clustering avec le score de silhouette
    # Score proche de 1 = clusters bien séparés, proche de 0 = clusters chevauchants
    clustering_metrics = cm.evaluate_model(df_clustered)
    
    # Classement des communes par nombre d'incidents (pour le tableau de bord)
    commune_ranking = cm.classify_communes(df_filtered)
    
    # Calcul de la corrélation de Spearman avec les données officielles du PNUD
    # Permet de valider la pertinence du modèle
    spearman_corr, spearman_p = cm.calculate_spearman(commune_ranking)
    
else:
    # Valeurs par défaut si le DataFrame est vide (évite les erreurs)
    cluster_stats = {'n_clusters': 0, 'n_noise': 0, 'noise_percentage': 0, 'clusters_detail': []}
    clustering_metrics = {'silhouette': 0.0, 'silhouette_interpretation': 'Non calculé'}
    commune_ranking = pd.DataFrame()  # DataFrame vide
    spearman_corr, spearman_p = 0.0, 1.0  # Corrélation nulle


# ============================================================================
# PAGE 1: TABLEAU DE BORD PRINCIPAL (ACCUEIL)
# ============================================================================

if menu == "🏠 Tableau de bord":
    # Affichage du titre principal avec emoji
    st.title("🚔 KinSécurité IA")
    
    # Sous-titre descriptif de l'application
    st.markdown("*Système de surveillance urbaine assisté par IA pour Kinshasa*")
    
    # Séparateur visuel (ligne horizontale)
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # INDICATEURS CLÉS DE PERFORMANCE (KPI)
    # ------------------------------------------------------------------------
    # Création de 4 colonnes de largeur égale pour les métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Nombre total d'incidents (formaté avec séparateur de milliers)
        st.metric("📊 Incidents", f"{len(df_filtered):,}")
    
    with col2:
        # Nombre de communes touchées (valeurs uniques dans la colonne 'commune')
        unique_communes = df_filtered['commune'].nunique() if 'commune' in df_filtered.columns else 0
        st.metric("📍 Communes touchées", unique_communes)
    
    with col3:
        # Nombre de types d'incidents différents (agressions, vols, etc.)
        types_count = df_filtered['type_incident'].nunique() if 'type_incident' in df_filtered.columns else 0
        st.metric("📋 Types d'incidents", types_count)
    
    with col4:
        # Nombre de zones à risque détectées par l'algorithme DBSCAN
        st.metric("🎯 Zones à risque", cluster_stats['n_clusters'])
    
    # Séparateur visuel
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # GRAPHIQUES PRINCIPAUX (2 COLONNES)
    # ------------------------------------------------------------------------
    col_left, col_right = st.columns([2, 1])  # Colonne gauche 2x plus large que colonne droite
    
    # -----------------------------------------
    # COLONNE DROITE : CLASSEMENT DES COMMUNES
    # -----------------------------------------
    with col_right:
        st.subheader("📊 Classement des communes")
        
        # Vérification que le classement des communes n'est pas vide
        if not commune_ranking.empty:
            # Sélection des 15 communes les plus touchées
            top_communes = commune_ranking.head(15).copy()
            
            # Création d'un graphique à barres horizontales avec Plotly
            fig = px.bar(
                top_communes,
                x='count',  # Abscisse : nombre d'incidents
                y='commune',  # Ordonnée : nom des communes
                orientation='h',  # Barres horizontales (plus lisibles pour les longues listes)
                title="Nombre d'incidents par commune",
                labels={'count': "Nombre d'incidents", 'commune': "Commune"},
                color='count',  # Coloration basée sur le nombre d'incidents
                color_continuous_scale='Reds',  # Dégradé de rouge (plus rouge = plus d'incidents)
                text='count'  # Affichage des valeurs sur les barres
            )
            
            # Ajustement de la hauteur et de l'ordre des communes (croissant pour que la plus touchée soit en haut)
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            
            # Positionnement des étiquettes (valeurs numériques) à l'extérieur des barres
            fig.update_traces(textposition='outside')
            
            # Affichage du graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcul du pourcentage d'incidents concentrés dans les 5 premières communes
            total_top5 = top_communes.head(5)['count'].sum()
            total_all = commune_ranking['count'].sum()
            
            # Légende explicative pour l'utilisateur
            st.caption(f"📌 Les 5 communes les plus touchées représentent {total_top5/total_all*100:.1f}% des incidents")
        else:
            st.info("Données insuffisantes pour le classement")
    
    # -----------------------------------------
    # COLONNE GAUCHE : ÉVOLUTION MENSUELLE
    # -----------------------------------------
    with col_left:
        st.subheader("📊 Évolution mensuelle des incidents")
        
        # Identification de la colonne contenant les dates (deux formats possibles)
        date_col = None
        if 'date' in df_filtered.columns:
            date_col = 'date'  # Format standardisé après renommage
        elif 'date_incident' in df_filtered.columns:
            date_col = 'date_incident'  # Format original du CSV
        
        if date_col:
            try:
                # Copie du DataFrame pour éviter les modifications accidentelles
                df_temp = df_filtered.copy()
                
                # Conversion de la colonne date en datetime (gère les erreurs avec coerce = NaN)
                df_temp['date_conv'] = pd.to_datetime(df_temp[date_col], format='%d/%m/%Y %H:%M', errors='coerce')
                
                # Suppression des lignes avec des dates invalides (NaN)
                df_temp = df_temp.dropna(subset=['date_conv'])
                
                # Vérification que des données valides existent après conversion
                if not df_temp.empty:
                    # Extraction du mois et de l'année (exemple: "2024-01" pour janvier 2024)
                    df_temp['mois_annee'] = df_temp['date_conv'].dt.to_period('M').astype(str)
                    
                    # Agrégation : comptage du nombre d'incidents par mois
                    monthly = df_temp.groupby('mois_annee').size().reset_index(name='count')
                    
                    # Tri par ordre chronologique pour que le graphique ait un sens
                    monthly = monthly.sort_values('mois_annee')
                    
                    # Création du graphique à barres verticales
                    fig = px.bar(
                        monthly,
                        x='mois_annee',
                        y='count',
                        title="Nombre d'incidents par mois",
                        labels={'mois_annee': 'Mois', 'count': "Nombre d'incidents"},
                        color='count',
                        color_continuous_scale='Reds'
                    )
                    
                    # Rotation des étiquettes de l'axe X (45 degrés) pour améliorer la lisibilité
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    
                    # Affichage du graphique
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Identification et affichage du mois record (maximum d'incidents)
                    max_row = monthly.loc[monthly['count'].idxmax()]
                    st.caption(f"📊 Maximum: {int(max_row['count'])} incidents en {max_row['mois_annee']}")
                else:
                    st.error("Aucune date n'a pu être convertie")
                    
            except Exception as e:
                st.error(f"Erreur lors du traitement des dates: {e}")
        else:
            st.info("Colonne de date non disponible dans les données")
    
    # Séparateur visuel
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # DERNIERS INCIDENTS SIGNALÉS (TABLEAU RÉCENT)
    # ------------------------------------------------------------------------
    st.subheader("🕐 Derniers incidents signalés")
    
    if 'date' in df_filtered.columns:
        # Tri par date décroissante (du plus récent au plus ancien) et sélection des 10 premiers
        latest = df_filtered.sort_values('date', ascending=False).head(10)
        
        # Définition des colonnes à afficher dans le tableau
        cols_to_display = ['date', 'commune', 'type_incident']
        if 'gravite' in latest.columns:
            cols_to_display.append('gravite')  # Ajout de la gravité si disponible
        
        # Affichage du tableau avec utilisation de toute la largeur disponible
        st.dataframe(latest[cols_to_display], use_container_width=True)
        
    elif 'date_incident' in df_filtered.columns:
        # Version alternative avec le nom de colonne original
        latest = df_filtered.sort_values('date_incident', ascending=False).head(10)
        st.dataframe(latest[['date_incident', 'commune', 'type_incident']], use_container_width=True)


# ============================================================================
# PAGE 2: CARTE INTERACTIVE DES INCIDENTS
# ============================================================================

elif menu == "🗺️ Carte interactive":
    # Titre de la page
    st.title("🗺️ Carte interactive des incidents")
    st.markdown("*Visualisation spatiale des incidents à Kinshasa*")
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # VÉRIFICATION DES DONNÉES GÉOGRAPHIQUES
    # ------------------------------------------------------------------------
    # Vérification que le DataFrame contient des coordonnées valides
    if df_filtered.empty or df_filtered[['latitude', 'longitude']].dropna().empty:
        st.error("❌ Aucune donnée géographique valide à afficher")
    else:
        # Comptage des incidents avec coordonnées valides
        valid_count = df_filtered[['latitude', 'longitude']].dropna().shape[0]
        st.info(f"📍 **{len(df_filtered):,} incidents** au total - **{valid_count:,}** avec coordonnées valides")
        
        try:
            # --------------------------------------------------------------------
            # CRÉATION DE LA CARTE DE BASE (FOLIUM)
            # --------------------------------------------------------------------
            # Création d'une carte Folium centrée sur Kinshasa
            # tiles='CartoDB positron' : fond de carte clair et élégant
            m = folium.Map(
                location=[-4.35, 15.30],  # Coordonnées du centre de Kinshasa
                zoom_start=11,            # Niveau de zoom (11 = vue de la ville entière)
                control_scale=True,       # Affichage d'une échelle en mètres
                tiles='CartoDB positron'  # Style de carte
            )
            
            # --------------------------------------------------------------------
            # HEATMAP (CARTE DE CHALEUR) - ZONES À FORTE CONCENTRATION
            # --------------------------------------------------------------------
            # Extraction des coordonnées pour la heatmap
            heat_data = df_filtered[['latitude', 'longitude']].dropna().values.tolist()
            
            # Ajout de la heatmap seulement si assez de points (minimum 10)
            if heat_data and len(heat_data) > 10:
                HeatMap(
                    heat_data,
                    radius=15,        # Rayon d'influence de chaque point
                    blur=10,          # Niveau de flou (estompage entre les zones)
                    min_opacity=0.4,  # Opacité minimale (évite la transparence totale)
                    gradient={        # Dégradé de couleurs pour représenter l'intensité
                        0.2: 'blue',    # Bleu pour faible concentration
                        0.4: 'lime',    # Vert clair pour concentration modérée
                        0.6: 'yellow',  # Jaune pour concentration élevée
                        0.8: 'orange',  # Orange pour très élevée
                        1: 'red'        # Rouge pour concentration maximale
                    }
                ).add_to(m)
            
            # --------------------------------------------------------------------
            # MARQUEURS DES INCIDENTS (AVEC CLUSTER POUR PERFORMANCE)
            # --------------------------------------------------------------------
            # Création d'un cluster de marqueurs pour regrouper les incidents proches
            # Améliore les performances et la lisibilité de la carte
            marker_cluster = MarkerCluster(name="Incidents").add_to(m)
            markers_count = 0  # Compteur de marqueurs ajoutés
            
            # Limitation à 1500 points pour éviter de surcharger l'affichage
            sample_size = min(1500, len(df_filtered))
            sample_df = df_filtered.head(sample_size)
            
            # Parcours de chaque incident pour l'ajouter sur la carte
            for _, row in sample_df.iterrows():
                # Vérification que les coordonnées sont valides (non NaN)
                if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                    continue
                
                # Détermination de la couleur selon la gravité de l'incident
                gravite = str(row.get('gravite', 'Moyenne'))
                if 'Critique' in gravite:
                    color = '#dc2626'  # Rouge vif pour critique
                    size = 5
                elif 'Élevée' in gravite:
                    color = '#f97316'  # Orange pour élevé
                    size = 5
                elif 'Moyenne' in gravite:
                    color = '#eab308'  # Jaune pour moyen
                    size = 5
                else:
                    color = '#22c55e'  # Vert pour faible
                    size = 5
                
                # Construction du HTML pour la popup (info-bulle au clic)
                popup_html = f"""
                <div style="font-family: monospace; font-size: 12px; min-width: 150px;">
                    <b>🚨 {row.get('type_incident', 'N/A')}</b><br>
                    📍 Commune: {row.get('commune', 'N/A')}<br>
                    ⚠️ Gravité: {gravite}<br>
                    📅 Date: {str(row.get('date', 'N/A'))[:16]}
                </div>
                """
                
                # Ajout d'un cercle marqueur (plus esthétique qu'un simple point)
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(marker_cluster)
                
                markers_count += 1
            
            # --------------------------------------------------------------------
            # CERCLES DE CLUSTERS DBSCAN (ZONES À RISQUE DÉTECTÉES)
            # --------------------------------------------------------------------
            # Ajout des cercles représentant les zones à risque détectées par DBSCAN
            if cluster_stats and cluster_stats.get('clusters_detail'):
                try:
                    # Palette de couleurs pour différencier les clusters
                    colors = ['#dc2626', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']
                    
                    # Affichage des 10 premiers clusters maximum
                    for idx, cluster in enumerate(cluster_stats['clusters_detail'][:10]):
                        color = colors[idx % len(colors)]  # Cycle dans les couleurs disponibles
                        
                        # Ajout d'un cercle centré sur le centroïde du cluster
                        folium.Circle(
                            location=[cluster['center_lat'], cluster['center_lon']],
                            radius=cluster['radius_km'] * 1000,  # Conversion km -> mètres
                            color=color,
                            fill=True,
                            fill_opacity=0.1,  # Opacité faible pour ne pas cacher les marqueurs
                            weight=2,
                            popup=f"🎯 Cluster #{cluster['cluster_id']}<br>📊 {cluster['size']} incidents"
                        ).add_to(m)
                except Exception as e:
                    pass  # Ignorer les erreurs d'affichage des clusters
            
            # --------------------------------------------------------------------
            # LÉGENDE PERSONNALISÉE
            # --------------------------------------------------------------------
            # Création d'une légende HTML positionnée en bas à droite de la carte
            legend_html = '''
            <div style="position: fixed; bottom: 50px; right: 50px; background: white; padding: 10px; border-radius: 8px; border: 1px solid #ccc; z-index:1000; font-size: 12px;">
                <b>📖 LÉGENDE</b><br>
                <span style="color:#dc2626;">●</span> Critique<br>
                <span style="color:#f97316;">●</span> Élevé<br>
                <span style="color:#eab308;">●</span> Moyen<br>
                <span style="color:#22c55e;">●</span> Faible<br>
                <span style="color:#888;">🌀</span> Heatmap (carte de chaleur)
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            # --------------------------------------------------------------------
            # CONTRÔLE DES COUCHES ET AFFICHAGE
            # --------------------------------------------------------------------
            # Ajout du contrôle des couches pour activer/désactiver les différents éléments
            folium.LayerControl().add_to(m)
            
            # Affichage de la carte dans Streamlit
            folium_static(m, width=900, height=600)
            
            # Information sur le nombre de marqueurs affichés
            st.caption(f"📌 **{markers_count}** marqueurs affichés sur la carte")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la création de la carte: {e}")


# ============================================================================
# PAGE 3: VISION PAR ORDINATEUR (DÉTECTION YOLO SUR IMAGES UPLOADÉES)
# ============================================================================

elif menu == "👁️ Vision IA":
    st.title("👁️ Vision par Ordinateur - Détection d'incidents")
    st.markdown("*Analysez des images avec intelligence artificielle pour détecter des situations suspectes*")
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION D'INFORMATION (DÉPLIABLE)
    # ------------------------------------------------------------------------
    with st.expander("ℹ️ Comment ça fonctionne", expanded=True):
        st.markdown("""
        **1. Téléchargez une photo** prise sur le terrain (incident, situation suspecte, attroupement, etc.)
        
        **2. L'IA analyse l'image** avec YOLOv8 (You Only Look Once) pour détecter:
        - Personnes (individus, foules)
        - Véhicules suspects
        - Armes (couteaux, armes à feu)
        - Incendies, fumées
        - Attroupements
        
        **3. Résultats instantanés** avec niveau de confiance et alerte si nécessaire
        
        **4. Créez un incident** à partir de l'analyse pour l'ajouter à la base de données
        """)
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION UPLOAD D'IMAGE
    # ------------------------------------------------------------------------
    st.subheader("📸 Téléchargez une image à analyser")
    
    # Widget de téléchargement de fichier (images uniquement)
    uploaded_file = st.file_uploader(
        "Choisir une image (JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Formats acceptés: JPG, JPEG, PNG. Taille maximale: 10MB",
        key="vision_upload_main"
    )
    
    # Traitement si un fichier a été téléchargé
    if uploaded_file is not None:
        # Ouverture de l'image avec PIL pour l'affichage
        image = Image.open(uploaded_file)
        
        # Création de deux colonnes : image à gauche, informations à droite
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="📷 Image à analyser", width=400)
        
        with col_info:
            # Affichage des métadonnées du fichier
            st.markdown(f"**Fichier:** {uploaded_file.name}")
            st.markdown(f"**Taille:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
            
            # Sélection de la commune où la photo a été prise
            commune_choice = st.selectbox(
                "📍 Sélectionner la commune de l'incident",
                list(COMMUNES.keys()),
                key="vision_commune"
            )
            
            # Bouton principal pour lancer l'analyse YOLO
            if st.button("🚀 ANALYSER L'IMAGE AVEC YOLO", type="primary", use_container_width=True, key="analyze_btn_main"):
                
                # Création d'un fichier temporaire pour stocker l'image
                # delete=False car nous devons y accéder après la fermeture du contexte
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(uploaded_file.getvalue())  # Écriture du contenu binaire
                    tmp_path = tmp.name  # Récupération du chemin absolu
                
                # Lancement de la détection avec le module VisionModule
                with st.spinner("🔍 YOLOv8 analyse l'image en cours..."):
                    detections = vm.detect_from_image(tmp_path)
                
                # Suppression du fichier temporaire pour libérer l'espace disque
                os.unlink(tmp_path)
                
                # Stockage des résultats dans l'état de session pour persistance
                st.session_state['detections'] = detections
                st.session_state['commune_choice'] = commune_choice
                st.session_state['analysis_done'] = True
                
                # Rechargement de la page pour afficher les résultats
                st.rerun()
    
    # ------------------------------------------------------------------------
    # AFFICHAGE DES RÉSULTATS DE L'ANALYSE
    # ------------------------------------------------------------------------
    if st.session_state.get('analysis_done', False):
        detections = st.session_state.get('detections', [])
        commune_choice = st.session_state.get('commune_choice', 'Gombe')
        
        st.markdown("---")
        st.subheader("📊 RÉSULTATS DE L'ANALYSE")
        
        # Vérification que des objets ont été détectés
        if detections:
            # Filtrage des alertes (objets dangereux ou suspects)
            alerts = [d for d in detections if d.get('alert', False)]
            
            if alerts:
                # Affichage des alertes en rouge avec animation de ballons
                for alert in alerts:
                    st.error(f"🚨 **ALERTE {alert.get('severity', 'CRITIQUE')}** : {alert['label']} (confiance: {alert.get('confidence_pct', 'N/A')})")
                st.balloons()  # Animation festive pour attirer l'attention
            
            # Affichage détaillé de tous les objets détectés
            st.markdown("**📋 Objets détectés:**")
            for d in detections:
                if not d.get('alert', False):
                    st.write(f"- **{d['label']}** : {d.get('confidence_pct', 'N/A')} de confiance")
            
            # Indication du mode d'analyse (réel ou simulé)
            if vm.yolo_available:
                st.success("✅ Analyse réalisée avec YOLOv8 (détection réelle)")
            else:
                st.info("ℹ️ Mode simulation - YOLO non installé")
            
            # --------------------------------------------------------------------
            # FORMULAIRE DE CRÉATION D'INCIDENT
            # --------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📝 Créer un incident à partir de cette analyse")
            
            # Sélection de la détection à transformer en incident officiel
            detection_options = [f"{d['label']} ({d.get('confidence_pct', 'N/A')})" for d in detections]
            selected_idx = st.selectbox(
                "Sélectionner la détection à convertir en incident",
                range(len(detection_options)),
                format_func=lambda x: detection_options[x],
                key="detection_select"
            )
            selected_detection = detections[selected_idx]
            
            # Résumé de l'incident à créer
            st.info(f"📌 **Incident à créer:** {selected_detection['label']} (confiance: {selected_detection.get('confidence_pct', 'N/A')})")
            
            # Bouton de validation finale
            if st.button("✅ VALIDER ET CRÉER L'INCIDENT", type="primary", use_container_width=True, key="final_create_btn"):
                
                # Récupération des coordonnées de la commune sélectionnée
                commune_name = commune_choice
                commune_coords = COMMUNES[commune_name]
                
                # Ajout d'un décalage aléatoire pour une meilleure dispersion spatiale
                # Évite que tous les incidents créés soient pile au centre de la commune
                lat = commune_coords["lat"] + random.uniform(-0.003, 0.003)
                lon = commune_coords["lon"] + random.uniform(-0.003, 0.003)
                
                # Classification automatique basée sur le type d'objet détecté
                detection_label = selected_detection.get('label', 'Inconnu').lower()
                
                # Règles de mapping entre objets YOLO et types d'incidents
                if 'attroupement' in detection_label:
                    incident_type = "Attroupement suspect"
                    gravite = "Moyenne"
                elif 'arme' in detection_label or 'couteau' in detection_label:
                    incident_type = "Port d'arme"
                    gravite = "Critique"
                elif 'bagarre' in detection_label or 'altercation' in detection_label:
                    incident_type = "Violence"
                    gravite = "Élevée"
                elif 'incendie' in detection_label or 'feu' in detection_label:
                    incident_type = "Incendie"
                    gravite = "Critique"
                else:
                    incident_type = selected_detection.get('label', 'Incident signalé')
                    gravite = "Moyenne"
                
                # Construction de l'objet incident au format attendu par Firebase
                incident = {
                    "commune": commune_name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type_incident": incident_type,
                    "gravite": gravite,
                    "source": "Vision IA (Interface utilisateur)",
                    "confidence": selected_detection.get("confidence", 0.8),
                    "original_detection": selected_detection.get("label", "Inconnu"),
                    "firebase_timestamp": datetime.now().isoformat(),
                    "status": "verified"
                }
                
                # Debug : affichage des données envoyées (visible par le développeur)
                st.write("**🔍 Incident envoyé à Firebase:**")
                st.json(incident)
                
                # Envoi à Firebase via le gestionnaire
                result = fm.push_incident(incident)
                st.write(f"**📡 Résultat de l'envoi:** {'✅ Réussi' if result else '❌ Échoué'}")
                
                if result:
                    st.success(f"✅ Incident créé pour {commune_name} !")
                    st.balloons()
                    
                    # Invalidation du cache pour forcer le rechargement des données dans les autres pages
                    st.cache_data.clear()
                    
                    # Proposition de nouvelle analyse
                    if st.button("🔄 ANALYSER UNE NOUVELLE IMAGE", key="new_analysis_btn"):
                        st.session_state['analysis_done'] = False
                        st.session_state['detections'] = []
                        st.rerun()
                else:
                    st.error("❌ Erreur lors de la création de l'incident")
        
        else:
            # Cas où aucun objet n'a été détecté par YOLO
            st.warning("⚠️ Aucun objet pertinent détecté dans cette image")
            
            if st.button("🔄 RÉESSAYER AVEC UNE AUTRE IMAGE", key="retry_btn"):
                st.session_state['analysis_done'] = False
                st.rerun()
    
    else:
        # Message affiché avant tout téléchargement d'image
        st.info("👆 **Téléchargez une image ci-dessus pour commencer l'analyse**")
        
        # Exemples d'images suggérées (section dépliable)
        with st.expander("📖 Exemples d'images à analyser", expanded=False):
            st.markdown("""
            - Photo d'un **attroupement** dans un espace public
            - Image d'un **véhicule suspect** stationné
            - Photo d'une **altercation** ou bagarre
            - Image d'un **objet abandonné** suspect
            - Photo d'un **incendie** ou fumée
            """)
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION D'INFORMATION SUR YOLOV8 (DÉPLIABLE)
    # ------------------------------------------------------------------------
    with st.expander("⚙️ Configuration YOLOv8"):
        if vm.yolo_available:
            st.success("✅ YOLOv8 est installé et fonctionnel")
            st.code("""
            Modèle: YOLOv8 nano (yolov8n.pt)
            Classes détectées: 80 (personnes, véhicules, armes, etc.)
            Temps d'inférence: ~30ms par image sur GPU
            """)
        else:
            st.warning("⚠️ YOLOv8 n'est pas installé")
            st.code("""
            Pour installer YOLOv8 et activer la détection réelle:
            
            pip install ultralytics
            pip install opencv-python
            pip install pillow
            """)
            st.info("Après installation, redémarrez l'application")


# ============================================================================
# PAGE 4: ANALYSES STATISTIQUES DÉTAILLÉES
# ============================================================================

elif menu == "📊 Analyses":
    st.title("📊 Analyses statistiques")
    st.markdown("*Indicateurs clés et tendances de la sécurité à Kinshasa*")
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION 1: RÉSULTATS DU CLUSTERING DBSCAN
    # ------------------------------------------------------------------------
    st.subheader("🎯 Analyse spatiale - Clustering DBSCAN")
    
    # Affichage des 4 métriques principales du clustering dans des colonnes
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    with col_c1:
        st.metric("Clusters détectés", cluster_stats['n_clusters'])
    
    with col_c2:
        st.metric("Points isolés (bruit)", cluster_stats['n_noise'])
    
    with col_c3:
        st.metric("Taux de bruit", f"{cluster_stats['noise_percentage']:.1f}%")
    
    with col_c4:
        st.metric("Score silhouette", f"{clustering_metrics['silhouette']:.3f}")
        st.caption(clustering_metrics['silhouette_interpretation'])
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION 2: DÉTAIL DES CLUSTERS DÉTECTÉS
    # ------------------------------------------------------------------------
    if cluster_stats['clusters_detail']:
        st.subheader("📋 Détail des clusters détectés")
        
        # Affichage des 5 premiers clusters dans des expanders (sections dépliables)
        for cluster in cluster_stats['clusters_detail'][:5]:
            with st.expander(f"🎯 Cluster #{cluster['cluster_id']} - {cluster['size']} incidents ({cluster['percentage']:.1f}%)"):
                st.write(f"- **Centre géographique:** {cluster['center_lat']:.5f}, {cluster['center_lon']:.5f}")
                st.write(f"- **Rayon approximatif:** {cluster['radius_km']:.2f} km")
                st.write(f"- **Type d'incident principal:** {cluster['main_incident_type']}")
                st.write(f"- **Gravité dominante:** {cluster['main_gravite']}")
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION 3: CLASSEMENT DES COMMUNES
    # ------------------------------------------------------------------------
    st.subheader("🏆 Classement des communes")
    
    if not commune_ranking.empty:
        # Graphique à barres horizontales du classement des communes
        fig = px.bar(
            commune_ranking.head(15),
            x='commune',
            y='incidents_mensuels',
            color='niveau_insecurite',
            color_discrete_map={
                'Très élevé': '#dc2626',
                'Élevé': '#f97316',
                'Modéré': '#eab308',
                'Faible': '#22c55e'
            },
            title="Incidents mensuels par commune"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION 4: VALIDATION DU MODÈLE (CORRÉLATION DE SPEARMAN)
    # ------------------------------------------------------------------------
    st.subheader("📊 Validation du modèle")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.metric("Corrélation de Spearman", f"{spearman_corr:.3f}")
        st.caption(f"p-value: {spearman_p:.4f}")
    
    with col_v2:
        if spearman_corr > 0.5:
            st.success("✅ Forte corrélation avec les risques officiels (PNUD)")
        elif spearman_corr > 0.3:
            st.info("⚠️ Corrélation modérée avec les risques officiels")
        else:
            st.warning("❌ Faible corrélation - données à enrichir")
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # SECTION 5: MATRICE DES TYPES D'INCIDENTS PAR COMMUNE
    # ------------------------------------------------------------------------
    if 'type_incident' in df_filtered.columns and 'commune' in df_filtered.columns:
        st.subheader("📊 Distribution des types d'incidents par commune")
        
        # Création d'un tableau croisé (communes en lignes, types en colonnes)
        pivot = pd.crosstab(df_filtered['commune'], df_filtered['type_incident'])
        st.dataframe(pivot, use_container_width=True)
    
    # ------------------------------------------------------------------------
    # SECTION 6: DIAGNOSTIC TECHNIQUE (DÉPLIABLE)
    # ------------------------------------------------------------------------
    with st.expander("🔍 Diagnostic des données"):
        st.write("**Colonnes disponibles dans le DataFrame:**")
        st.write(df_filtered.columns.tolist())
        
        st.write("**Aperçu des données (5 premières lignes):**")
        st.dataframe(df_filtered.head())
    
    # ------------------------------------------------------------------------
    # SECTION 7: DEBUG FIREBASE (DÉPLIABLE)
    # ------------------------------------------------------------------------
    with st.expander("🔧 DEBUG - Incidents Firebase"):
        if st.button("Afficher les incidents Firebase", key="debug_firebase_btn"):
            fb_incidents = fm.get_incidents(limit=20)
            if fb_incidents:
                st.write(f"**{len(fb_incidents)} incidents dans Firebase:**")
                for inc in fb_incidents[:10]:  # Limite à 10 pour lisibilité
                    st.write(f"- {inc.get('commune')}: {inc.get('type_incident')} (source: {inc.get('source', 'inconnue')})")
            else:
                st.write("Aucun incident trouvé dans Firebase")


# ============================================================================
# FIN DE L'APPLICATION
# ============================================================================
# Le code se termine ici. Les différentes pages sont gérées par la structure if/elif/else
# basée sur la sélection du menu dans la barre latérale.
