### Modules du projet KinSécurité IA
## Voici la liste complète et organisée des modules de votre projet :
```
Architecture du projet
text
kinsecurite-ia/
│
├── 📄 app.py                          # Application principale Streamlit (interface utilisateur)
│
├── 📄 clustering_module.py            # Module de clustering spatial DBSCAN
│
├── 📄 vision_module.py                # Module de vision par ordinateur (YOLOv8)
│
├── 📄 firebase_manager.py             # Module de gestion Firebase (cloud)
│
├── 📄 data_preprocessing.py           # Module de prétraitement des données
│
├── 📄 reporting_module.py             # Module de signalement citoyen
│
├── 📄 requirements.txt                # Dépendances Python
│
├── 📄 incidents_kinshasa_30000.csv    # Base de données des incidents (30k lignes)
│
└── 📄 serviceAccountKey.json          # Clé d'authentification Firebase
```
### Description détaillée de chaque module

## 1. app.py - Application principale
```
Rôle : Interface utilisateur Streamlit (Frontend)
Fonctions principales :

   1. Affichage du tableau de bord avec métriques

   2. Carte interactive Folium (heatmap + clusters)

   3. Interface d'upload et analyse d'images (Vision IA)

   4. Page d'analyses statistiques

   5. Gestion des filtres (commune, type, gravité)

   6. Affichage du statut Firebase

   7. Gestion de session state

   8. Dépendances : tous les autres modules
```
## 2. clustering_module.py - Clustering spatial
### Rôle : Analyse de densité avec DBSCAN
### Classes/Méthodes principales :

# ClusteringModule : classe principale
# run_clustering() : exécute DBSCAN sur les coordonnées
# get_cluster_statistics() : statistiques détaillées des clusters
# evaluate_model() : calcul du score de silhouette
# classify_communes() : classification des communes par risque
# calculate_spearman() : corrélation avec risques officiels
# get_cluster_geometries() : calcul des centres/rayons
# get_zones_by_commune() : scores de risque par commune
# get_hotspots() : identification des points chauds

### Paramètres DBSCAN :
# eps = 0.008 (≈ 888 mètres)
# min_samples = 5

## 3. vision_module.py - Vision par ordinateur
### Rôle : Détection d'objets avec YOLOv8
### Classes/Méthodes principales :

# VisionModule : classe principale
# _load_yolo_model() : chargement du modèle YOLOv8
# detect_from_image() : détection sur image
# _analyze_uploaded_file() : analyse d'image uploadée
# _simulate_detection() : simulation si YOLO indisponible
# generate_incident_from_detection() : conversion détection → incident
# _extract_gps_from_file() : extraction métadonnées GPS
# Classes détectées :
## Personnes, véhicules, armes, incendies, attroupements

## 4. firebase_manager.py - Gestion Firebase
### Rôle : Synchronisation cloud des données
### Classes/Méthodes principales :
# FirebaseManager : classe principale
# _connect_firebase() : connexion à Firestore
# push_incident() : envoi d'un incident
# get_incidents() : récupération des incidents
# push_to_collection() : envoi vers collection spécifique
# get_from_collection() : récupération depuis collection
# update_document() : mise à jour d'un document
# get_stats() : statistiques Firebase
# Collections :
## incidents : incidents standards
## citizen_reports : signalements citoyens

## Modes :
# Mode REEL : connexion à Firebase Cloud
# Mode MOCK : stockage local (fallback)

## 5. data_preprocessing.py - Prétraitement
### Rôle : Nettoyage et préparation des données
## Fonctions principales :
# load_and_preprocess() : chargement + nettoyage CSV
# generate_demo_data() : génération de données de démonstration
# get_commune_coordinates() : dictionnaire des communes
### Nettoyages effectués :
## Standardisation des colonnes
## Correction des encodages (latin1 → utf-8)
## Filtrage des coordonnées hors Kinshasa
## Parsing des dates
## Catégorisation des sources

## 6. reporting_module.py - Signalement citoyen
### Rôle : Gestion des signalements par les citoyens
### Classes/Méthodes principales :
# CitizenReportingModule : classe principale
# create_report_form() : formulaire de signalement
# display_reports_list() : affichage des signalements
# _calculate_trust_score() : score de fiabilité (0-100)
# _save_images() : sauvegarde des preuves visuelles
## AdminVerificationModule : classe d'administration
# admin_panel() : panneau de vérification
# add_reports_to_map() : ajout des signalements sur la carte
## Score de confiance (0-100) :
# Coordonnées valides : +20
# Présence d'images : +20
# Métadonnées GPS : +25
# Détections YOLO : +15
# Description détaillée : +10-15
# Contact fourni : +10

## Flux de données entre modules

┌─────────────────────────────────────────────────────────────────────────┐
│                           FLUX DE DONNÉES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   incidents_kinshasa_30000.csv                                          │
│           │                                                             │
│           ▼                                                             │
│   ┌────────────────────┐                                                │
│   │ data_preprocessing │ → Données nettoyées                            │
│   └─────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│   ┌───────────────────┐     ┌─────────────────┐                         │
│   │ clustering_module │ ←   │    app.py       │                         │
│   └─────────┬─────────┘     │  (interface)    │                         │
│             │               └─────────┬───────┘                         │
│             ▼                         │                                 │
│   Résultats clustering                │                                 │
│   (clusters, risques)                 │                                 │
│                                       │                                 │
│                    ┌──────────────────┼──────────────────┐              │
│                    │                  │                  │              │
│                    ▼                  ▼                  ▼              │
│            ┌─────────────┐    ┌─────────────┐    ┌────────────────┐     │
│            │firebase_    │    │vision_      │    │reporting_module│     │
│            │manager      │    │module       │    │                │     │
│            └──────┬──────┘    └──────┬──────┘    └──────┬─────────┘     │
│                   │                  │                  │               │
│                   ▼                  ▼                  ▼               │
│            Firebase Cloud      Détections YOLO    Signalements          │
│            (stockage)          (images → objet)   (citoyens)            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
## Résumé des responsabilités

Module	                               Responsabilité
app.py	                      Interface utilisateur, orchestration
clustering_module.py	          Analyse spatiale, détection zones à risque
vision_module.py	             Analyse d'images, détection d'objets
firebase_manager.py	          Stockage cloud, synchronisation
data_preprocessing.py	       Nettoyage et préparation des données
reporting_module.py	          Signalements citoyens, vérification

## Technologies utilisées

Technologie	           Module                      	Usage
Streamlit	           app.py	                    Interface web
Folium	              app.py	                    Cartes interactives
Plotly	              app.py	                    Graphiques
Pandas	              Tous	                    Manipulation données
NumPy	                 clustering, vision	        Calculs numériques
Scikit-learn	        clustering_module	        DBSCAN, métriques
YOLOv8 (Ultralytics)	  vision_module	           Détection d'objets
Firebase Admin	        firebase_manager	        Cloud Firestore
OpenCV	              vision_module	           Traitement d'images
PIL/Pillow	           vision_module	           Manipulation d'images
