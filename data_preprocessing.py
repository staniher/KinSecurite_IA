"""
MODULE DE PRÉTRAITEMENT DES DONNÉES
Version robuste avec gestion des erreurs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_preprocess(filepath="incidents_kinshasa_30000.csv"):
    """
    Charge et nettoie le fichier CSV des incidents
    Version avec fallback si fichier non trouvé
    """
    # Vérifie si le fichier existe
    if not os.path.exists(filepath):
        print(f"Fichier {filepath} non trouvé")
        print(f"Répertoire courant: {os.getcwd()}")
        print(f"Fichiers disponibles: {os.listdir('.')}")
        
        # Génère des données de démonstration
        print("Génération de données de démonstration...")
        return generate_demo_data()
    
    try:
        # Essaye différents séparateurs et encodages
        df = None
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"Fichier chargé avec encoding {encoding}: {len(df)} lignes")
                break
            except:
                continue
        
        if df is None:
            raise Exception("Impossible de lire le fichier")
        
        # Nettoyage des colonnes
        df.columns = df.columns.str.strip().str.lower()
        
        # Vérification des colonnes nécessaires
        required_cols = ['latitude', 'longitude', 'commune']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Colonnes manquantes: {missing_cols}")
            print(f"Colonnes disponibles: {df.columns.tolist()}")
            return generate_demo_data()
        
        # Nettoyage des valeurs
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Filtrage des coordonnées valides
        valid_lat = (df['latitude'] >= -4.60) & (df['latitude'] <= -4.00)
        valid_lon = (df['longitude'] >= 15.00) & (df['longitude'] <= 15.70)
        df = df[valid_lat & valid_lon]
        
        if len(df) == 0:
            print("Aucune donnée valide après filtrage")
            return generate_demo_data()
        
        print(f"Données après nettoyage: {len(df)} lignes")
        return df
        
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return generate_demo_data()


def generate_demo_data(n_incidents=500):
    """
    Génère des données de démonstration réalistes
    """
    print(f"Génération de {n_incidents} incidents de démonstration...")
    
    # Communes de Kinshasa
    communes = {
        "Gombe": {"lat": -4.303056, "lon": 15.303333},
        "Limete": {"lat": -4.374389, "lon": 15.345417},
        "Masina": {"lat": -4.383610, "lon": 15.391390},
        "Ngaliema": {"lat": -4.369733, "lon": 15.256448},
        "Kalamu": {"lat": -4.341800, "lon": 15.318700},
        "Lemba": {"lat": -4.405769, "lon": 15.316123},
        "Kisenso": {"lat": -4.409440, "lon": 15.342500},
        "Kimbanseke": {"lat": -4.441940, "lon": 15.395000},
        "Selembao": {"lat": -4.371540, "lon": 15.284530},
        "Matete": {"lat": -4.388890, "lon": 15.351670},
        "Bumbu": {"lat": -4.370135, "lon": 15.294240},
        "Makala": {"lat": -4.379788, "lon": 15.309706},
        "N'Djili": {"lat": -4.385750, "lon": 15.444569},
        "Mont-Ngafula": {"lat": -4.455893, "lon": 15.228310},
        "Bandalungwa": {"lat": -4.341848, "lon": 15.283361},
        "Barumbu": {"lat": -4.318979, "lon": 15.325618},
        "Kasa-Vubu": {"lat": -4.338800, "lon": 15.303200},
        "Kintambo": {"lat": -4.326983, "lon": 15.272884},
        "Lingwala": {"lat": -4.320280, "lon": 15.298330},
        "Ngiri-Ngiri": {"lat": -4.357500, "lon": 15.298330},
        "Kinshasa": {"lat": -4.323330, "lon": 15.308060},
        "Maluku": {"lat": -4.073060, "lon": 15.537500},
        "N'Sele": {"lat": -4.420400, "lon": 15.494700}
    }
    
    # Types d'incidents
    incident_types = ["Vol", "Agression", "Manifestation", "Accident", 
                      "Vandalisme", "Violence", "Incendie", "Tapage"]
    
    # Gravités
    gravites = ["Faible", "Moyenne", "Élevée", "Critique"]
    gravite_weights = [0.30, 0.35, 0.25, 0.10]
    
    # Sources
    sources = ["Kimia-kin", "Rapport ONG", "Police", "Réseaux sociaux", "Vision IA"]
    source_weights = [0.35, 0.20, 0.25, 0.15, 0.05]
    
    # Génération des données
    data = []
    communes_list = list(communes.keys())
    
    # Certaines communes ont plus d'incidents (zones à risque)
    high_risk_communes = ["Selembao", "Ngaba", "Matete", "Bumbu", "Kimbanseke"]
    medium_risk_communes = ["Kalamu", "Masina", "N'Djili", "Makala", "Lemba"]
    
    for i in range(n_incidents):
        # Sélection pondérée des communes
        r = np.random.random()
        if r < 0.4:
            commune = np.random.choice(high_risk_communes)
        elif r < 0.7:
            commune = np.random.choice(medium_risk_communes)
        else:
            commune = np.random.choice(communes_list)
        
        coords = communes[commune]
        
        # Ajout de dispersion
        lat = coords["lat"] + np.random.normal(0, 0.008)
        lon = coords["lon"] + np.random.normal(0, 0.008)
        
        # Type et gravité
        type_inc = np.random.choice(incident_types)
        
        # Gravité plus élevée dans les zones à risque
        if commune in high_risk_communes:
            gravite = np.random.choice(gravites, p=[0.15, 0.30, 0.35, 0.20])
        else:
            gravite = np.random.choice(gravites, p=gravite_weights)
        
        source = np.random.choice(sources, p=source_weights)
        
        # Date aléatoire (dernière année)
        days_ago = np.random.randint(0, 365)
        date = datetime.now() - pd.Timedelta(days=days_ago)
        
        data.append({
            "commune": commune,
            "latitude": lat,
            "longitude": lon,
            "date": date,
            "type_incident": type_inc,
            "gravite": gravite,
            "source": source,
            "annee": date.year,
            "mois": date.month,
            "heure": np.random.randint(0, 24)
        })
    
    df = pd.DataFrame(data)
    print(f"✅ {len(df)} incidents de démonstration générés")
    
    # Sauvegarde optionnelle
    try:
        df.to_csv("incidents_kinshasa_demo.csv", index=False)
        print("Données sauvegardées dans incidents_kinshasa_demo.csv")
    except:
        pass
    
    return df


def get_commune_coordinates():
    """
    Retourne le dictionnaire des communes
    """
    return {
        "Bandalungwa": {"lat": -4.341848, "lon": 15.283361, "population": 150000},
        "Barumbu": {"lat": -4.318979, "lon": 15.325618, "population": 120000},
        "Bumbu": {"lat": -4.370135, "lon": 15.294240, "population": 180000},
        "Gombe": {"lat": -4.303056, "lon": 15.303333, "population": 90000},
        "Kalamu": {"lat": -4.341800, "lon": 15.318700, "population": 200000},
        "Kasa-Vubu": {"lat": -4.338800, "lon": 15.303200, "population": 110000},
        "Kimbanseke": {"lat": -4.441940, "lon": 15.395000, "population": 350000},
        "Kinshasa": {"lat": -4.323330, "lon": 15.308060, "population": 100000},
        "Kintambo": {"lat": -4.326983, "lon": 15.272884, "population": 80000},
        "Kisenso": {"lat": -4.409440, "lon": 15.342500, "population": 250000},
        "Lemba": {"lat": -4.405769, "lon": 15.316123, "population": 200000},
        "Limete": {"lat": -4.374389, "lon": 15.345417, "population": 300000},
        "Lingwala": {"lat": -4.320280, "lon": 15.298330, "population": 70000},
        "Makala": {"lat": -4.379788, "lon": 15.309706, "population": 250000},
        "Maluku": {"lat": -4.073060, "lon": 15.537500, "population": 100000},
        "Masina": {"lat": -4.383610, "lon": 15.391390, "population": 400000},
        "Matete": {"lat": -4.388890, "lon": 15.351670, "population": 180000},
        "Mont-Ngafula": {"lat": -4.455893, "lon": 15.228310, "population": 200000},
        "N'Djili": {"lat": -4.385750, "lon": 15.444569, "population": 250000},
        "Ngaba": {"lat": -4.376113, "lon": 15.319617, "population": 130000},
        "Ngaliema": {"lat": -4.369733, "lon": 15.256448, "population": 280000},
        "Ngiri-Ngiri": {"lat": -4.357500, "lon": 15.298330, "population": 60000},
        "N'Sele": {"lat": -4.420400, "lon": 15.494700, "population": 80000},
        "Selembao": {"lat": -4.371540, "lon": 15.284530, "population": 220000}
    }


if __name__ == "__main__":
    df = load_and_preprocess("incidents_kinshasa_30000.csv")
    print(f"Résultat: {len(df)} lignes")
    print(df.head())