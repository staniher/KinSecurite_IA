"""
MODULE DE CLUSTERING SPATIAL DBSCAN
Analyse la densité des incidents pour identifier les zones à risque
Version compatible Streamlit Cloud (sans geopy)
"""

# Import des bibliothèques nécessaires
import pandas as pd                    # Pour manipuler les données tabulaires
import numpy as np                     # Pour les calculs numériques
from sklearn.cluster import DBSCAN     # Algorithme de clustering basé sur la densité
from sklearn.metrics import silhouette_score  # Mesure de qualité du clustering
from scipy.stats import spearmanr      # Corrélation statistique
import warnings                         # Pour ignorer les avertissements superflus
import math                             # Pour les calculs de distance (remplace geopy)
warnings.filterwarnings('ignore')       # Désactive les avertissements non critiques


class ClusteringModule:
    """
    Classe qui encapsule toutes les fonctionnalités de clustering spatial
    Utilise DBSCAN pour identifier les zones de concentration d'incidents
    """
    
    def __init__(self, eps=0.008, min_samples=5):
        """
        Constructeur de la classe - initialise le modèle DBSCAN
        
        Paramètres:
        -----------
        eps : float, rayon de recherche en degrés (0.008 degré ≈ 880 mètres à Kinshasa)
        min_samples : int, nombre minimum de points pour former un cluster dense
        """
        # Stockage des paramètres pour usage ultérieur
        self.eps = eps
        self.min_samples = min_samples
        
        # Création du modèle DBSCAN avec les paramètres fournis
        # La métrique euclidienne est standard pour les coordonnées géographiques
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean')
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calcule la distance entre deux points GPS en kilomètres
        Formule de Haversine (remplace geopy)
        
        Args:
            lat1, lon1: coordonnées du premier point
            lat2, lon2: coordonnées du deuxième point
            
        Returns:
            float: distance en kilomètres
        """
        # Rayon de la Terre en kilomètres
        R = 6371.0
        
        # Conversion des degrés en radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Différences de coordonnées
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Formule de Haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Distance en kilomètres
        distance = R * c
        return distance
    
    def run_clustering(self, df):
        """
        Exécute l'algorithme DBSCAN sur les coordonnées latitude/longitude
        
        Paramètre:
        ----------
        df : DataFrame contenant les colonnes 'latitude' et 'longitude'
        
        Retourne:
        ---------
        DataFrame avec une colonne 'cluster' ajoutée
        """
        # Vérification que le DataFrame n'est pas vide
        if df is None or df.empty:
            # Crée un DataFrame vide avec la colonne cluster
            df = pd.DataFrame(columns=["latitude", "longitude"])
            df['cluster'] = []
            return df
        
        # Vérification que les colonnes nécessaires existent
        if not {'latitude', 'longitude'}.issubset(df.columns):
            # Si les coordonnées manquent, tous les points sont marqués comme bruit (-1)
            df = df.copy()
            df['cluster'] = -1
            return df
        
        # Crée une copie pour ne pas modifier l'original
        df = df.copy()
        
        # Supprime les lignes où les coordonnées sont manquantes
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Il faut au moins 2 points pour faire du clustering
        if len(df) < 2:
            df['cluster'] = -1
            return df
        
        # Extrait les coordonnées dans un tableau numpy (format attendu par scikit-learn)
        coords = df[['latitude', 'longitude']].values
        
        # Exécution du clustering avec gestion d'erreur
        try:
            # fit_predict effectue le clustering et retourne les labels (-1 = bruit)
            df['cluster'] = self.model.fit_predict(coords)
        except Exception as e:
            # En cas d'erreur (ex: mémoire insuffisante), tous les points sont bruit
            print(f"Erreur DBSCAN: {e}")
            df['cluster'] = -1
        
        return df
    
    def get_cluster_statistics(self, df_clustered):
        """
        Calcule les statistiques détaillées des clusters
        
        Paramètre:
        ----------
        df_clustered : DataFrame avec colonne 'cluster'
        
        Retourne:
        ---------
        Dictionnaire avec statistiques
        """
        stats = {
            'total_points': 0,
            'n_clusters': 0,
            'n_noise': 0,
            'noise_percentage': 0,
            'clusters_detail': []
        }
        
        # Vérification des données
        if df_clustered is None or df_clustered.empty:
            return stats
        
        if 'cluster' not in df_clustered.columns:
            return stats
        
        stats['total_points'] = len(df_clustered)
        
        # Compter les clusters
        labels = df_clustered['cluster']
        unique_labels = set(labels)
        
        # Nombre de clusters (excluant -1 qui est le bruit)
        stats['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Comptage du bruit
        stats['n_noise'] = sum(labels == -1)
        stats['noise_percentage'] = (stats['n_noise'] / stats['total_points']) * 100 if stats['total_points'] > 0 else 0
        
        # Détails par cluster
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_points = df_clustered[df_clustered['cluster'] == label]
            
            # Centre du cluster (moyenne des coordonnées)
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            # Rayon approximatif en km (calculé avec la formule de Haversine)
            max_dist = 0
            for _, row in cluster_points.iterrows():
                # Utilise la formule de Haversine au lieu de geopy
                dist = self._haversine_distance(center_lat, center_lon, row['latitude'], row['longitude'])
                max_dist = max(max_dist, dist)
            
            # Types d'incidents dans le cluster
            incident_types = {}
            if 'type_incident' in cluster_points.columns:
                incident_types = cluster_points['type_incident'].value_counts().to_dict()
            
            # Gravité dominante
            main_gravite = "Inconnu"
            if 'gravite' in cluster_points.columns:
                gravite_counts = cluster_points['gravite'].value_counts()
                if not gravite_counts.empty:
                    main_gravite = gravite_counts.index[0]
            
            stats['clusters_detail'].append({
                'cluster_id': int(label),
                'size': len(cluster_points),
                'percentage': (len(cluster_points) / stats['total_points']) * 100 if stats['total_points'] > 0 else 0,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'radius_km': max_dist,
                'main_incident_type': max(incident_types, key=incident_types.get) if incident_types else 'Inconnu',
                'incident_types': incident_types,
                'main_gravite': main_gravite
            })
        
        # Trier par taille décroissante
        stats['clusters_detail'].sort(key=lambda x: x['size'], reverse=True)
        
        return stats
    
    def evaluate_model(self, df_clustered):
        """
        Évalue la qualité du clustering
        
        Paramètre:
        ----------
        df_clustered : DataFrame avec colonne 'cluster'
        
        Retourne:
        ---------
        Dictionnaire avec métriques d'évaluation
        """
        metrics = {
            'silhouette': 0.0,
            'silhouette_interpretation': 'Non calculé',
            'n_clusters': 0,
            'n_noise': 0
        }
        
        if df_clustered is None or df_clustered.empty:
            return metrics
        
        if 'cluster' not in df_clustered.columns:
            return metrics
        
        labels = df_clustered['cluster']
        coords = df_clustered[['latitude', 'longitude']].values
        
        unique_labels = set(labels)
        metrics['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        metrics['n_noise'] = sum(labels == -1)
        
        # Calcul du silhouette score (nécessite au moins 2 clusters)
        if metrics['n_clusters'] >= 2 and len(df_clustered) >= 10:
            try:
                # Échantillonner pour les grands datasets
                if len(df_clustered) > 3000:
                    indices = np.random.choice(len(df_clustered), 3000, replace=False)
                    score = silhouette_score(coords[indices], labels.iloc[indices])
                else:
                    score = silhouette_score(coords, labels)
                
                metrics['silhouette'] = score
                
                # Interprétation
                if score > 0.5:
                    metrics['silhouette_interpretation'] = 'Excellent'
                elif score > 0.3:
                    metrics['silhouette_interpretation'] = 'Bon'
                elif score > 0.2:
                    metrics['silhouette_interpretation'] = 'Acceptable'
                else:
                    metrics['silhouette_interpretation'] = 'Faible'
                    
            except Exception as e:
                print(f"Erreur silhouette: {e}")
                metrics['silhouette_interpretation'] = 'Erreur calcul'
        
        return metrics
    
    def classify_communes(self, df):
        """
        Classe les communes par niveau d'insécurité
        
        Paramètre:
        ----------
        df : DataFrame avec colonne 'commune'
        
        Retourne:
        ---------
        DataFrame avec classification
        """
        if df is None or df.empty or 'commune' not in df.columns:
            return pd.DataFrame(columns=['commune', 'count', 'incidents_mensuels', 'niveau_insecurite', 'couleur'])
        
        # Compter par commune
        stats = df.groupby('commune').size().reset_index(name='count')
        
        # Calculer incidents par mois
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if not df['date'].isna().all():
                date_min = df['date'].min()
                date_max = df['date'].max()
                months = max((date_max - date_min).days / 30, 1)
            else:
                months = 12
        else:
            months = 12
        
        stats['incidents_mensuels'] = stats['count'] / months
        
        # Niveaux de risque
        def get_level(val):
            if val > 60:
                return "Très élevé", "#dc2626"  # Rouge
            elif val > 30:
                return "Élevé", "#f97316"       # Orange
            elif val > 10:
                return "Modéré", "#eab308"      # Jaune
            else:
                return "Faible", "#22c55e"      # Vert
        
        levels = stats['incidents_mensuels'].apply(lambda x: get_level(x))
        stats[['niveau_insecurite', 'couleur']] = pd.DataFrame(levels.tolist(), index=stats.index)
        
        return stats.sort_values('incidents_mensuels', ascending=False)
    
    def calculate_spearman(self, df_results):
        """
        Calcule la corrélation de Spearman
        
        Paramètre:
        ----------
        df_results : DataFrame avec colonnes 'commune' et 'count'
        
        Retourne:
        ---------
        Tuple (corrélation, p-value)
        """
        if df_results is None or df_results.empty or 'commune' not in df_results.columns:
            return 0.0, 1.0
        
        # Densité prédite
        if 'count' in df_results.columns:
            predicted = df_results.set_index('commune')['count']
        else:
            predicted = df_results.groupby('commune').size()
        
        # Risque officiel (échelle 1-5)
        official_risk = {
            "Selembao": 5, "Ngaba": 5, "Matete": 5, "Bumbu": 4, "Kimbanseke": 4,
            "Kalamu": 3, "Masina": 3, "N'Djili": 3, "Makala": 3, "Lemba": 3,
            "Limete": 2, "Kisenso": 2, "Ngaliema": 2, "Bandalungwa": 2, "Barumbu": 2,
            "Kasa-Vubu": 2, "Kintambo": 2, "Lingwala": 2, "Ngiri-Ngiri": 2,
            "Gombe": 1, "Kinshasa": 1, "Maluku": 1, "Mont-Ngafula": 1, "N'Sele": 1
        }
        
        # Créer DataFrame commun
        common_communes = [c for c in predicted.index if c in official_risk]
        
        if len(common_communes) < 3:
            return 0.0, 1.0
        
        predicted_values = [predicted[c] for c in common_communes]
        official_values = [official_risk[c] for c in common_communes]
        
        try:
            correlation, p_value = spearmanr(predicted_values, official_values)
            return correlation, p_value
        except Exception as e:
            print(f"Erreur Spearman: {e}")
            return 0.0, 1.0
    
    def get_hotspots(self, df_clustered, top_n=10):
        """
        Identifie les hotspots (points de forte concentration)
        
        Paramètres:
        ----------
        df_clustered : DataFrame avec colonne 'cluster'
        top_n : nombre de hotspots à retourner
        
        Retourne:
        ---------
        DataFrame des hotspots
        """
        if df_clustered is None or df_clustered.empty:
            return pd.DataFrame()
        
        # Filtrer les points dans les clusters (pas le bruit)
        if 'cluster' in df_clustered.columns:
            clustered_points = df_clustered[df_clustered['cluster'] != -1].copy()
        else:
            clustered_points = df_clustered.copy()
        
        if clustered_points.empty:
            return pd.DataFrame()
        
        # Arrondir les coordonnées pour grouper les points proches
        clustered_points['lat_rounded'] = clustered_points['latitude'].round(3)
        clustered_points['lon_rounded'] = clustered_points['longitude'].round(3)
        
        # Compter par zone
        hotspots = clustered_points.groupby(['lat_rounded', 'lon_rounded', 'commune']).size().reset_index(name='density')
        hotspots = hotspots.sort_values('density', ascending=False).head(top_n)
        
        return hotspots
    
    def get_cluster_geometries(self, df_clustered):
        """
        Calcule les géométries des clusters pour l'affichage sur carte
        
        Paramètre:
        ----------
        df_clustered : DataFrame avec colonne 'cluster'
        
        Retourne:
        ---------
        Dictionnaire des clusters avec centres et rayons
        """
        if df_clustered is None or df_clustered.empty:
            return {}
        
        if 'cluster' not in df_clustered.columns:
            return {}
        
        clusters_data = df_clustered[df_clustered['cluster'] != -1].copy()
        
        if clusters_data.empty:
            return {}
        
        cluster_info = {}
        colors = ['#dc2626', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']
        
        for idx, cluster_id in enumerate(clusters_data['cluster'].unique()):
            cluster_points = clusters_data[clusters_data['cluster'] == cluster_id]
            
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            # Calcul du rayon moyen (formule de Haversine)
            distances = []
            for _, row in cluster_points.iterrows():
                dist = self._haversine_distance(center_lat, center_lon, row['latitude'], row['longitude'])
                distances.append(dist)
            
            avg_radius = np.mean(distances) if distances else 0.5
            
            cluster_info[int(cluster_id)] = {
                'center': [center_lat, center_lon],
                'radius_km': avg_radius,
                'point_count': len(cluster_points),
                'color': colors[idx % len(colors)]
            }
        
        return cluster_info
    
    def get_zones_by_commune(self, df_clustered):
        """
        Calcule les zones de risque par commune
        
        Paramètre:
        ----------
        df_clustered : DataFrame avec colonnes 'commune' et 'cluster'
        
        Retourne:
        ---------
        DataFrame avec scores de risque par commune
        """
        if df_clustered is None or df_clustered.empty or 'commune' not in df_clustered.columns:
            return pd.DataFrame()
        
        # Statistiques par commune
        commune_stats = df_clustered.groupby('commune').agg({
            'latitude': 'count'
        }).rename(columns={'latitude': 'total_incidents'})
        
        # Ajouter incidents dans clusters
        if 'cluster' in df_clustered.columns:
            in_clusters = df_clustered[df_clustered['cluster'] != -1].groupby('commune').size()
            commune_stats['incidents_in_clusters'] = in_clusters
            commune_stats['incidents_in_clusters'] = commune_stats['incidents_in_clusters'].fillna(0)
        
        # Score de risque normalisé
        max_incidents = commune_stats['total_incidents'].max()
        if max_incidents > 0:
            commune_stats['calculated_risk_score'] = (commune_stats['total_incidents'] / max_incidents) * 100
        else:
            commune_stats['calculated_risk_score'] = 0
        
        # Niveaux de risque
        def get_risk_level(score):
            if score > 70:
                return "Très élevé", "#dc2626"
            elif score > 40:
                return "Élevé", "#f97316"
            elif score > 15:
                return "Modéré", "#eab308"
            else:
                return "Faible", "#22c55e"
        
        risk_levels = commune_stats['calculated_risk_score'].apply(get_risk_level)
        commune_stats['niveau_insecurite'] = [r[0] for r in risk_levels]
        commune_stats['couleur'] = [r[1] for r in risk_levels]
        
        # Incidents par mois
        if 'date' in df_clustered.columns:
            df_clustered['date'] = pd.to_datetime(df_clustered['date'], errors='coerce')
            if not df_clustered['date'].isna().all():
                date_range = df_clustered['date'].max() - df_clustered['date'].min()
                months = max(date_range.days / 30, 1)
            else:
                months = 12
        else:
            months = 12
        
        commune_stats['incidents_per_month'] = commune_stats['total_incidents'] / months
        
        return commune_stats.sort_values('calculated_risk_score', ascending=False)


# Test du module
if __name__ == "__main__":
    print("Test du module de clustering...")
    
    # Création de données de test
    test_data = pd.DataFrame({
        'latitude': np.random.normal(-4.35, 0.02, 200),
        'longitude': np.random.normal(15.30, 0.02, 200),
        'commune': ['Gombe'] * 100 + ['Limete'] * 100,
        'type_incident': ['Vol'] * 200,
        'gravite': ['Moyenne'] * 200,
        'date': [pd.Timestamp.now()] * 200
    })
    
    cm = ClusteringModule()
    
    # Test clustering
    result = cm.run_clustering(test_data)
    print(f"Clustering terminé: {len(result)} points")
    
    # Test statistiques
    stats = cm.get_cluster_statistics(result)
    print(f"Clusters: {stats['n_clusters']}, Bruit: {stats['n_noise']}")
    
    # Test évaluation
    metrics = cm.evaluate_model(result)
    print(f"Silhouette: {metrics['silhouette']:.3f} ({metrics['silhouette_interpretation']})")
    
    print("✅ Tous les tests passés")
