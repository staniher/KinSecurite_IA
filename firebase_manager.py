"""
MODULE FIREBASE MANAGER
Gère la synchronisation des incidents avec Firebase Firestore
Support: fichier JSON local, variable d'environnement, Streamlit secrets
"""

# ============================================================================
# IMPORT DES BIBLIOTHÈQUES
# ============================================================================

import os
# Module pour interagir avec le système d'exploitation (fichiers, chemins, variables d'environnement)
import json
# Module pour manipuler les données au format JSON (JavaScript Object Notation)
from datetime import datetime
# Module pour manipuler les dates et heures, utilisé pour horodater les incidents
import pandas as pd
# Bibliothèque de manipulation de données (DataFrames), utilisée pour convertir les données

# Tentative d'import de Streamlit pour vérifier si l'application tourne sur Streamlit Cloud
try:
    import streamlit as st
    # Si l'import réussit, on est dans un environnement Streamlit
    STREAMLIT_AVAILABLE = True
    # Variable booléenne indiquant que Streamlit est disponible
except ImportError:
    # Si l'import échoue, on n'est pas dans un environnement Streamlit
    STREAMLIT_AVAILABLE = False
    # Variable booléenne indiquant que Streamlit n'est pas disponible


# ============================================================================
# CLASSE PRINCIPALE FIREBASE MANAGER
# ============================================================================

class FirebaseManager:
    """
    Classe qui gère la connexion et les opérations Firebase
    Fonctionne en mode réel (cloud) ou en mode mock (local)
    Supporte:
    - Fichier serviceAccountKey.json (développement local)
    - Variable d'environnement FIREBASE_CREDENTIALS (Render, Heroku)
    - Streamlit secrets (Streamlit Cloud)
    """
    
    def __init__(self, mock_mode=False):
        """
        Constructeur de la classe - Initialise le gestionnaire Firebase
        
        Args:
            mock_mode: booléen - True = stockage local (simulation), 
                      False = connexion réelle au cloud Firebase
        """
        # Variable qui stockera le client Firestore (None = pas encore connecté)
        self.db = None
        # Mode d'opération (mock = simulation sans cloud)
        self.mock_mode = mock_mode
        # Base de données locale pour le mode mock (dictionnaire stockant des listes)
        self.mock_db = {}
        # Nom de la collection principale dans Firestore
        self.collection_name = "incidents"
        
        # Si le mode réel est demandé (mock_mode = False)
        if not mock_mode:
            # Appel de la méthode qui établit la connexion à Firebase
            self._connect_firebase()
        
        # Affiche le mode d'opération dans la console (pour débogage)
        if self.mock_mode:
            print("Mode MOCK - Stockage local uniquement")
        else:
            print("Mode REEL - Connecté à Firebase Cloud")
    
    def _connect_firebase(self):
        """
        Tente d'établir une connexion avec Firebase
        Supporte 4 méthodes (ordre de priorité):
        1. Streamlit secrets (pour Streamlit Cloud) - PRIORITÉ MAXIMALE
        2. Fichier serviceAccountKey.json (développement local)
        3. Variable d'environnement FIREBASE_CREDENTIALS (Render, Heroku)
        4. Fichier firebase-key.json (fallback)
        """
        try:
            # Tentative d'import des modules Firebase
            # Ces imports sont placés à l'intérieur pour éviter une erreur si Firebase n'est pas installé
            import firebase_admin
            # Module principal Firebase Admin SDK
            from firebase_admin import credentials, firestore
            # credentials: pour charger les clés d'authentification
            # firestore: pour interagir avec la base de données Firestore
            
            # Variable qui contiendra les identifiants Firebase
            cred = None
            
            # ============================================================
            # MÉTHODE 1: Streamlit secrets (pour déploiement cloud)
            # ============================================================
            if STREAMLIT_AVAILABLE and hasattr(st, 'secrets') and 'FIREBASE_CREDENTIALS' in st.secrets:
                # Vérifie que Streamlit est disponible ET qu'il existe des secrets ET que la clé Firebase existe
                try:
                    # st.secrets["FIREBASE_CREDENTIALS"] contient le JSON de la clé Firebase
                    # Convertit la chaîne JSON en dictionnaire Python
                    cred_dict = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
                    # Crée un objet Credentials à partir du dictionnaire
                    cred = credentials.Certificate(cred_dict)
                    # Message de confirmation dans la console
                    print("Firebase: Connexion via Streamlit secrets")
                except Exception as e:
                    # En cas d'erreur (JSON invalide par exemple)
                    print(f"Erreur lecture Streamlit secrets: {e}")
            
            # ============================================================
            # MÉTHODE 2: Fichier serviceAccountKey.json (développement local)
            # ============================================================
            # Si aucune clé n'a été trouvée ET que le fichier serviceAccountKey.json existe
            if not cred and os.path.exists("serviceAccountKey.json"):
                # Charge le fichier JSON de clé Firebase
                cred = credentials.Certificate("serviceAccountKey.json")
                print("Firebase: Connexion via serviceAccountKey.json")
            
            # ============================================================
            # MÉTHODE 3: Variable d'environnement (Render, Heroku, etc.)
            # ============================================================
            # Si aucune clé n'a été trouvée ET que la variable d'environnement existe
            if not cred and os.getenv("FIREBASE_CREDENTIALS"):
                try:
                    # Récupère la variable d'environnement
                    cred_json = os.getenv("FIREBASE_CREDENTIALS")
                    # Convertit la chaîne JSON en dictionnaire
                    cred_dict = json.loads(cred_json)
                    # Crée l'objet Credentials
                    cred = credentials.Certificate(cred_dict)
                    print("Firebase: Connexion via variable d'environnement")
                except Exception as e:
                    print(f"Erreur lecture variable FIREBASE_CREDENTIALS: {e}")
            
            # ============================================================
            # MÉTHODE 4: Fichier alternatif firebase-key.json
            # ============================================================
            # Si aucune clé n'a été trouvée ET que le fichier firebase-key.json existe
            if not cred and os.path.exists("firebase-key.json"):
                # Charge le fichier JSON de clé Firebase
                cred = credentials.Certificate("firebase-key.json")
                print("Firebase: Connexion via firebase-key.json")
            
            # ============================================================
            # Initialisation si une clé a été trouvée
            # ============================================================
            if cred:
                # Vérifie si une application Firebase n'a pas déjà été initialisée
                # _apps est un dictionnaire interne de firebase_admin contenant les applications initialisées
                if not firebase_admin._apps:
                    # Initialise l'application Firebase avec les identifiants
                    firebase_admin.initialize_app(cred)
                # Crée un client Firestore pour interagir avec la base de données
                self.db = firestore.client()
                # Désactive le mode mock car la connexion a réussi
                self.mock_mode = False
                print("Connexion Firebase établie avec succès")
            else:
                # Aucune configuration trouvée
                print("Aucune configuration Firebase trouvée - Passage en mode MOCK")
                # Active le mode mock pour que l'application continue de fonctionner
                self.mock_mode = True
                
        except ImportError:
            # Exception: le package firebase-admin n'est pas installé
            print("firebase-admin non installé - Mode MOCK")
            print("Installation: pip install firebase-admin")
            self.mock_mode = True
        except Exception as e:
            # Exception générique pour toute autre erreur
            print(f"Erreur Firebase: {e} - Mode MOCK")
            self.mock_mode = True
    
    def push_incident(self, incident_data):
        """
        Envoie un incident vers Firebase ou le stockage local (mode mock)
        
        Args:
            incident_data: dict - Dictionnaire contenant les informations de l'incident
            
        Returns:
            bool - True si l'opération a réussi, False sinon
        """
        # Ajoute un timestamp (horodatage) pour le tri chronologique
        # isoformat() génère une chaîne comme "2024-01-15T14:30:00.123456"
        incident_data["firebase_timestamp"] = datetime.now().isoformat()
        
        # ============================================================
        # CAS 1: Mode MOCK - Stockage LOCAL (simulation)
        # ============================================================
        if self.mock_mode:
            # Vérifie si la clé 'mock_incidents' existe dans le dictionnaire mock_db
            if 'mock_incidents' not in self.mock_db:
                # Si elle n'existe pas, crée une liste vide
                self.mock_db['mock_incidents'] = []
            # Ajoute l'incident à la liste locale
            self.mock_db['mock_incidents'].append(incident_data)
            # Message de confirmation dans la console
            print(f"[MOCK] Incident ajouté: {incident_data.get('type_incident', 'N/A')}")
            # Retourne True pour indiquer le succès (même en simulation)
            return True
        
        # ============================================================
        # CAS 2: Mode REEL - Envoi vers Firebase Cloud
        # ============================================================
        else:
            try:
                # self.db.collection(self.collection_name) sélectionne la collection "incidents"
                # .add(incident_data) ajoute un nouveau document avec les données
                self.db.collection(self.collection_name).add(incident_data)
                # Message de confirmation
                print(f"[FIREBASE] Incident envoyé: {incident_data.get('type_incident', 'N/A')}")
                return True
            except Exception as e:
                # En cas d'erreur (problème réseau, authentification, etc.)
                print(f"Erreur push Firebase: {e}")
                return False
    
    def get_incidents(self, limit=500):
        """
        Récupère les derniers incidents depuis Firebase ou le stockage local
        
        Args:
            limit: int - Nombre maximum d'incidents à récupérer (défaut: 500)
            
        Returns:
            list - Liste des incidents (dictionnaires)
        """
        # ============================================================
        # CAS 1: Mode MOCK - Récupération depuis le stockage LOCAL
        # ============================================================
        if self.mock_mode:
            # Récupère la liste 'mock_incidents' si elle existe, sinon une liste vide
            incidents = self.mock_db.get('mock_incidents', [])
            # Retourne les 'limit' derniers incidents (slice [-limit:] prend les derniers)
            return incidents[-limit:]
        
        # ============================================================
        # CAS 2: Mode REEL - Requête vers Firebase Cloud
        # ============================================================
        else:
            try:
                # Construit une requête Firestore:
                # .collection(self.collection_name) : sélectionne la collection "incidents"
                # .order_by("firebase_timestamp", direction="DESCENDING") : trie par date décroissante (plus récent en premier)
                # .limit(limit) : limite le nombre de résultats
                # .stream() : exécute la requête et retourne un itérateur
                docs = self.db.collection(self.collection_name)\
                    .order_by("firebase_timestamp", direction="DESCENDING")\
                    .limit(limit)\
                    .stream()
                
                # Liste qui contiendra les incidents
                incidents = []
                # Parcourt chaque document retourné par la requête
                for doc in docs:
                    # Convertit le document Firestore en dictionnaire Python
                    incident = doc.to_dict()
                    # Ajoute l'ID Firestore du document (utile pour les mises à jour)
                    incident['firebase_id'] = doc.id
                    # Ajoute l'incident à la liste
                    incidents.append(incident)
                # Message de confirmation
                print(f"[FIREBASE] {len(incidents)} incidents récupérés")
                return incidents
                
            except Exception as e:
                # En cas d'erreur
                print(f"Erreur get Firebase: {e}")
                # Retourne une liste vide
                return []
    
    def push_batch(self, df, batch_size=100, on_progress=None):
        """
        Envoie un lot d'incidents (utile pour l'upload initial des données CSV)
        
        Args:
            df: DataFrame pandas - Données à uploader
            batch_size: int - Nombre d'incidents à envoyer par lot (défaut: 100)
            on_progress: function - Fonction callback appelée régulièrement pour suivre la progression
            
        Returns:
            int - Nombre d'incidents uploadés avec succès
        """
        # Vérifie si on est en mode mock
        if self.mock_mode:
            print("Mode MOCK - Pas d'upload réel")
            return 0
        
        # Initialise le compteur
        count = 0
        # Nombre total d'incidents à uploader
        total = len(df)
        
        # Parcourt chaque ligne du DataFrame
        # iterrows() retourne (index, Series) pour chaque ligne
        for idx, row in df.iterrows():
            # Convertit la ligne en dictionnaire d'incident
            incident = self._row_to_incident(row)
            # Envoie l'incident à Firebase
            if self.push_incident(incident):
                # Incrémente le compteur si l'envoi a réussi
                count += 1
            
            # Si une fonction de progression a été fournie ET qu'on a atteint un multiple de batch_size
            if on_progress and count % batch_size == 0:
                # Appelle la fonction avec le nombre traité et le total
                on_progress(count, total)
        
        # Message final
        print(f"Upload terminé: {count}/{total} incidents")
        return count
    
    def _row_to_incident(self, row):
        """
        Convertit une ligne de DataFrame en dictionnaire d'incident
        
        Args:
            row: pandas Series - Ligne du DataFrame
            
        Returns:
            dict - Incident prêt pour être envoyé à Firebase
        """
        # Construit un dictionnaire avec les champs standard
        # .get() permet d'éviter les erreurs si la colonne n'existe pas
        incident = {
            # Nom de la commune (converti en string)
            "commune": str(row.get("commune", "")),
            # Latitude (convertie en float, 0 par défaut)
            "latitude": float(row.get("latitude", 0)),
            # Longitude (convertie en float, 0 par défaut)
            "longitude": float(row.get("longitude", 0)),
            # Date de l'incident (convertie en string)
            "date": str(row.get("date", "")),
            # Type d'incident (vol, agression, etc.)
            "type_incident": str(row.get("type_incident", "")),
            # Gravité de l'incident (Faible, Moyenne, Élevée, Critique)
            "gravite": str(row.get("gravite", "")),
            # Source de l'information (Kimia-kin, Police, Réseaux sociaux, etc.)
            "source": str(row.get("source", "")),
            # Timestamp Firebase pour le tri (date et heure actuelles)
            "firebase_timestamp": datetime.now().isoformat()
        }
        
        # Vérifie si la colonne 'source_categorie' existe dans la ligne
        if "source_categorie" in row:
            # Ajoute la catégorie de source au dictionnaire
            incident["source_categorie"] = str(row["source_categorie"])
        
        # Retourne le dictionnaire complet
        return incident
    
    def push_to_collection(self, collection_name, data):
        """
        Envoie des données vers une collection spécifique (ex: citizen_reports)
        
        Args:
            collection_name: str - Nom de la collection Firestore
            data: dict - Données à envoyer
            
        Returns:
            bool - True si l'opération a réussi, False sinon
        """
        # ============================================================
        # Mode MOCK - Stockage local
        # ============================================================
        if self.mock_mode:
            # Vérifie si la collection existe dans le dictionnaire mock_db
            if collection_name not in self.mock_db:
                # Si elle n'existe pas, crée une liste vide
                self.mock_db[collection_name] = []
            # Ajoute les données à la collection locale
            self.mock_db[collection_name].append(data)
            # Message de confirmation
            print(f"[MOCK] Données ajoutées à {collection_name}")
            return True
        
        # ============================================================
        # Mode REEL - Envoi vers Firebase
        # ============================================================
        else:
            try:
                # Sélectionne la collection et ajoute les données
                self.db.collection(collection_name).add(data)
                print(f"[FIREBASE] Données envoyées à {collection_name}")
                return True
            except Exception as e:
                print(f"Erreur push to {collection_name}: {e}")
                return False
    
    def get_from_collection(self, collection_name, limit=500):
        """
        Récupère des données depuis une collection spécifique
        
        Args:
            collection_name: str - Nom de la collection Firestore
            limit: int - Nombre maximum de documents à récupérer (défaut: 500)
            
        Returns:
            list - Liste des documents récupérés
        """
        # ============================================================
        # Mode MOCK - Récupération locale
        # ============================================================
        if self.mock_mode:
            # Récupère la collection si elle existe, sinon une liste vide
            # Retourne les 'limit' derniers éléments
            return self.mock_db.get(collection_name, [])[-limit:]
        
        # ============================================================
        # Mode REEL - Requête Firebase
        # ============================================================
        else:
            try:
                # Sélectionne la collection, limite le nombre et récupère les documents
                docs = self.db.collection(collection_name).limit(limit).stream()
                # Convertit chaque document en dictionnaire et les retourne dans une liste
                return [doc.to_dict() for doc in docs]
            except Exception as e:
                print(f"Erreur get from {collection_name}: {e}")
                return []
    
    def update_document(self, collection_name, doc_id, data):
        """
        Met à jour un document spécifique (utile pour la modération des signalements)
        
        Args:
            collection_name: str - Nom de la collection
            doc_id: str - ID du document (report_id pour les signalements)
            data: dict - Nouvelles données à fusionner
            
        Returns:
            bool - True si la mise à jour a réussi, False sinon
        """
        # ============================================================
        # Mode MOCK - Mise à jour locale
        # ============================================================
        if self.mock_mode:
            # Récupère la collection locale
            collection = self.mock_db.get(collection_name, [])
            # Parcourt la liste pour trouver le document avec le bon report_id
            for idx, item in enumerate(collection):
                if item.get('report_id') == doc_id:
                    # Met à jour le document en fusionnant les données
                    collection[idx] = data
                    # Sauvegarde la collection modifiée
                    self.mock_db[collection_name] = collection
                    print(f"[MOCK] Document {doc_id} mis à jour")
                    return True
            # Document non trouvé
            print(f"[MOCK] Document {doc_id} non trouvé")
            return False
        
        # ============================================================
        # Mode REEL - Mise à jour Firebase
        # ============================================================
        else:
            try:
                # Crée une requête pour trouver le document avec report_id = doc_id
                query = self.db.collection(collection_name).where('report_id', '==', doc_id).limit(1).stream()
                # Parcourt les résultats (normalement 1 seul)
                for doc in query:
                    # Met à jour le document avec les nouvelles données
                    doc.reference.update(data)
                    print(f"[FIREBASE] Document {doc_id} mis à jour")
                    return True
                # Document non trouvé
                print(f"[FIREBASE] Document {doc_id} non trouvé")
                return False
            except Exception as e:
                print(f"Erreur update: {e}")
                return False
    
    def get_stats(self):
        """
        Récupère des statistiques sur les incidents dans Firebase
        
        Returns:
            dict - Statistiques contenant le mode et le nombre total d'incidents
        """
        # ============================================================
        # Mode MOCK
        # ============================================================
        if self.mock_mode:
            return {
                "mode": "MOCK",  # Indique le mode
                "total": len(self.mock_db.get('mock_incidents', []))  # Compte les incidents locaux
            }
        
        # ============================================================
        # Mode REEL
        # ============================================================
        else:
            try:
                # Récupère tous les documents de la collection incidents
                docs = self.db.collection(self.collection_name).get()
                return {
                    "mode": "REEL",  # Indique le mode
                    "total": len(docs)  # Compte le nombre de documents
                }
            except Exception as e:
                # En cas d'erreur, retourne l'erreur
                return {
                    "mode": "REEL", 
                    "error": str(e),
                    "total": 0
                }
    
    def clear_mock(self):
        """
        Nettoie la base de données mock (utile pour les tests ou le redémarrage)
        """
        # Réinitialise le dictionnaire mock_db à un dictionnaire vide
        self.mock_db = {}
        # Message de confirmation
        print("Base mock vidée")


# ============================================================================
# POINT D'ENTRÉE POUR TESTER LE MODULE (s'exécute uniquement si le fichier est lancé directement)
# ============================================================================
if __name__ == "__main__":
    # Affiche une séparation dans la console
    print("=" * 50)
    print("TEST DU MODULE FIREBASE MANAGER")
    print("=" * 50)
    
    # Test en mode mock (simulation)
    print("\n1. Test mode MOCK:")
    # Crée une instance de FirebaseManager en mode mock
    fm_mock = FirebaseManager(mock_mode=True)
    # Envoie un incident de test
    fm_mock.push_incident({"test": "data", "type_incident": "Test Mock"})
    # Affiche le nombre d'incidents dans la base mock
    print(f"   Incidents: {len(fm_mock.get_incidents())}")
    
    # Test en mode réel (si Firebase est configuré)
    print("\n2. Test mode REEL (si Firebase configuré):")
    # Crée une instance en mode réel (tente de se connecter)
    fm_real = FirebaseManager(mock_mode=False)
    # Vérifie si on est bien en mode réel (non mock)
    if not fm_real.mock_mode:
        # Récupère les statistiques
        stats = fm_real.get_stats()
        print(f"   Mode: {stats.get('mode')}")
        print(f"   Total incidents dans le cloud: {stats.get('total', 0)}")
    else:
        # Firebase non configuré
        print("   Firebase non configuré - Test ignoré")
    
    # Fin du test
    print("\nTest terminé")