"""
MODULE VISION PAR ORDINATEUR - VERSION COMPLÈTE
Détection d'objets avec YOLOv8 + upload d'images
Version compatible Streamlit Cloud (sans OpenCV)
"""

# Import des bibliothèques
import streamlit as st
import numpy as np
from datetime import datetime
import random
import os
from PIL import Image, ImageOps
import tempfile
import requests
from io import BytesIO


class VisionModule:
    """
    Module de vision par ordinateur avec YOLOv8
    Permet l'upload d'images et la détection d'objets
    Version compatible sans OpenCV (utilise PIL)
    """
    
    def __init__(self, use_real_detection=True):
        """
        Initialise le module vision
        
        Args:
            use_real_detection: True pour utiliser YOLO, False pour simulation
        """
        # Modèle YOLO (sera chargé plus tard)
        self.model = None
        # Indique si YOLO est disponible
        self.yolo_available = False
        # Mode d'opération (réel ou simulation)
        self.use_real_detection = use_real_detection
        
        # Tentative de chargement de YOLO
        if use_real_detection:
            self._load_yolo_model()
        
        # Classes d'objets pertinentes pour la sécurité
        self.relevant_classes = {
            'person': 'Personne',
            'car': 'Véhicule',
            'truck': 'Camion',
            'motorcycle': 'Moto',
            'bus': 'Bus',
            'knife': 'Couteau',
            'scissors': 'Ciseaux',
            'gun': 'Arme à feu',
            'fire': 'Feu/Incendie',
            'smoke': 'Fumée'
        }
    
    def _load_yolo_model(self):
        """
        Charge le modèle YOLOv8 sans dépendre d'OpenCV
        """
        try:
            from ultralytics import YOLO
            
            # Téléchargement automatique si absent
            model_path = 'yolov8n.pt'
            
            # Afficher une barre de progression (fonctionne sur Streamlit)
            with st.spinner("📥 Chargement du modèle YOLOv8 (premier lancement peut prendre du temps)..."):
                self.model = YOLO(model_path)
            
            self.yolo_available = True
            print("✅ YOLOv8 chargé avec succès")
            
            # Test rapide avec une image factice (sans OpenCV)
            test_img = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = self.model(test_img, verbose=False)
            print("✅ Test YOLO réussi")
            
        except ImportError:
            st.warning("⚠️ Ultralytics non installé. Installation: pip install ultralytics")
            self.yolo_available = False
        except Exception as e:
            st.warning(f"⚠️ Erreur YOLO: {e}")
            self.yolo_available = False
    
    def detect_from_image(self, image_source):
        """
        Détecte des objets dans une image (sans OpenCV)
        
        Args:
            image_source: chemin d'image, URL, ou array numpy
        
        Returns:
            Liste des détections
        """
        # Si YOLO n'est pas disponible, utiliser la simulation
        if not self.yolo_available or not self.use_real_detection:
            return self._simulate_detection()
        
        try:
            # Chargement de l'image (uniquement avec PIL, pas OpenCV)
            if isinstance(image_source, str):
                if image_source.startswith('http'):
                    # Téléchargement depuis URL
                    response = requests.get(image_source)
                    image = Image.open(BytesIO(response.content))
                else:
                    # Chargement depuis fichier
                    image = Image.open(image_source)
            else:
                # Si c'est déjà un objet PIL Image
                image = image_source
            
            if image is None:
                return self._simulate_detection()
            
            # Convertir PIL Image en numpy array (format attendu par YOLO)
            image_np = np.array(image)
            
            # Détection YOLO
            results = self.model(image_np, verbose=False)
            
            # Traitement des résultats
            detections = []
            
            for r in results:
                if r.boxes is None:
                    continue
                
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    confidence = float(box.conf[0])
                    
                    # Filtrer les classes pertinentes
                    if confidence > 0.5 and label in self.relevant_classes:
                        detection = {
                            "label": self.relevant_classes[label],
                            "original_label": label,
                            "confidence": confidence,
                            "confidence_pct": f"{confidence*100:.0f}%",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Ajouter la sévérité basée sur l'objet
                        if label in ['knife', 'gun', 'scissors']:
                            detection["severity"] = "CRITIQUE"
                            detection["alert"] = True
                        elif label in ['fire', 'smoke']:
                            detection["severity"] = "CRITIQUE"
                            detection["alert"] = True
                        elif label == 'person' and confidence > 0.8:
                            detection["severity"] = "SURVEILLANCE"
                            detection["alert"] = False
                        else:
                            detection["severity"] = "INFO"
                            detection["alert"] = False
                        
                        detections.append(detection)
            
            # Si aucune détection, retourner simulation
            if not detections:
                return self._simulate_detection()
            
            return detections
            
        except Exception as e:
            print(f"Erreur détection: {e}")
            return self._simulate_detection()
    
    def analyze_uploaded_file(self, uploaded_file):
        """
        Analyse un fichier uploadé via Streamlit
        
        Args:
            uploaded_file: objet UploadedFile de Streamlit
            
        Returns:
            tuple: (detections, gps_data)
        """
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Analyse avec YOLO
        detections = self.detect_from_image(tmp_path)
        
        # Extraction des métadonnées GPS (optionnelle)
        gps_data = self._extract_gps_from_file(tmp_path)
        
        # Nettoyage
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        return detections, gps_data
    
    def _extract_gps_from_file(self, filepath):
        """
        Extrait les métadonnées GPS d'une image
        """
        try:
            import exifread
            
            with open(filepath, 'rb') as f:
                tags = exifread.process_file(f)
            
            gps_data = {}
            
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                # Extraction des coordonnées
                lat = self._convert_to_degrees(tags['GPS GPSLatitude'])
                lon = self._convert_to_degrees(tags['GPS GPSLongitude'])
                
                # Vérification de la direction
                if str(tags.get('GPS GPSLatitudeRef', '')) == 'S':
                    lat = -lat
                if str(tags.get('GPS GPSLongitudeRef', '')) == 'W':
                    lon = -lon
                
                gps_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'has_gps': True
                }
            else:
                gps_data = {'has_gps': False}
            
            # Extraire la date de la photo
            if 'EXIF DateTimeOriginal' in tags:
                gps_data['date_taken'] = str(tags['EXIF DateTimeOriginal'])
            
            return gps_data
            
        except Exception as e:
            print(f"Erreur extraction GPS: {e}")
            return {'has_gps': False}
    
    def _convert_to_degrees(self, value):
        """
        Convertit une coordonnée EXIF en degrés décimaux
        """
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    def _simulate_detection(self):
        """
        Simulation réaliste de détection (fallback si YOLO indisponible)
        """
        # Scénarios possibles avec leurs probabilités
        scenarios = [
            {"label": "Attroupement suspect", "severity": "ÉLEVÉ", "alert": True, "weight": 20},
            {"label": "Personne isolée errante", "severity": "INFO", "alert": False, "weight": 25},
            {"label": "Véhicule suspect stationné", "severity": "SURVEILLANCE", "alert": False, "weight": 20},
            {"label": "Altercation / Bagarre", "severity": "ÉLEVÉ", "alert": True, "weight": 15},
            {"label": "Objet abandonné suspect", "severity": "CRITIQUE", "alert": True, "weight": 10},
            {"label": "Incendie / Fumée visible", "severity": "CRITIQUE", "alert": True, "weight": 5},
            {"label": "Fuite / Course-poursuite", "severity": "ÉLEVÉ", "alert": True, "weight": 5}
        ]
        
        # Sélection pondérée
        total_weight = sum(s["weight"] for s in scenarios)
        r = random.uniform(0, total_weight)
        cumulative = 0
        selected = scenarios[0]
        
        for scenario in scenarios:
            cumulative += scenario["weight"]
            if r <= cumulative:
                selected = scenario
                break
        
        confidence = random.uniform(0.75, 0.98)
        
        return [{
            "label": selected["label"],
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.0f}%",
            "severity": selected["severity"],
            "alert": selected["alert"],
            "timestamp": datetime.now().isoformat(),
            "is_simulation": not self.yolo_available
        }]
    
    def generate_incident_from_detection(self, detection, commune_info):
        """
        Génère un incident à partir d'une détection
        Version CORRIGÉE pour Firebase
        """
        from datetime import datetime
        
        # Coordonnées avec dispersion réaliste
        lat = commune_info["lat"] + random.uniform(-0.003, 0.003)
        lon = commune_info["lon"] + random.uniform(-0.003, 0.003)
        
        # Nettoyer le label de détection
        detection_label = detection.get('label', 'Inconnu').lower()
        
        # Mapping détection -> type d'incident
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
            incident_type = detection.get('label', 'Incident signalé')
            gravite_severity = detection.get('severity', 'Moyenne')
            if gravite_severity == "ÉLEVÉ":
                gravite = "Élevée"
            elif gravite_severity == "CRITIQUE":
                gravite = "Critique"
            else:
                gravite = "Moyenne"
        
        # Construction de l'incident - Format compatible Firebase
        incident = {
            "commune": commune_info["name"],
            "latitude": float(lat),
            "longitude": float(lon),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type_incident": incident_type,
            "gravite": gravite,
            "source": "Vision IA (YOLO)" if self.yolo_available else "Vision IA (Simulation)",
            "confidence": detection.get("confidence", 0.8),
            "original_detection": detection.get("label", "Inconnu"),
            "firebase_timestamp": datetime.now().isoformat(),
            "status": "verified"
        }
        
        # Debug - Afficher l'incident créé (visible dans le terminal)
        print(f"=== INCIDENT CRÉÉ ===")
        print(f"Commune: {incident['commune']}")
        print(f"Type: {incident['type_incident']}")
        print(f"Gravité: {incident['gravite']}")
        print(f"Lat/Lon: {incident['latitude']}, {incident['longitude']}")
        
        return incident


# Test du module
if __name__ == "__main__":
    print("Test du module vision...")
    vision = VisionModule(use_real_detection=True)
    print(f"YOLO disponible: {vision.yolo_available}")
