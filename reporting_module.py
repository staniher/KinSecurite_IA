"""
MODULE DE SIGNALEMENT CITOYEN
Permet aux utilisateurs de signaler des incidents avec preuves
Système de vérification pour lutter contre la désinformation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import os
import tempfile
from PIL import Image, ImageOps
import exifread


class CitizenReportingModule:
    """
    Module de signalement citoyen avec vérification anti-désinformation
    """
    
    def __init__(self, firebase_manager, vision_module):
        """
        Initialise le module de signalement
        """
        self.fm = firebase_manager
        self.vm = vision_module
        self.reports_collection = "citizen_reports"
    
    def create_report_form(self):
        """
        Affiche le formulaire de signalement citoyen
        """
        st.markdown("### 📝 Nouveau signalement citoyen")
        st.markdown("Remplissez ce formulaire pour signaler un incident. Les signalements avec preuves sont prioritaires.")
        
        # Informations sur l'incident
        st.markdown("---")
        st.markdown("#### 📋 Informations sur l'incident")
        
        incident_types = [
            "Vol à main armée", "Agression physique", "Violence domestique",
            "Violence sexuelle", "Attroupement suspect", "Trafic de drogue",
            "Infraction routière", "Accident de la route", "Incendie",
            "Vandalisme", "Enlèvement", "Extorsion", "Autre"
        ]
        
        type_incident = st.selectbox("Type d'incident *", incident_types)
        
        if type_incident == "Autre":
            custom_type = st.text_input("Précisez le type d'incident")
            if custom_type:
                type_incident = custom_type
        
        description = st.text_area(
            "Description détaillée *",
            height=150,
            placeholder="Décrivez précisément ce qui s'est passé, les personnes impliquées, l'heure approximative..."
        )
        
        gravite = st.select_slider(
            "Gravité perçue *",
            options=["Très faible", "Faible", "Modérée", "Élevée", "Critique"],
            value="Modérée"
        )
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            incident_date = st.date_input("Date de l'incident *", datetime.now())
        with col_date2:
            incident_time = st.time_input("Heure approximative", datetime.now().time())
        
        incident_datetime = datetime.combine(incident_date, incident_time)
        
        # Localisation
        st.markdown("---")
        st.markdown("#### 📍 Localisation")
        
        col_loc1, col_loc2 = st.columns(2)
        
        with col_loc1:
            # Coordonnées manuelles
            latitude = st.number_input("Latitude *", value=-4.35, format="%.6f")
            longitude = st.number_input("Longitude *", value=15.30, format="%.6f")
        
        with col_loc2:
            # Sélection commune
            from app import COMMUNES
            communes_list = list(COMMUNES.keys())
            commune = st.selectbox("Commune *", communes_list)
            
            # Vérification de cohérence
            min_dist = float('inf')
            for com_name, com_coords in COMMUNES.items():
                dist = self._calculate_distance(latitude, longitude, com_coords['lat'], com_coords['lon'])
                if dist < min_dist:
                    min_dist = dist
                    closest = com_name
            
            if min_dist < 5:
                st.info(f"📍 Distance de la commune: {min_dist:.1f} km")
            else:
                st.warning(f"⚠️ Les coordonnées sont éloignées de la commune sélectionnée")
        
        adresse = st.text_input("Adresse ou point de repère (optionnel)", 
                                placeholder="Ex: Avenue du Commerce, près du marché de Limete")
        
        # Preuves visuelles
        st.markdown("---")
        st.markdown("#### 📸 Preuves visuelles (obligatoires)")
        st.markdown("*Les signalements avec photos sont plus fiables et vérifiés plus rapidement*")
        
        uploaded_images = st.file_uploader(
            "Téléchargez des photos de l'incident (max 5)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_images and len(uploaded_images) > 5:
            st.warning("Maximum 5 images. Seules les 5 premières seront conservées")
            uploaded_images = uploaded_images[:5]
        
        # Analyse automatique des images
        image_analysis_results = []
        if uploaded_images:
            st.markdown("**Analyse automatique des images en cours...**")
            progress_bar = st.progress(0)
            
            for idx, img_file in enumerate(uploaded_images):
                progress_bar.progress((idx + 1) / len(uploaded_images))
                
                # Analyse YOLO
                detections, gps_data = self.vm.analyze_uploaded_file(img_file)
                
                image_analysis_results.append({
                    'filename': img_file.name,
                    'detections': detections[:3] if detections else [],
                    'gps_data': gps_data,
                    'has_metadata': gps_data is not None
                })
            
            progress_bar.empty()
            st.success(f"✅ {len(uploaded_images)} images analysées")
            
            for result in image_analysis_results:
                if result['has_metadata']:
                    st.caption(f"📸 {result['filename']}: géolocalisée")
                if result['detections']:
                    det_str = ", ".join([d['label'] for d in result['detections'][:2]])
                    st.caption(f"🔍 Détections: {det_str}")
        
        # Identité du signalant
        st.markdown("---")
        st.markdown("#### 👤 Votre identité (confidentiel)")
        
        col_id1, col_id2 = st.columns(2)
        with col_id1:
            pseudo = st.text_input("Pseudonyme (optionnel)", placeholder="Anonyme")
            phone = st.text_input("Téléphone (optionnel)")
        with col_id2:
            email = st.text_input("Email (optionnel)")
        
        consent = st.checkbox(
            "Je certifie que les informations fournies sont exactes *",
            help="Les signalements frauduleux peuvent être poursuivis"
        )
        
        # Soumission
        st.markdown("---")
        submitted = st.button("📢 SIGNALER L'INCIDENT", type="primary", use_container_width=True)
        
        if submitted:
            errors = []
            if not description:
                errors.append("La description est obligatoire")
            if not commune:
                errors.append("La commune est obligatoire")
            if not consent:
                errors.append("Vous devez certifier l'exactitude des informations")
            if not uploaded_images:
                errors.append("Au moins une image est requise")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                report_id = self._generate_report_id()
                image_urls = self._save_images(uploaded_images, report_id)
                
                report = {
                    'report_id': report_id,
                    'type_incident': type_incident,
                    'description': description,
                    'gravite': gravite,
                    'date_incident': incident_datetime.isoformat(),
                    'date_signalement': datetime.now().isoformat(),
                    'latitude': latitude,
                    'longitude': longitude,
                    'commune': commune,
                    'adresse': adresse,
                    'image_urls': image_urls,
                    'image_analysis': image_analysis_results,
                    'pseudo': pseudo if pseudo else "Anonyme",
                    'contact_phone': phone,
                    'contact_email': email,
                    'status': 'pending_verification',
                    'verification_score': self._calculate_trust_score(latitude, longitude, uploaded_images, description),
                    'views_count': 0,
                    'is_verified': False
                }
                
                if self.fm.push_to_collection(self.reports_collection, report):
                    st.success("✅ Signalement enregistré avec succès !")
                    st.balloons()
                    st.info("📋 Votre signalement sera vérifié par nos modérateurs sous 24-48h")
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de l'enregistrement")
    
    def display_reports_list(self, status_filter=None):
        """
        Affiche la liste des signalements citoyens
        """
        reports = self.fm.get_from_collection(self.reports_collection, limit=100)
        
        if not reports:
            st.info("Aucun signalement citoyen pour le moment")
            return
        
        if status_filter:
            reports = [r for r in reports if r.get('status') == status_filter]
        
        st.markdown(f"### 📋 Signalements citoyens ({len(reports)})")
        
        for report in reports[:20]:
            with st.container():
                is_verified = report.get('is_verified', False)
                status_color = "#22c55e" if is_verified else "#eab308"
                
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span style="font-size: 18px; font-weight: bold;">🚨 {report.get('type_incident', 'Incident')}</span>
                            <span style="margin-left: 10px; padding: 2px 8px; border-radius: 20px; 
                                 background-color: {status_color}; color: white; font-size: 12px;">
                                {report.get('status', 'inconnu')}
                            </span>
                        </div>
                        <div style="font-size: 12px; color: #666;">
                            🏷️ Score: {report.get('verification_score', 0)}%
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>📝 Description:</strong> {report.get('description', '')[:200]}...
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 20px; font-size: 13px; color: #555;">
                        <span>📍 {report.get('commune', 'N/A')}</span>
                        <span>⚠️ {report.get('gravite', 'N/A')}</span>
                        <span>📅 {report.get('date_incident', '')[:10]}</span>
                        <span>📸 {len(report.get('image_urls', []))} preuve(s)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Voir détails", key=f"view_{report.get('report_id')}"):
                    self._show_report_details(report)
    
    def _show_report_details(self, report):
        """
        Affiche les détails complets d'un signalement
        """
        st.markdown("---")
        st.markdown(f"## 📋 Détails du signalement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informations générales**")
            st.write(f"📍 **Commune:** {report.get('commune', 'N/A')}")
            st.write(f"📌 **Type:** {report.get('type_incident', 'N/A')}")
            st.write(f"⚠️ **Gravité:** {report.get('gravite', 'N/A')}")
            st.write(f"📅 **Date:** {report.get('date_incident', 'N/A')[:16]}")
        
        with col2:
            st.markdown("**Vérification**")
            st.write(f"🏷️ **Score:** {report.get('verification_score', 0)}%")
            st.write(f"📊 **Statut:** {report.get('status', 'N/A')}")
            st.write(f"👤 **Signalé par:** {report.get('pseudo', 'Anonyme')}")
        
        st.markdown("**Description complète**")
        st.write(report.get('description', 'Pas de description'))
        
        if report.get('image_urls'):
            st.markdown("**Preuves visuelles**")
            cols = st.columns(min(3, len(report['image_urls'])))
            for idx, url in enumerate(report['image_urls'][:3]):
                if os.path.exists(url):
                    with cols[idx]:
                        st.image(url, use_column_width=True)
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calcule la distance entre deux points en km"""
        from math import radians, sin, cos, sqrt, atan2
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    def _calculate_trust_score(self, latitude, longitude, images, description):
        """Calcule un score de confiance pour le signalement"""
        score = 0
        
        # Coordonnées valides
        if -4.5 <= latitude <= -4.0 and 15.1 <= longitude <= 15.6:
            score += 20
        
        # Images
        if images:
            score += min(20, len(images) * 5)
        
        # Description détaillée
        if len(description) > 100:
            score += 15
        elif len(description) > 50:
            score += 10
        
        return min(100, score)
    
    def _generate_report_id(self):
        """Génère un ID unique pour le signalement"""
        import uuid
        return f"REP-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
    
    def _save_images(self, images, report_id):
        """Sauvegarde les images localement"""
        image_urls = []
        os.makedirs("reports_images", exist_ok=True)
        
        for idx, img in enumerate(images):
            filepath = f"reports_images/{report_id}_{idx}.jpg"
            with open(filepath, "wb") as f:
                f.write(img.getvalue())
            image_urls.append(filepath)
        
        return image_urls


class AdminVerificationModule:
    """
    Module d'administration pour vérifier les signalements
    """
    
    def __init__(self, firebase_manager):
        self.fm = firebase_manager
        self.reports_collection = "citizen_reports"
    
    def admin_panel(self):
        """
        Panneau d'administration
        """
        password = st.text_input("Mot de passe administrateur", type="password")
        
        if password != "kinshasa2024":
            st.warning("Accès restreint. Mot de passe requis.")
            return
        
        st.success("✅ Accès administrateur accordé")
        
        reports = self.fm.get_from_collection(self.reports_collection, limit=200)
        pending = [r for r in reports if r.get('status') == 'pending_verification']
        
        st.subheader(f"📋 Signalements à vérifier ({len(pending)})")
        
        if not pending:
            st.info("Aucun signalement en attente")
            return
        
        for report in pending:
            with st.expander(f"🚨 {report.get('type_incident')} - {report.get('commune')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {report.get('description', '')[:300]}")
                    st.write(f"**Gravité:** {report.get('gravite')}")
                    st.write(f"**Score:** {report.get('verification_score')}%")
                
                with col2:
                    if report.get('image_urls') and os.path.exists(report['image_urls'][0]):
                        st.image(report['image_urls'][0], use_column_width=True)
                    
                    col_accept, col_reject = st.columns(2)
                    with col_accept:
                        if st.button("✅ Valider", key=f"accept_{report.get('report_id')}"):
                            report['status'] = 'verified'
                            report['is_verified'] = True
                            report['verification_date'] = datetime.now().isoformat()
                            self.fm.update_document(self.reports_collection, report['report_id'], report)
                            st.success("Signalement validé")
                            st.rerun()
                    
                    with col_reject:
                        if st.button("❌ Rejeter", key=f"reject_{report.get('report_id')}"):
                            report['status'] = 'rejected'
                            self.fm.update_document(self.reports_collection, report['report_id'], report)
                            st.warning("Signalement rejeté")
                            st.rerun()


def add_reports_to_map(m, reports, COMMUNES):
    """
    Ajoute les signalements citoyens sur la carte
    """
    from folium.plugins import MarkerCluster
    
    report_cluster = MarkerCluster(name="Signalements citoyens").add_to(m)
    
    for report in reports:
        if not report.get('is_verified', False):
            continue
        
        lat = report.get('latitude')
        lon = report.get('longitude')
        
        if lat and lon:
            popup_html = f"""
            <div style="font-family: sans-serif; min-width: 200px;">
                <b>🚨 SIGNALEMENT CITOYEN</b><br>
                <b>Type:</b> {report.get('type_incident', 'N/A')}<br>
                <b>Commune:</b> {report.get('commune', 'N/A')}<br>
                <b>Gravité:</b> {report.get('gravite', 'N/A')}<br>
                <b>Fiabilité:</b> {report.get('verification_score', 0)}%<br>
                <i>{report.get('description', '')[:100]}...</i>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"📢 {report.get('type_incident')}",
                icon=folium.Icon(color='green', icon='info-sign', prefix='glyphicon')
            ).add_to(report_cluster)