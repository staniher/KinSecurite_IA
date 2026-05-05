"""
APPLICATION PRINCIPALE KINSÉCURITÉ IA
Version complète avec YOLO, upload d'images, carte interactive, et Firebase
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import tempfile
from PIL import Image
import random

# Configuration de la page
st.set_page_config(
    page_title="KinSécurité IA - Surveillance Urbaine Kinshasa",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# COMMUNES DE KINSHASA
# ============================================================================
COMMUNES = {
    "Bandalungwa": {"lat": -4.341848, "lon": 15.283361},
    "Barumbu": {"lat": -4.318979, "lon": 15.325618},
    "Bumbu": {"lat": -4.370135, "lon": 15.294240},
    "Gombe": {"lat": -4.303056, "lon": 15.303333},
    "Kalamu": {"lat": -4.341800, "lon": 15.318700},
    "Kasa-Vubu": {"lat": -4.338800, "lon": 15.303200},
    "Kimbanseke": {"lat": -4.441940, "lon": 15.395000},
    "Kinshasa": {"lat": -4.323330, "lon": 15.308060},
    "Kintambo": {"lat": -4.326983, "lon": 15.272884},
    "Kisenso": {"lat": -4.409440, "lon": 15.342500},
    "Lemba": {"lat": -4.405769, "lon": 15.316123},
    "Limete": {"lat": -4.374389, "lon": 15.345417},
    "Lingwala": {"lat": -4.320280, "lon": 15.298330},
    "Makala": {"lat": -4.379788, "lon": 15.309706},
    "Maluku": {"lat": -4.073060, "lon": 15.537500},
    "Masina": {"lat": -4.383610, "lon": 15.391390},
    "Matete": {"lat": -4.388890, "lon": 15.351670},
    "Mont-Ngafula": {"lat": -4.455893, "lon": 15.228310},
    "N'Djili": {"lat": -4.385750, "lon": 15.444569},
    "Ngaba": {"lat": -4.376113, "lon": 15.319617},
    "Ngaliema": {"lat": -4.369733, "lon": 15.256448},
    "Ngiri-Ngiri": {"lat": -4.357500, "lon": 15.298330},
    "N'Sele": {"lat": -4.420400, "lon": 15.494700},
    "Selembao": {"lat": -4.371540, "lon": 15.284530}
}

# ============================================================================
# IMPORT DES MODULES
# ============================================================================
from clustering_module import ClusteringModule
from vision_module import VisionModule
from firebase_manager import FirebaseManager
from data_preprocessing import load_and_preprocess

# ============================================================================
# INITIALISATION DES MODULES
# ============================================================================
@st.cache_resource
def init_modules():
    """Initialise tous les modules avec mise en cache"""
    cm = ClusteringModule(eps=0.008, min_samples=5)
    vm = VisionModule(use_real_detection=True)
    fm = FirebaseManager(mock_mode=False)
    return cm, vm, fm

cm, vm, fm = init_modules()

# ============================================================================
# INITIALISATION DE SESSION STATE
# ============================================================================
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'detections' not in st.session_state:
    st.session_state['detections'] = []
if 'commune_choice' not in st.session_state:
    st.session_state['commune_choice'] = 'Gombe'

# ============================================================================
# AFFICHAGE STATUT FIREBASE (calculé pour la sidebar)
# ============================================================================
if not fm.mock_mode:
    try:
        # Compter les incidents dans Firebase
        fb_incidents_list = fm.get_incidents(limit=500)
        fb_count = len(fb_incidents_list)
        firebase_status = f"☁️ Cloud: {fb_count} incidents synchronisés"
        firebase_color = "success"
        
        # Debug - Afficher dans le terminal
        print(f"Firebase: {fb_count} incidents trouvés")
        for inc in fb_incidents_list[:3]:
            print(f"   - {inc.get('commune')}: {inc.get('type_incident')}")
            
    except Exception as e:
        firebase_status = f"☁️ Cloud: Erreur de connexion"
        firebase_color = "error"
        fb_incidents_list = []
        fb_count = 0
else:
    firebase_status = "☁️ Mode MOCK - Données locales uniquement"
    firebase_color = "warning"
    fb_incidents_list = []
    fb_count = 0

# ============================================================================
# CHARGEMENT DES DONNÉES (CSV + Firebase)
# ============================================================================
@st.cache_data
def load_data():
    """Charge les données depuis le CSV + Firebase"""
    # Charger CSV
    df = load_and_preprocess("incidents_kinshasa_30000.csv")
    
    # Charger les incidents Firebase
    fb_incidents = fm.get_incidents(limit=500)
    if fb_incidents:
        df_fb = pd.DataFrame(fb_incidents)
        print(f"{len(df_fb)} incidents chargés depuis Firebase")
        if not df.empty:
            df = pd.concat([df, df_fb], ignore_index=True)
        else:
            df = df_fb
    
    # Générer des données de démonstration si vide
    if df.empty:
        from data_preprocessing import generate_demo_data
        df = generate_demo_data(2000)
    
    return df, fb_incidents_list

df_total, firebase_incidents = load_data()

if df_total.empty:
    st.error("Aucune donnée disponible")
    st.stop()

# ============================================================================
# SIDEBAR - MENU DE NAVIGATION
# ============================================================================
st.sidebar.title("🚔 KinSécurité IA")
st.sidebar.markdown(f"📊 **{len(df_total):,} incidents**")

# Statut YOLO
if vm.yolo_available:
    st.sidebar.success("✅ YOLOv8 actif - Détection réelle")
else:
    st.sidebar.warning("⚠️ YOLO non disponible - Mode simulation")
    st.sidebar.code("pip install ultralytics", language="bash")

st.sidebar.markdown("---")

# ===== AFFICHAGE FIREBASE DANS LA SIDEBAR =====
if firebase_color == "success":
    st.sidebar.success(firebase_status)
elif firebase_color == "warning":
    st.sidebar.warning(firebase_status)
else:
    st.sidebar.error(firebase_status)
# ============================================

st.sidebar.markdown("---")

# Menu principal
menu = st.sidebar.radio(
    "📱 Navigation",
    ["🏠 Tableau de bord", "🗺️ Carte interactive", "👁️ Vision IA", "📊 Analyses"],
    key="main_menu_radio"
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 KinSécurité IA")
st.sidebar.caption(f"Dernière MAJ: {datetime.now().strftime('%d/%m/%Y')}")

# ============================================================================
# FILTRES POUR LE TABLEAU DE BORD ET LA CARTE
# ============================================================================
if menu in ["🏠 Tableau de bord", "🗺️ Carte interactive"]:
    st.sidebar.markdown("### 🔍 Filtres")
    
    # Filtre par commune
    if 'commune' in df_total.columns:
        communes_list = ["Toutes"] + sorted(df_total['commune'].dropna().unique().tolist())
        selected_commune = st.sidebar.selectbox(
            "📍 Commune",
            communes_list,
            key="filter_commune"
        )
    else:
        selected_commune = "Toutes"
    
    # Filtre par type d'incident
    if 'type_incident' in df_total.columns:
        types_list = df_total['type_incident'].dropna().unique().tolist()
        selected_types = st.sidebar.multiselect(
            "📋 Type d'incident",
            types_list,
            default=[],
            key="filter_types"
        )
    else:
        selected_types = []
    
    # Filtre par gravité
    if 'gravite' in df_total.columns:
        gravite_list = df_total['gravite'].dropna().unique().tolist()
        selected_gravite = st.sidebar.multiselect(
            "⚠️ Gravité",
            gravite_list,
            default=[],
            key="filter_gravite"
        )
    else:
        selected_gravite = []
    
    # Application des filtres
    df_filtered = df_total.copy()
    
    if selected_commune != "Toutes" and 'commune' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['commune'] == selected_commune]
    
    if selected_types and 'type_incident' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['type_incident'].isin(selected_types)]
    
    if selected_gravite and 'gravite' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['gravite'].isin(selected_gravite)]
else:
    df_filtered = df_total

# ============================================================================
# CLUSTERING (pour toutes les pages)
# ============================================================================
if not df_filtered.empty:
    df_clustered = cm.run_clustering(df_filtered)
    cluster_stats = cm.get_cluster_statistics(df_clustered)
    clustering_metrics = cm.evaluate_model(df_clustered)
    commune_ranking = cm.classify_communes(df_filtered)
    spearman_corr, spearman_p = cm.calculate_spearman(commune_ranking)
else:
    cluster_stats = {'n_clusters': 0, 'n_noise': 0, 'noise_percentage': 0, 'clusters_detail': []}
    clustering_metrics = {'silhouette': 0.0, 'silhouette_interpretation': 'Non calculé'}
    commune_ranking = pd.DataFrame()
    spearman_corr, spearman_p = 0.0, 1.0

# ============================================================================
# PAGE: TABLEAU DE BORD
# ============================================================================
if menu == "🏠 Tableau de bord":
    st.title("🚔 KinSécurité IA")
    st.markdown("*Système de surveillance urbaine assisté par IA pour Kinshasa*")
    st.markdown("---")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Incidents", f"{len(df_filtered):,}")
    
    with col2:
        unique_communes = df_filtered['commune'].nunique() if 'commune' in df_filtered.columns else 0
        st.metric("📍 Communes touchées", unique_communes)
    
    with col3:
        types_count = df_filtered['type_incident'].nunique() if 'type_incident' in df_filtered.columns else 0
        st.metric("📋 Types d'incidents", types_count)
    
    with col4:
        clusters_count = cluster_stats['n_clusters']
        st.metric("🎯 Zones à risque", clusters_count)
    
    st.markdown("---")
    
    # Deux colonnes principales
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Évolution temporelle")
        if 'date' in df_filtered.columns and not df_filtered['date'].isna().all():
            df_filtered['mois'] = pd.to_datetime(df_filtered['date']).dt.to_period('M').astype(str)
            monthly = df_filtered.groupby('mois').size().tail(12)
            
            fig = px.line(
                x=monthly.index,
                y=monthly.values,
                markers=True,
                title="Nombre d'incidents par mois"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données temporelles non disponibles")
    
    with col_right:
        st.subheader("📋 Types d'incidents")
        if 'type_incident' in df_filtered.columns:
            types = df_filtered['type_incident'].value_counts().head(8)
            fig = px.pie(
                values=types.values,
                names=types.index,
                hole=0.4,
                title="Répartition"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Données de type non disponibles")
    
    st.markdown("---")
    
    # Classement des communes
    st.subheader("🏆 Classement des communes par niveau d'insécurité")
    
    if not commune_ranking.empty:
        display_df = commune_ranking[['commune', 'count', 'incidents_mensuels', 'niveau_insecurite']].head(10)
        st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Derniers incidents
    st.subheader("🕐 Derniers incidents signalés")
    if 'date' in df_filtered.columns:
        latest = df_filtered.sort_values('date', ascending=False).head(10)
        cols_to_display = ['date', 'commune', 'type_incident']
        if 'gravite' in latest.columns:
            cols_to_display.append('gravite')
        st.dataframe(latest[cols_to_display], use_container_width=True)

# ============================================================================
# PAGE: CARTE INTERACTIVE
# ============================================================================
elif menu == "🗺️ Carte interactive":
    st.title("🗺️ Carte interactive des incidents")
    st.markdown("*Visualisation spatiale des incidents à Kinshasa*")
    st.markdown("---")
    
    # Vérification des données
    if df_filtered.empty or df_filtered[['latitude', 'longitude']].dropna().empty:
        st.error("Aucune donnée géographique valide à afficher")
    else:
        # Info sur le nombre d'incidents
        valid_count = df_filtered[['latitude', 'longitude']].dropna().shape[0]
        st.info(f"📍 **{len(df_filtered):,} incidents** au total - **{valid_count:,}** avec coordonnées valides")
        
        # Création de la carte
        try:
            # Création de la carte de base
            m = folium.Map(
                location=[-4.35, 15.30],
                zoom_start=11,
                control_scale=True,
                tiles='CartoDB positron'
            )
            
            # Heatmap (carte de chaleur)
            heat_data = df_filtered[['latitude', 'longitude']].dropna().values.tolist()
            if heat_data and len(heat_data) > 10:
                HeatMap(
                    heat_data,
                    radius=15,
                    blur=10,
                    min_opacity=0.4,
                    gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
                ).add_to(m)
            
            # Marqueurs des incidents (limités à 1500 pour performance)
            marker_cluster = MarkerCluster(name="Incidents").add_to(m)
            markers_count = 0
            
            for _, row in df_filtered.head(1500).iterrows():
                if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                    continue
                
                # Couleur selon gravité
                gravite = str(row.get('gravite', 'Moyenne'))
                if 'Critique' in gravite:
                    color = '#dc2626'
                elif 'Élevée' in gravite:
                    color = '#f97316'
                elif 'Moyenne' in gravite:
                    color = '#eab308'
                else:
                    color = '#22c55e'
                
                # Popup avec informations
                popup_html = f"""
                <div style="font-family: monospace; font-size: 12px; min-width: 150px;">
                    <b>🚨 {row.get('type_incident', 'N/A')}</b><br>
                    📍 Commune: {row.get('commune', 'N/A')}<br>
                    ⚠️ Gravité: {gravite}<br>
                    📅 Date: {str(row.get('date', 'N/A'))[:16]}
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=250)
                ).add_to(marker_cluster)
                markers_count += 1
            
            # Ajout des cercles de clusters DBSCAN (si disponibles)
            if cluster_stats and cluster_stats.get('clusters_detail'):
                try:
                    colors = ['#dc2626', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']
                    for idx, cluster in enumerate(cluster_stats['clusters_detail'][:10]):
                        color = colors[idx % len(colors)]
                        folium.Circle(
                            location=[cluster['center_lat'], cluster['center_lon']],
                            radius=cluster['radius_km'] * 1000,
                            color=color,
                            fill=True,
                            fill_opacity=0.1,
                            weight=2,
                            popup=f"🎯 Cluster #{cluster['cluster_id']}<br>📊 {cluster['size']} incidents"
                        ).add_to(m)
                except Exception as e:
                    pass  # Ignorer les erreurs de clusters
            
            # Légende
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
            
            # Contrôle des couches
            folium.LayerControl().add_to(m)
            
            # Affichage de la carte
            folium_static(m, width=900, height=600)
            
            st.caption(f"📌 **{markers_count}** marqueurs affichés sur la carte")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la création de la carte: {e}")
            st.info("Vérifiez que les données contiennent des coordonnées valides (latitude entre -4.60 et -4.00, longitude entre 15.10 et 15.60)")

# ============================================================================
# PAGE: VISION IA (AVEC UPLOAD D'IMAGES) - VERSION CORRIGÉE
# ============================================================================
elif menu == "👁️ Vision IA":
    st.title("👁️ Vision par Ordinateur - Détection d'incidents")
    st.markdown("*Analysez des images avec intelligence artificielle pour détecter des situations suspectes*")
    st.markdown("---")
    
    # Section d'information
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
    
    # SECTION UPLOAD D'IMAGE
    st.subheader("📸 Téléchargez une image à analyser")
    
    # Widget d'upload
    uploaded_file = st.file_uploader(
        "Choisir une image (JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Formats acceptés: JPG, JPEG, PNG. Taille max: 10MB",
        key="vision_upload_main"
    )
    
    if uploaded_file is not None:
        # Chargement et affichage de l'image
        image = Image.open(uploaded_file)
        
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="📷 Image à analyser", width=400)
        
        with col_info:
            st.markdown(f"**Fichier:** {uploaded_file.name}")
            st.markdown(f"**Taille:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
            
            # Sélection de la commune
            commune_choice = st.selectbox(
                "📍 Sélectionner la commune de l'incident",
                list(COMMUNES.keys()),
                key="vision_commune"
            )
            
            # Bouton d'analyse
            if st.button("🚀 ANALYSER L'IMAGE AVEC YOLO", type="primary", use_container_width=True, key="analyze_btn_main"):
                
                # Sauvegarde temporaire
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Analyse YOLO
                with st.spinner("🔍 YOLOv8 analyse l'image en cours..."):
                    detections = vm.detect_from_image(tmp_path)
                
                # Nettoyage
                os.unlink(tmp_path)
                
                # Stocker les détections dans session_state
                st.session_state['detections'] = detections
                st.session_state['commune_choice'] = commune_choice
                st.session_state['analysis_done'] = True
                
                st.rerun()
    
    # ================================================================
    # AFFICHAGE DES RÉSULTATS (en dehors du bloc uploaded_file)
    # ================================================================
    if st.session_state.get('analysis_done', False):
        detections = st.session_state.get('detections', [])
        commune_choice = st.session_state.get('commune_choice', 'Gombe')
        
        st.markdown("---")
        st.subheader("📊 RÉSULTATS DE L'ANALYSE")
        
        if detections:
            # Alertes critiques
            alerts = [d for d in detections if d.get('alert', False)]
            if alerts:
                for alert in alerts:
                    st.error(f"🚨 **ALERTE {alert.get('severity', 'CRITIQUE')}** : {alert['label']} (confiance: {alert.get('confidence_pct', 'N/A')})")
                st.balloons()
            
            # Détails des détections
            st.markdown("**📋 Objets détectés:**")
            for d in detections:
                if not d.get('alert', False):
                    st.write(f"- **{d['label']}** : {d.get('confidence_pct', 'N/A')} de confiance")
            
            # Indicateur YOLO
            if vm.yolo_available:
                st.success("✅ Analyse réalisée avec YOLOv8 (détection réelle)")
            else:
                st.info("ℹ️ Mode simulation - YOLO non installé")
            
            # ============================================================
            # FORMULAIRE DE CRÉATION D'INCIDENT (CORRIGÉ)
            # ============================================================
            st.markdown("---")
            st.subheader("📝 Créer un incident à partir de cette analyse")
            
            # Sélectionner la détection à utiliser
            detection_options = [f"{d['label']} ({d.get('confidence_pct', 'N/A')})" for d in detections]
            selected_idx = st.selectbox(
                "Sélectionner la détection à convertir en incident",
                range(len(detection_options)),
                format_func=lambda x: detection_options[x],
                key="detection_select"
            )
            
            selected_detection = detections[selected_idx]
            
            # Afficher un résumé
            st.info(f"📌 **Incident à créer:** {selected_detection['label']} (confiance: {selected_detection.get('confidence_pct', 'N/A')})")
            
            # Bouton de création - VERSION CORRIGÉE
            if st.button("✅ VALIDER ET CRÉER L'INCIDENT", type="primary", use_container_width=True, key="final_create_btn"):
                
                # Récupérer les informations de la commune
                # commune_choice contient le NOM de la commune (string)
                commune_name = commune_choice
                commune_coords = COMMUNES[commune_name]
                
                # Construction de l'incident
                lat = commune_coords["lat"] + random.uniform(-0.003, 0.003)
                lon = commune_coords["lon"] + random.uniform(-0.003, 0.003)
                
                detection_label = selected_detection.get('label', 'Inconnu').lower()
                
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
                
                incident = {
                    "commune": commune_name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type_incident": incident_type,
                    "gravite": gravite,
                    "source": "Vision IA (Interface)",
                    "confidence": selected_detection.get("confidence", 0.8),
                    "original_detection": selected_detection.get("label", "Inconnu"),
                    "firebase_timestamp": datetime.now().isoformat(),
                    "status": "verified"
                }
                
                # Debug
                st.write("**🔍 Incident envoyé à Firebase:**")
                st.json(incident)
                
                # Envoi à Firebase
                result = fm.push_incident(incident)
                st.write(f"**📡 Résultat de l'envoi:** {'✅ Réussi' if result else '❌ Échoué'}")
                
                if result:
                    st.success(f"✅ Incident créé pour {commune_name} !")
                    st.balloons()
                    
                    # Vider le cache pour forcer le rechargement
                    st.cache_data.clear()
                    
                    # Proposition de nouvelle analyse
                    if st.button("🔄 ANALYSER UNE NOUVELLE IMAGE", key="new_analysis_btn"):
                        st.session_state['analysis_done'] = False
                        st.session_state['detections'] = []
                        st.rerun()
                else:
                    st.error("❌ Erreur lors de la création")
        
        else:
            st.warning("⚠️ Aucun objet pertinent détecté dans cette image")
            if st.button("🔄 RÉESSAYER AVEC UNE AUTRE IMAGE", key="retry_btn"):
                st.session_state['analysis_done'] = False
                st.rerun()
    
    else:
        # Message quand aucune image n'est uploadée
        st.info("👆 **Téléchargez une image ci-dessus pour commencer l'analyse**")
        
        # Exemples d'images suggérées
        with st.expander("📖 Exemples d'images à analyser", expanded=False):
            st.markdown("""
            - Photo d'un **attroupement** dans un espace public
            - Image d'un **véhicule suspect** stationné
            - Photo d'une **altercation** ou bagarre
            - Image d'un **objet abandonné** suspect
            - Photo d'un **incendie** ou fumée
            """)
    
    st.markdown("---")
    
    # Section d'information sur YOLO
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
# PAGE: ANALYSES STATISTIQUES
# ============================================================================
elif menu == "📊 Analyses":
    st.title("📊 Analyses statistiques")
    st.markdown("*Indicateurs clés et tendances de la sécurité*")
    st.markdown("---")
    
    # Métriques du clustering
    st.subheader("🎯 Analyse spatiale - Clustering DBSCAN")
    
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
    
    # Détail des clusters
    if cluster_stats['clusters_detail']:
        st.subheader("📋 Détail des clusters détectés")
        
        for cluster in cluster_stats['clusters_detail'][:5]:
            with st.expander(f"🎯 Cluster #{cluster['cluster_id']} - {cluster['size']} incidents ({cluster['percentage']:.1f}%)"):
                st.write(f"- **Centre géographique:** {cluster['center_lat']:.5f}, {cluster['center_lon']:.5f}")
                st.write(f"- **Rayon approximatif:** {cluster['radius_km']:.2f} km")
                st.write(f"- **Type principal:** {cluster['main_incident_type']}")
                st.write(f"- **Gravité dominante:** {cluster['main_gravite']}")
    
    st.markdown("---")
    
    # Classement des communes
    st.subheader("🏆 Classement des communes")
    
    if not commune_ranking.empty:
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
    
    # Corrélation de Spearman
    st.subheader("📊 Validation du modèle")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.metric("Corrélation de Spearman", f"{spearman_corr:.3f}")
        st.caption(f"p-value: {spearman_p:.4f}")
    
    with col_v2:
        if spearman_corr > 0.5:
            st.success("✅ Forte corrélation avec les risques officiels")
        elif spearman_corr > 0.3:
            st.info("⚠️ Corrélation modérée avec les risques officiels")
        else:
            st.warning("❌ Faible corrélation - données à enrichir")
    
    st.markdown("---")
    
    # Matrice de corrélation des types d'incidents
    if 'type_incident' in df_filtered.columns and 'commune' in df_filtered.columns:
        st.subheader("📊 Distribution des types d'incidents par commune")
        
        pivot = pd.crosstab(df_filtered['commune'], df_filtered['type_incident'])
        st.dataframe(pivot, use_container_width=True)
    
    # ===== SECTION DEBUG FIREBASE =====
    st.markdown("---")
    with st.expander("🔧 DEBUG - Voir les incidents Firebase"):
        if st.button("Afficher les incidents Firebase", key="debug_firebase_btn"):
            fb_incidents = fm.get_incidents(limit=20)
            if fb_incidents:
                st.write(f"**{len(fb_incidents)} incidents dans Firebase:**")
                for inc in fb_incidents:
                    st.write(f"- {inc.get('commune')}: {inc.get('type_incident')} ({inc.get('source')})")
            else:
                st.write("Aucun incident dans Firebase")

# ============================================================================
# FIN DE L'APPLICATION# ============================================================================