import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from pdf_table_extractor_improved import extract_tables_and_graphs_from_image, clean_and_prepare_table
from dotenv import load_dotenv
import logging
import PyPDF2
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Vérifier que la clé API est définie
if not os.getenv("OPENAI_API_KEY"):
    st.error("La clé API OpenAI n'est pas définie. Veuillez vérifier votre fichier .env")
    st.stop()

# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("Extracteur de tableaux et graphiques PDF")

# Paramètres de l'application
model = st.sidebar.selectbox("Modèle GPT à utiliser", ["gpt-4o", "gpt-4o-mini"], index=0)
max_workers = st.sidebar.slider("Nombre de workers pour le traitement parallèle", 1, os.cpu_count(), 2)

# Initialisation du session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}
if 'pdf_document' not in st.session_state:
    st.session_state.pdf_document = None


def convert_pdf_to_image(page):
    try:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        logging.error(f"Erreur lors de la conversion de la page en image: {str(e)}")
        return None


def attempt_pdf_repair(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            writer = PyPDF2.PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                repaired_pdf_path = tmp_file.name
                with open(repaired_pdf_path, 'wb') as output_file:
                    writer.write(output_file)

        return repaired_pdf_path
    except Exception as e:
        logging.error(f"Erreur lors de la tentative de réparation du PDF: {e}")
        return None


def extract_data_from_single_page(pdf_doc, page_num, output_folder, model):
    try:
        page = pdf_doc[page_num - 1]
        image = convert_pdf_to_image(page)
        if image is None:
            raise ValueError("Impossible d'extraire le contenu de la page")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            image.save(temp_image.name, format="PNG")
            temp_image_path = temp_image.name

        extracted_data = extract_tables_and_graphs_from_image(temp_image_path, model)
        os.unlink(temp_image_path)
        return extracted_data
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction de la page {page_num}: {e}")
        logging.error(traceback.format_exc())
        st.error(f"Erreur lors de l'extraction de la page {page_num}: {e}")
        return {"metadata": {}, "tables": [], "graphs": []}


def update_session_state(data_key, df, data_structure, metadata, extraction_time, data_type):
    page_num = int(data_key.split('_')[1])
    idx = int(data_key.split('_')[3])
    page_key = f"page_{page_num}"

    if page_key not in st.session_state.extracted_data:
        st.session_state.extracted_data[page_key] = {"tables": {}, "graphs": {}}

    storage_type = data_type + 's' if not data_type.endswith('s') else data_type

    # Mettre à jour les métadonnées avec les propriétés globales
    metadata.update(st.session_state.global_properties)

    st.session_state.extracted_data[page_key][storage_type][idx] = {
        'df': df,
        'structure': data_structure,
        'metadata': metadata,
        'extraction_time': extraction_time
    }


def display_data(data, page_num, data_idx, metadata=None, extraction_time=None, data_type="table"):
    data_key = f"page_{page_num}_{data_type}_{data_idx}"
    editor_key = f"editor_{data_key}"
    download_key = f"download_{data_key}"
    structure_key = f"structure_button_{data_key}"

    st.subheader(f"{data_type.capitalize()} {data_idx + 1} de la page {page_num}")
    if extraction_time:
        st.write(f"Dernière modification : {extraction_time}")

    # Mettre à jour les métadonnées avec les propriétés globales
    metadata.update(st.session_state.global_properties)

    # Afficher les métadonnées
    st.write("Métadonnées:")
    st.json(metadata)

    # Afficher la structure sous forme de dictionnaire JSON
    if st.button(f"Voir la structure du {data_type}", key=structure_key):
        st.json(data)

    # Préparer le DataFrame
    if isinstance(data, dict):
        df = clean_and_prepare_table(data, metadata)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        st.warning(f"Format de données non reconnu pour ce {data_type}.")
        return

    if df.empty:
        st.warning(f"Le {data_type} extrait est vide ou n'a pas pu être correctement formaté.")
        return

    st.write(f"Données du {data_type}:")

    # Gestion des colonnes
    col1, col2 = st.columns([2, 1])
    with col1:
        new_column = st.text_input("Nom de la nouvelle colonne:", key=f"new_col_{data_key}")
        if st.button("Ajouter une colonne", key=f"add_col_{data_key}"):
            if new_column and new_column not in df.columns:
                df[new_column] = ""
                st.success(f"Colonne '{new_column}' ajoutée avec succès.")
            elif new_column in df.columns:
                st.warning(f"La colonne '{new_column}' existe déjà.")
            else:
                st.warning("Veuillez entrer un nom de colonne valide.")

    with col2:
        columns_to_remove = st.multiselect("Sélectionner les colonnes à supprimer:", df.columns,
                                           key=f"remove_cols_{data_key}")
        if st.button("Supprimer les colonnes sélectionnées", key=f"remove_col_{data_key}"):
            if columns_to_remove:
                df = df.drop(columns=columns_to_remove)
                st.success(f"Colonne(s) {', '.join(columns_to_remove)} supprimée(s) avec succès.")
            else:
                st.warning("Aucune colonne sélectionnée pour la suppression.")

    # Afficher l'éditeur de données
    try:
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=editor_key
        )
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de l'éditeur de données: {str(e)}")
        st.write(f"Affichage du {data_type} en lecture seule:")
        st.dataframe(df)
        edited_df = df  # Utiliser le DataFrame original si l'édition échoue

    # Mettre à jour le DataFrame dans le session state si des modifications ont été apportées
    if not df.equals(edited_df):
        update_session_state(data_key, edited_df, data, metadata, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             data_type)

    # Préparer le nom du fichier pour le téléchargement
    bank_name = metadata.get('bank_name', 'Banque_inconnue').replace(' ', '_')
    client_name = metadata.get('client_name', 'Client_inconnu').replace(' ', '_')
    portfolio_number = metadata.get('portfolio_number', 'Portfolio_inconnu').replace(' ', '_')
    statement_date = metadata.get('statement_date', 'Date_inconnue').replace('/', '-')
    extraction_date = datetime.now().strftime("%Y-%m-%d")

    file_name = f"{bank_name}_{client_name}_{portfolio_number}_page{page_num}_{data_type}{data_idx + 1}_{statement_date}_extrait_{extraction_date}.csv"

    # Nettoyer le nom du fichier pour éviter les caractères problématiques
    file_name = "".join(c for c in file_name if c.isalnum() or c in ['_', '-', '.'])

    # Bouton de téléchargement
    csv = edited_df.to_csv(index=False, encoding='utf-8-sig', sep=';').encode('utf-8-sig')
    st.download_button(
        label=f"Télécharger le {data_type} {data_idx + 1} de la page {page_num}",
        data=csv,
        file_name=file_name,
        mime="text/csv",
        key=download_key
    )


def display_global_properties():
    st.header("Propriétés globales du document")

    # Extraction automatique des propriétés globales à partir de la première page du PDF
    if 'pdf_document' in st.session_state and st.session_state.pdf_document:
        try:
            # Extraire les données de la première page pour obtenir les métadonnées
            first_page = st.session_state.pdf_document[0]
            image = convert_pdf_to_image(first_page)
            if image is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                    image.save(temp_image.name, format="PNG")
                    temp_image_path = temp_image.name

                # Utiliser le LLM pour extraire les métadonnées de la première page
                extracted_data = extract_tables_and_graphs_from_image(temp_image_path, model)
                os.unlink(temp_image_path)

                # Récupérer les métadonnées
                metadata = extracted_data.get("metadata", {})
                client_name = metadata.get('client_name', '')
                bank_name = metadata.get('bank_name', '')

                # Amélioration de l'identification de la date du relevé
                # Rechercher la date spécifique associée au relevé plutôt que d'autres dates possibles
                possible_dates = metadata.get('dates', [])  # Supposons que toutes les dates possibles sont listées
                statement_date = ''
                for date in possible_dates:
                    if 'janvier' in date.lower() or 'février' in date.lower() or 'mars' in date.lower() or \
                            'avril' in date.lower() or 'mai' in date.lower() or 'juin' in date.lower() or \
                            'juillet' in date.lower() or 'août' in date.lower() or 'septembre' in date.lower() or \
                            'octobre' in date.lower() or 'novembre' in date.lower() or 'décembre' in date.lower():
                        statement_date = date
                        break

            else:
                client_name, bank_name, statement_date = '', '', ''
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction des propriétés globales: {str(e)}")
            client_name, bank_name, statement_date = '', '', ''
    else:
        client_name, bank_name, statement_date = '', '', ''

    # Initialiser les propriétés globales dans le session state à chaque changement de document PDF
    st.session_state.global_properties = {
        'client_name': client_name,
        'bank_name': bank_name,
        'statement_date': statement_date
    }

    # Afficher les propriétés globales et permettre la modification
    col1, col2, col3 = st.columns(3)

    with col1:
        client_name = st.text_input("Nom du client", st.session_state.global_properties['client_name'])
    with col2:
        bank_name = st.text_input("Nom de la banque", st.session_state.global_properties['bank_name'])
    with col3:
        statement_date = st.text_input("Date de l'extrait", st.session_state.global_properties['statement_date'])

    # Bouton pour mettre à jour les propriétés globales
    if st.button("Mettre à jour les propriétés globales"):
        st.session_state.global_properties = {
            'client_name': client_name,
            'bank_name': bank_name,
            'statement_date': statement_date
        }
        st.success("Propriétés globales mises à jour avec succès!")

    return st.session_state.global_properties

def display_page_and_results(page_num, pdf_document):
    st.markdown(f"### Page {page_num}")
    col1, col2 = st.columns([3, 2])

    with col1:
        try:
            page = pdf_document[page_num - 1]
            image = convert_pdf_to_image(page)
            if image:
                st.image(image, caption=f"Page {page_num}", use_column_width=True)
            else:
                st.warning(f"Impossible d'afficher la page {page_num}")
        except Exception as e:
            st.warning(f"Erreur lors de l'affichage de la page {page_num}: {str(e)}")

    with col2:
        # Bouton d'extraction pour chaque page
        if st.button(f"Extraire les données de la page {page_num}", key=f"extract_{page_num}"):
            with st.spinner(f'Extraction des données de la page {page_num} en cours...'):
                extracted_data = extract_data_from_single_page(pdf_document, page_num, output_folder, model)
                if extracted_data:
                    extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f'Extraction de la page {page_num} terminée !')

                    if extracted_data.get("tables"):
                        for idx, table in enumerate(extracted_data["tables"]):
                            update_session_state(f"page_{page_num}_table_{idx}", table, table,
                                                 extracted_data["metadata"], extraction_time, "tables")

                    if extracted_data.get("graphs"):
                        for idx, graph in enumerate(extracted_data["graphs"]):
                            update_session_state(f"page_{page_num}_graph_{idx}", graph, graph,
                                                 extracted_data["metadata"], extraction_time, "graphs")

                    if not extracted_data.get("tables") and not extracted_data.get("graphs"):
                        st.warning(f"Aucune donnée trouvée sur la page {page_num}")
                else:
                    st.warning(f"Aucune donnée extraite de la page {page_num}")

        # Afficher les données extraites pour cette page
        page_key = f"page_{page_num}"
        if page_key in st.session_state.extracted_data:
            for data_type in ["tables", "graphs"]:
                for data_idx, data in st.session_state.extracted_data[page_key][data_type].items():
                    display_data(
                        data['df'],
                        page_num,
                        data_idx,
                        data.get('metadata'),
                        data.get('extraction_time'),
                        data_type[:-1]  # Enlever le 's' pour avoir "table" ou "graph"
                    )


# Upload du fichier PDF
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # Créer un fichier temporaire pour stocker le PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # Tenter d'ouvrir le PDF
    try:
        st.session_state.pdf_document = fitz.open(pdf_path)
    except Exception as e:
        st.warning("Le PDF semble être corrompu. Tentative de réparation...")
        repaired_pdf_path = attempt_pdf_repair(pdf_path)
        if repaired_pdf_path:
            st.session_state.pdf_document = fitz.open(repaired_pdf_path)
            st.success("PDF réparé avec succès!")
        else:
            st.error("Impossible de réparer le PDF. Veuillez essayer avec un autre fichier.")
            st.stop()

    # Afficher et gérer les propriétés globales
    global_properties = display_global_properties()

    # Définir le dossier de sortie pour les CSV
    output_folder = os.path.join(os.getcwd(), 'Tests', 'output')
    os.makedirs(output_folder, exist_ok=True)

    # Sélection des pages
    num_pages = len(st.session_state.pdf_document)
    selected_pages = st.multiselect(
        "Sélectionnez les pages à traiter",
        options=list(range(1, num_pages + 1)),
        default=list(range(1, num_pages + 1))
    )

    # Bouton pour extraire toutes les pages sélectionnées
    if st.button("Extraire les données de toutes les pages sélectionnées"):
        with st.spinner('Extraction des données de toutes les pages sélectionnées en cours...'):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {executor.submit(extract_data_from_single_page, st.session_state.pdf_document, page_num, output_folder, model): page_num for page_num in selected_pages}
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        extracted_data = future.result()
                        if extracted_data:
                            extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if extracted_data.get("tables"):
                                for idx, table in enumerate(extracted_data["tables"]):
                                    update_session_state(f"page_{page_num}_table_{idx}", table, table, extracted_data["metadata"], extraction_time, "tables")
                            if extracted_data.get("graphs"):
                                for idx, graph in enumerate(extracted_data["graphs"]):
                                    update_session_state(f"page_{page_num}_graph_{idx}", graph, graph, extracted_data["metadata"], extraction_time, "graphs")
                    except Exception as e:
                        st.error(f"Erreur lors de l'extraction de la page {page_num}: {str(e)}")

        st.success('Extraction de toutes les pages terminée !')

    # Affichage des pages individuelles et de leurs résultats
    for page_num in selected_pages:
        display_page_and_results(page_num, st.session_state.pdf_document)

    # Fermer le document PDF
    st.session_state.pdf_document.close()

    # Supprimer les fichiers temporaires
    os.unlink(pdf_path)
    if 'repaired_pdf_path' in locals():
        os.unlink(repaired_pdf_path)

else:
    st.info("Veuillez télécharger un fichier PDF pour commencer.")