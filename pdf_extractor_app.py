import streamlit as st
import os
import tempfile
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
from pdf_table_extractor_improved import extract_tables_from_image, clean_and_prepare_table
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
st.title("Extracteur de tableaux PDF")

# Paramètres de l'application
model = st.sidebar.selectbox("Modèle GPT à utiliser", ["gpt-4o", "gpt-4o-mini"], index=0)
max_workers = st.sidebar.slider("Nombre de workers pour le traitement parallèle", 1, os.cpu_count(), 2)

# Initialisation du session state
if 'extracted_tables' not in st.session_state:
    st.session_state.extracted_tables = {}
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

        extracted_data = extract_tables_from_image(temp_image_path, model)
        os.unlink(temp_image_path)
        return extracted_data
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction de la page {page_num}: {e}")
        logging.error(traceback.format_exc())
        st.error(f"Erreur lors de l'extraction de la page {page_num}: {e}")
        return {"metadata": {}, "tables": []}


def update_session_state(table_key, df, table_structure, metadata, extraction_time):
    page_num = int(table_key.split('_')[1])
    table_idx = int(table_key.split('_')[3])
    page_key = f"page_{page_num}"

    if page_key not in st.session_state.extracted_tables:
        st.session_state.extracted_tables[page_key] = {}

    st.session_state.extracted_tables[page_key][table_idx] = {
        'df': df,
        'structure': table_structure,
        'metadata': metadata,
        'extraction_time': extraction_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def display_table(table, page_num, table_idx, metadata=None, extraction_time=None):
    table_key = f"page_{page_num}_table_{table_idx}"
    editor_key = f"editor_{table_key}"
    download_key = f"download_{table_key}"
    structure_key = f"structure_button_{table_key}"

    st.subheader(f"Tableau {table_idx + 1} de la page {page_num}")
    if extraction_time:
        st.write(f"Dernière modification : {extraction_time}")

    # Afficher les métadonnées
    st.write("Métadonnées:")
    st.json({
        "Banque": metadata.get("bank_name", "Non spécifié"),
        "Client": metadata.get("client_name", "Non spécifié"),
        "Date du relevé": metadata.get("statement_date", "Non spécifié"),
        "Numéro de portefeuille": metadata.get("portfolio_number", "Non spécifié")
    })

    # Afficher la structure sous forme de dictionnaire JSON
    if st.button("Voir la structure du tableau", key=structure_key):
        st.json(table)

    # Préparer le DataFrame
    df = clean_and_prepare_table(table, metadata) if isinstance(table, dict) else table

    if df.empty:
        st.warning("Le tableau extrait est vide ou n'a pas pu être correctement formaté.")
        return

    st.write("Données du tableau:")

    # Afficher l'éditeur de données
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key
    )

    # Mettre à jour le DataFrame dans le session state si des modifications ont été apportées
    if not df.equals(edited_df):
        update_session_state(table_key, edited_df, table, metadata, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Bouton de téléchargement
    csv = edited_df.to_csv(index=False, encoding='utf-8-sig', sep=';').encode('utf-8-sig')
    st.download_button(
        label=f"Télécharger le tableau {table_idx + 1} de la page {page_num}",
        data=csv,
        file_name=f"page_{page_num}_tableau_{table_idx + 1}.csv",
        mime="text/csv",
        key=download_key
    )


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
        if st.button(f"Extraire les tableaux de la page {page_num}", key=f"extract_{page_num}"):
            with st.spinner(f'Extraction des tableaux de la page {page_num} en cours...'):
                extracted_data = extract_data_from_single_page(pdf_document, page_num, output_folder, model)
                if extracted_data and extracted_data["tables"]:
                    extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f'Extraction de la page {page_num} terminée !')
                    for idx, table in enumerate(extracted_data["tables"]):
                        update_session_state(f"page_{page_num}_table_{idx}", table, table, extracted_data["metadata"],
                                             extraction_time)
                elif extracted_data:
                    st.warning(f"Aucun tableau trouvé sur la page {page_num}")

        # Afficher les tableaux extraits pour cette page
        page_key = f"page_{page_num}"
        if page_key in st.session_state.extracted_tables:
            for table_idx, table_data in st.session_state.extracted_tables[page_key].items():
                display_table(
                    table_data['df'],
                    page_num,
                    table_idx,
                    table_data['metadata'],
                    table_data['extraction_time']
                )


# Upload du fichier PDF
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")

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
    if st.button("Extraire les tableaux de toutes les pages sélectionnées"):
        with st.spinner('Extraction des tableaux de toutes les pages sélectionnées en cours...'):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {
                    executor.submit(extract_data_from_single_page, st.session_state.pdf_document, page_num,
                                    output_folder, model): page_num for page_num in selected_pages}
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        extracted_data = future.result()
                        if extracted_data and extracted_data["tables"]:
                            extraction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for idx, table in enumerate(extracted_data["tables"]):
                                update_session_state(f"page_{page_num}_table_{idx}", table, table,
                                                     extracted_data["metadata"], extraction_time)
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