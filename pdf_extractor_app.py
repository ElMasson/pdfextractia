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


def display_table(table, page_num, table_idx, metadata):
    table_key = f"page_{page_num}_table_{table_idx}"

    if table_key not in st.session_state.extracted_tables:
        st.session_state.extracted_tables[table_key] = clean_and_prepare_table(table, metadata)

    df = st.session_state.extracted_tables[table_key]

    st.subheader(f"Tableau {table_idx + 1} de la page {page_num}")

    # Afficher les métadonnées
    st.write("Métadonnées:")
    st.json({
        "Banque": metadata.get("bank_name", "Non spécifié"),
        "Client": metadata.get("client_name", "Non spécifié"),
        "Date du relevé": metadata.get("statement_date", "Non spécifié"),
        "Numéro de portefeuille": metadata.get("portfolio_number", "Non spécifié")
    })

    if "table_info" in table:
        st.write("Informations du tableau:")
        st.json(table["table_info"])

    if df.empty:
        st.warning("Le tableau extrait est vide ou n'a pas pu être correctement formaté.")
        return

    st.write("Données du tableau:")

    # Permettre l'ajout et la suppression de colonnes
    col1, col2 = st.columns(2)
    with col1:
        new_column = st.text_input("Ajouter une nouvelle colonne", key=f"new_col_{table_key}")
        if new_column and new_column not in df.columns:
            df[new_column] = ''
            st.session_state.extracted_tables[table_key] = df

    with col2:
        columns_to_remove = st.multiselect("Sélectionner les colonnes à supprimer", df.columns,
                                           key=f"remove_cols_{table_key}")
        if columns_to_remove:
            df = df.drop(columns=columns_to_remove)
            st.session_state.extracted_tables[table_key] = df

    # Afficher l'éditeur de données
    edited_df = st.data_editor(df, num_rows="dynamic", key=f"editor_{table_key}")

    # Mettre à jour le DataFrame dans le session state
    st.session_state.extracted_tables[table_key] = edited_df

    # Bouton de téléchargement
    csv = edited_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Télécharger le tableau {table_idx + 1} de la page {page_num}",
        data=csv,
        file_name=f"page_{page_num}_tableau_{table_idx + 1}.csv",
        mime="text/csv",
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
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        st.warning("Le PDF semble être corrompu. Tentative de réparation...")
        repaired_pdf_path = attempt_pdf_repair(pdf_path)
        if repaired_pdf_path:
            pdf_document = fitz.open(repaired_pdf_path)
            st.success("PDF réparé avec succès!")
        else:
            st.error("Impossible de réparer le PDF. Veuillez essayer avec un autre fichier.")
            st.stop()

    # Définir le dossier de sortie pour les CSV
    output_folder = os.path.join(os.getcwd(), 'Tests', 'output')
    os.makedirs(output_folder, exist_ok=True)

    # Sélection des pages
    num_pages = len(pdf_document)
    selected_pages = st.multiselect(
        "Sélectionnez les pages à traiter",
        options=list(range(1, num_pages + 1)),
        default=list(range(1, num_pages + 1))
    )

    # Bouton pour extraire toutes les pages sélectionnées
    if st.button("Extraire les tableaux de toutes les pages sélectionnées"):
        for page_num in selected_pages:
            with st.spinner(f'Extraction des tableaux de la page {page_num} en cours...'):
                extracted_data = extract_data_from_single_page(pdf_document, page_num, output_folder, model)
                if extracted_data["tables"]:
                    st.success(f'Extraction de la page {page_num} terminée !')
                    for idx, table in enumerate(extracted_data["tables"]):
                        display_table(table, page_num, idx, extracted_data["metadata"])
                else:
                    st.warning(f"Aucun tableau trouvé sur la page {page_num}")

    # Affichage des pages individuelles
    for page_num in selected_pages:
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
            if st.button(f"Extraire les tableaux de la page {page_num}", key=f"extract_{page_num}"):
                with st.spinner(f'Extraction des tableaux de la page {page_num} en cours...'):
                    extracted_data = extract_data_from_single_page(pdf_document, page_num, output_folder, model)
                    if extracted_data["tables"]:
                        st.success(f'Extraction de la page {page_num} terminée !')
                        for idx, table in enumerate(extracted_data["tables"]):
                            display_table(table, page_num, idx, extracted_data["metadata"])
                    else:
                        st.warning(f"Aucun tableau trouvé sur la page {page_num}")

        st.markdown("---")

    # Afficher tous les tableaux extraits précédemment
    st.markdown("## Tous les tableaux extraits")
    for key, df in st.session_state.extracted_tables.items():
        page_num, table_idx = key.split('_')[1], key.split('_')[3]
        st.subheader(f"Tableau {table_idx} de la page {page_num}")
        st.dataframe(df)

        # Bouton de téléchargement pour chaque tableau
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Télécharger le tableau {table_idx} de la page {page_num}",
            data=csv,
            file_name=f"{key}.csv",
            mime="text/csv",
            key=f"download_{key}"
        )

    # Fermer le document PDF
    pdf_document.close()

    # Supprimer les fichiers temporaires
    os.unlink(pdf_path)
    if 'repaired_pdf_path' in locals():
        os.unlink(repaired_pdf_path)

else:
    st.info("Veuillez télécharger un fichier PDF pour commencer.")