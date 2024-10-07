import os
import logging
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
import openai
from dotenv import load_dotenv

# Configuration du logging et chargement des variables d'environnement
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration du client OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La clé API OpenAI n'est pas définie. Assurez-vous qu'elle est présente dans le fichier .env")
openai.api_key = api_key

def extract_text_from_page(page: PyPDF2.PageObject) -> str:
    """Extrait le texte d'une page PDF."""
    try:
        return page.extract_text()
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du texte : {e}")
        return ""

def perform_ocr(image: Image.Image) -> str:
    """Effectue l'OCR sur une image."""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"Erreur lors de l'OCR : {e}")
        return ""

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame en supprimant les lignes vides, en rendant les noms de colonnes uniques et en convertissant les types de données."""
    # Supprimer les lignes où toutes les valeurs sont NaN
    df = df.dropna(how='all').reset_index(drop=True)

    if df.empty:
        logging.warning("Le DataFrame est vide après suppression des lignes NaN.")
        return df

    # Rendre les noms de colonnes uniques
    df = make_column_names_unique(df)

    for col in df.columns:
        # Convertir en numérique si possible
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception as e:
            logging.error(f"Erreur lors de la conversion de la colonne {col} en numérique : {e}")
            pass

    return df

def make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Rend les noms de colonnes uniques en ajoutant des suffixes numériques aux doublons."""
    df.columns = df.columns.astype(str).str.strip()  # Supprimer les espaces

    seen = {}
    new_columns = []
    for col in df.columns:
        cnt = seen.get(col, 0)
        if cnt > 0:
            new_col = f"{col}_{cnt}"
            while new_col in seen:
                cnt += 1
                new_col = f"{col}_{cnt}"
            seen[col] = cnt
            seen[new_col] = 1
            new_columns.append(new_col)
        else:
            seen[col] = 1
            new_columns.append(col)
    df.columns = new_columns
    return df

def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """Sauvegarde un DataFrame en CSV."""
    try:
        dataframe.to_csv(output_path, index=False)
        logging.info(f"Table sauvegardée dans {output_path}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde en CSV : {e}")