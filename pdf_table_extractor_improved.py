import os
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json
import base64
import logging
import ast
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration du client OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La clé API OpenAI n'est pas définie. Assurez-vous qu'elle est présente dans le fichier .env")

client = OpenAI(api_key=api_key)


def extract_tables_from_image(image_path: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Extrait les tableaux et les métadonnées d'une image en utilisant GPT-4o en mode multimodal."""

    # Lire l'image et l'encoder en base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = """
    Analysez attentivement cette image d'un relevé bancaire et suivez ces étapes :

    1. Identification générale :
       - Nom de la banque
       - Date du relevé
       - Nom du client
       - Numéro(s) de portefeuille client visible(s)

    2. Pour chaque tableau identifié :
       a. Repérez TOUS les en-têtes de colonnes, y compris ceux sur plusieurs lignes. Fusionnez-les correctement.
       b. Pour les colonnes sans en-tête explicite, attribuez un nom descriptif basé sur le contenu.
       c. Assurez-vous que chaque colonne a un en-tête unique et significatif.
       d. Capturez toutes les données, y compris les lignes qui pourraient sembler être des sous-sections.

    3. Pour chaque tableau, créez un dictionnaire avec :
       - 'headers': liste de tous les en-têtes de colonnes (fusionnés et uniques)
       - 'data': liste de listes, chaque sous-liste représentant une ligne complète de données
       - 'table_info': informations spécifiques au tableau (ex: titre, sous-totaux)
       - 'metadata': lien vers les informations générales (banque, client, date, portefeuille)

    4. Réflexion critique :
       - Vérifiez que tous les en-têtes sont capturés et correctement fusionnés.
       - Assurez-vous que le nombre de colonnes dans 'headers' correspond exactement au nombre d'éléments dans chaque ligne de 'data'.
       - Vérifiez que chaque tableau est lié aux métadonnées appropriées.

    5. Amélioration :
       - Si des problèmes sont identifiés, corrigez-les avant de finaliser l'extraction.
       - Assurez-vous que la structure des données reflète fidèlement la disposition visuelle du relevé.

    Formatez votre réponse comme un dictionnaire Python valide contenant :
    {
        "metadata": {
            "bank_name": "...",
            "statement_date": "...",
            "client_name": "...",
            "portfolio_number": "..."
        },
        "tables": [
            {
                "headers": [...],
                "data": [...],
                "table_info": {...},
                "metadata": {...}
            },
            ...
        ]
    }
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        max_tokens=4096
    )

    content = response.choices[0].message.content.strip()

    # Extraire le dictionnaire Python de la réponse
    dict_match = re.search(r'\{.*\}', content, re.DOTALL)
    if dict_match:
        dict_string = dict_match.group()
        try:
            extracted_data = json.loads(dict_string)
        except json.JSONDecodeError:
            # Si json.loads échoue, on essaie avec ast.literal_eval
            try:
                extracted_data = ast.literal_eval(dict_string)
            except (ValueError, SyntaxError):
                logging.error(f"Erreur lors de l'évaluation du contenu. Contenu reçu : {content}")
                return {"metadata": {}, "tables": []}
    else:
        logging.error(f"Aucun dictionnaire Python valide trouvé dans la réponse. Contenu reçu : {content}")
        return {"metadata": {}, "tables": []}

    # Vérification et ajustement des données extraites
    for table in extracted_data['tables']:
        headers = table['headers']
        data = table['data']
        max_columns = max(len(row) for row in data)

        if len(headers) != max_columns:
            logging.warning(
                f"Discordance entre le nombre d'en-têtes ({len(headers)}) et le nombre maximum de colonnes ({max_columns}). Ajustement effectué.")
            if len(headers) < max_columns:
                headers.extend([f"Colonne non nommée {i + 1}" for i in range(len(headers), max_columns)])
            else:
                headers = headers[:max_columns]

        adjusted_data = []
        for row in data:
            if len(row) < max_columns:
                adjusted_row = row + [''] * (max_columns - len(row))
            elif len(row) > max_columns:
                adjusted_row = row[:max_columns]
            else:
                adjusted_row = row
            adjusted_data.append(adjusted_row)

        table['headers'] = headers
        table['data'] = adjusted_data

    def make_unique(headers):
        seen = {}
        unique_headers = []
        for header in headers:
            if header in seen:
                seen[header] += 1
                unique_headers.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                unique_headers.append(header)
        return unique_headers

    for table in extracted_data['tables']:
        table['headers'] = make_unique(table['headers'])

    return extracted_data


def clean_and_prepare_table(table: Dict[str, Any]) -> pd.DataFrame:
    headers = table['headers']
    data = table['data']

    # Fusionner les en-têtes multi-lignes
    merged_headers = []
    for header_group in zip(*[iter(headers)] * 2):
        merged_header = ' '.join(filter(None, header_group)).strip()
        merged_headers.append(merged_header if merged_header else f"Colonne {len(merged_headers) + 1}")

    # S'assurer que les en-têtes sont uniques
    unique_headers = make_unique(merged_headers)

    # Ajuster les données si nécessaire
    max_columns = len(unique_headers)
    adjusted_data = [row + [''] * (max_columns - len(row)) for row in data]

    df = pd.DataFrame(adjusted_data, columns=unique_headers)

    # Supprimer les lignes entièrement vides
    df = df.dropna(how='all')

    # Convertir les colonnes numériques
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df

def make_headers_unique(headers: List[str]) -> List[str]:
    """
    Rend les noms de colonnes uniques en ajoutant un suffixe numérique aux doublons.
    """
    seen = {}
    unique_headers = []
    for header in headers:
        if header in seen:
            seen[header] += 1
            unique_headers.append(f"{header}_{seen[header]}")
        else:
            seen[header] = 0
            unique_headers.append(header)
    return unique_headers

def make_unique(headers):
    seen = {}
    unique_headers = []
    for header in headers:
        if header in seen:
            seen[header] += 1
            unique_headers.append(f"{header}_{seen[header]}")
        else:
            seen[header] = 0
            unique_headers.append(header)
    return unique_headers


def clean_and_prepare_table(table: Dict[str, Any], metadata: Dict[str, str]) -> pd.DataFrame:
    headers = table.get('headers', [])
    data = table.get('data', [])

    logging.info(f"Headers: {headers}")
    logging.info(f"Data sample: {data[:2] if data else 'No data'}")

    # Fusionner les en-têtes multi-lignes si nécessaire
    if len(headers) > 1 and isinstance(headers[0], list):
        merged_headers = []
        for header_group in zip(*headers):
            merged_header = ' '.join(filter(None, header_group)).strip()
            merged_headers.append(merged_header if merged_header else f"Colonne {len(merged_headers) + 1}")
    else:
        merged_headers = headers

    # S'assurer que les en-têtes sont uniques
    unique_headers = make_unique(merged_headers)

    logging.info(f"Unique headers: {unique_headers}")

    # Ajuster les données si nécessaire
    max_columns = len(unique_headers)
    adjusted_data = []
    for row in data:
        if isinstance(row, list):
            adjusted_row = row + [''] * (max_columns - len(row))
            adjusted_data.append(adjusted_row[:max_columns])
        else:
            logging.warning(f"Unexpected row format: {row}")
            adjusted_data.append([str(row)] + [''] * (max_columns - 1))

    logging.info(f"Adjusted data sample: {adjusted_data[:2] if adjusted_data else 'No data'}")

    try:
        df = pd.DataFrame(adjusted_data, columns=unique_headers)
    except ValueError as e:
        logging.error(f"Error creating DataFrame: {str(e)}")
        logging.error(f"Columns: {unique_headers}")
        logging.error(f"Data shape: {len(adjusted_data)}x{len(adjusted_data[0]) if adjusted_data else 0}")
        # Créer un DataFrame vide si la création échoue
        df = pd.DataFrame(columns=unique_headers)

    # Ajouter les métadonnées comme colonnes
    df['Banque'] = metadata.get('bank_name', '')
    df['Client'] = metadata.get('client_name', '')
    df['Portefeuille'] = metadata.get('portfolio_number', '')
    df['Date du relevé'] = metadata.get('statement_date', '')

    # Supprimer les lignes entièrement vides
    df = df.dropna(how='all')

    # Convertir les colonnes numériques
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame en supprimant les lignes vides, en rendant les noms de colonnes uniques et en convertissant les types de données."""
    df = df.dropna(how='all').reset_index(drop=True)

    if df.empty:
        logging.warning("Le DataFrame est vide après suppression des lignes NaN.")
        return df

    df = make_column_names_unique(df)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception as e:
            logging.error(f"Erreur lors de la conversion de la colonne {col} en numérique : {e}")

    return df


def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """Sauvegarde un DataFrame en CSV."""
    try:
        dataframe.to_csv(output_path, index=False)
        logging.info(f"Table sauvegardée dans {output_path}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde en CSV : {e}")


def extract_tables_from_pdf(pdf_path: str, output_folder: str):
    """Extrait les tableaux d'un PDF et les sauvegarde en CSV."""
    os.makedirs(output_folder, exist_ok=True)

    import fitz  # PyMuPDF

    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            img.save(temp_image.name, format="PNG")
            tables = extract_tables_from_image(temp_image.name)

        os.unlink(temp_image.name)

        for i, table in enumerate(tables):
            table = clean_dataframe(table)
            output_path = os.path.join(output_folder, f"page_{page_num + 1}_table_{i + 1}.csv")
            save_to_csv(table, output_path)

    pdf_document.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrait les tableaux d'un PDF et les sauvegarde en CSV.")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF")
    parser.add_argument("--output", default="extracted_tables", help="Dossier de sortie pour les CSV")
    args = parser.parse_args()

    extract_tables_from_pdf(args.pdf_path, args.output)