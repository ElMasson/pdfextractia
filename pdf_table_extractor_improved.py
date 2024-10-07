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


def extract_tables_and_graphs_from_image(image_path: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """Extrait les tableaux et les graphiques d'une image en utilisant GPT-4o en mode multimodal."""

    # Lire l'image et l'encoder en base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    prompt = """
    Analysez attentivement cette image d'un relevé de fortune et suivez ces étapes :

    1. Identification générale :
       - Nom de la banque
       - Date du relevé
       - Nom du client
       - Tout numéro de portefeuille client visible

    2. Analyse des tableaux :
       - Déterminez s'il y a un ou plusieurs tableaux distincts.
       - Pour chaque tableau identifié :
         a. Repérez tous les en-têtes de colonnes, même s'ils sont sur plusieurs lignes.
         b. Identifiez les colonnes sans en-tête et attribuez-leur un nom générique (ex: "Colonne non nommée 1").
         c. Capturez toutes les données, y compris les lignes qui pourraient sembler être des sous-sections.

    3. Analyse des graphiques :
       - Identifiez tous les graphiques présents dans l'image.
       - Pour chaque graphique :
         a. Décrivez le type de graphique (ex: courbe, barre, camembert).
         b. Identifiez les axes et leurs étiquettes.
         c. Extrayez les données représentées dans le graphique sous forme de tableau.
         d. Soit très précis sur l'extraction des données pour obtenir les informations de graphiques complexes.

    4. Extraction des données :
       Pour chaque tableau et graphique, créez un dictionnaire avec :
       - 'headers': liste de tous les en-têtes de colonnes (incluant les noms génériques pour les colonnes sans en-tête)
       - 'data': liste de listes, chaque sous-liste représentant une ligne complète de données
       - 'info': informations spécifiques au tableau ou au graphique (ex: titre, description)

    5. Réflexion critique :
       - Assurez-vous que tous les en-têtes sont capturés, même ceux qui semblent être des titres de section.
       - Vérifiez que le nombre de colonnes dans 'headers' correspond exactement au nombre d'éléments dans chaque ligne de 'data'.
       - Examinez si des informations importantes ont été omises ou mal interprétées.

    6. Amélioration :
       - Si vous avez identifié des problèmes lors de la réflexion critique, corrigez-les.
       - Assurez-vous que la structure des données capturées reflète fidèlement la disposition visuelle du relevé.

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
                "info": {...}
            },
            ...
        ],
        "graphs": [
            {
                "type": "...",
                "headers": [...],
                "data": [...],
                "info": {...}
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
                return {"metadata": {}, "tables": [], "graphs": []}
    else:
        logging.error(f"Aucun dictionnaire Python valide trouvé dans la réponse. Contenu reçu : {content}")
        return {"metadata": {}, "tables": [], "graphs": []}

    return extracted_data

def clean_and_prepare_table(table: Dict[str, Any], metadata: Dict[str, str]) -> pd.DataFrame:
    """Nettoie et prépare le tableau pour la création d'un DataFrame."""
    headers = table['headers']
    data = table['data']

    # Fusionner les en-têtes multi-lignes si nécessaire
    if len(headers) > 1 and isinstance(headers[0], list):
        merged_headers = []
        for header_group in zip(*headers):
            merged_header = ' '.join(filter(None, header_group)).strip()
            merged_headers.append(merged_header if merged_header else f"Colonne {len(merged_headers)+1}")
    else:
        merged_headers = headers

    # S'assurer que les en-têtes sont uniques
    unique_headers = make_unique(merged_headers)

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

    df = pd.DataFrame(adjusted_data, columns=unique_headers)

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

def make_unique(headers):
    """Rend les noms de colonnes uniques en ajoutant un suffixe numérique aux doublons."""
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
# Autres fonctions utilitaires si nécessaire