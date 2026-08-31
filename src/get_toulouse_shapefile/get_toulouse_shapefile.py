import argparse
import os
import geopandas as gpd
import osmnx as ox

def extract_toulouse_shapefile(buffer_meters: float = 0.0, output_file: str = "emprise_toulouse_buffer.shp", metropole: bool = False):
    """
    Extrait l'emprise géographique de Toulouse (commune complète ou métropole)
    et l'exporte en Shapefile reprojeté en Lambert 93 (EPSG:2154) pour Eqasim ScenarioCutter.
    """
    if metropole:
        print("Téléchargement du polygone pour Toulouse Métropole...")
        query = "Toulouse Métropole, Haute-Garonne, France"
    else:
        print("Téléchargement du polygone pour la commune de Toulouse (entière)...")
        query = {'city': 'Toulouse', 'country': 'France'}

    # 1. Géocodage via OSM
    gdf = ox.geocode_to_gdf(query)
    print(f"Emprise trouvée : {gdf['display_name'].iloc[0]}")

    # 2. Reprojection en Lambert 93 (EPSG:2154) obligatoire pour la France / Eqasim
    gdf_l93 = gdf.to_crs(epsg=2154)

    # 3. Application d'un éventuel buffer (en mètres)
    if buffer_meters > 0:
        print(f"Application d'un buffer de {buffer_meters} mètres...")
        gdf_l93['geometry'] = gdf_l93.geometry.buffer(buffer_meters)

    # 4. Nettoyage des colonnes (ScenarioCutter n'a besoin que de la géométrie pure)
    gdf_clean = gdf_l93[['geometry']]

    # 5. Exportation en Shapefile
    gdf_clean.to_file(output_file)
    print(f"Fichier '{output_file}' généré avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraire l'emprise Shapefile de Toulouse (commune entière ou métropole) pour Eqasim ScenarioCutter"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Taille du buffer en mètres autour de l'emprise (ex: 500 ou 1000). Défaut: 0"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="emprise_toulouse_buffer.shp",
        help="Nom du fichier Shapefile de sortie (défaut: emprise_toulouse_buffer.shp)"
    )
    parser.add_argument(
        "--metropole",
        action="store_true",
        help="Extraire toute la métropole de Toulouse au lieu de la commune seule"
    )

    args = parser.parse_args()
    extract_toulouse_shapefile(
        buffer_meters=args.buffer,
        output_file=args.output,
        metropole=args.metropole
    )
