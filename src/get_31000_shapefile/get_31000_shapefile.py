import osmnx as ox
import geopandas as gpd

# 1. Requête ciblée sur le code postal 31000
print("Téléchargement du polygone pour le CP 31000...")
query = {'postalcode': '31000', 'country': 'France'}
gdf_31000 = ox.geocode_to_gdf(query)

# 2. Reprojection en Lambert 93 (EPSG:2154) obligatoire pour la France
gdf_l93 = gdf_31000.to_crs(epsg=2154)

# 3. Application du buffer (en mètres)
# Ajustez cette valeur selon le débordement souhaité autour de l'hypercentre
taille_buffer_metres = 0 
print(f"Application d'un buffer de {taille_buffer_metres} mètres...")
gdf_l93['geometry'] = gdf_l93.geometry.buffer(taille_buffer_metres)

# 4. Nettoyage des colonnes
# Le ScenarioCutter d'Eqasim n'a besoin que de la géométrie pure
gdf_clean = gdf_l93[['geometry']]

# 5. Exportation en Shapefile
nom_fichier = "emprise_31000_buffer.shp"
gdf_clean.to_file(nom_fichier)
print(f"Fichier '{nom_fichier}' généré avec succès dans le dossier courant.")
