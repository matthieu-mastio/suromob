import numpy as np
from smt.design_space import DesignSpace, IntegerVariable, FloatVariable
from smt.applications.mfk import NestedLHS

def create_amod_design_space():
    """
    Crée et retourne l'espace de conception (Design Space) SMT
    pour le problème d'optimisation AMoD (Toulouse).
    
    L'espace comprend :
    - 4 variables entières pour la composition de la flotte (0 à 15 navettes par type).
    - 1 variable entière pour le temps d'attente maximum (300 à 1800 secondes).
    - 1 variable continue pour l'élasticité du temps de trajet (alpha).
    - 1 variable continue pour le score / utilité de l'AMoD par rapport à la voiture.
    """
    
    ds = DesignSpace([
        # --- 1. Composition de la flotte (Nombres entiers de navettes) ---
        IntegerVariable(0, 5),  # nb_shuttles_4_seats
        IntegerVariable(0, 5),  # nb_shuttles_6_seats
        IntegerVariable(0, 5),  # nb_shuttles_15_seats
        IntegerVariable(0, 5),  # nb_shuttles_20_seats
        
        # --- 2. Contraintes de Service et Rebalancing ---
        # maxWaitTime (en secondes) : de 5 minutes à 30 minutes
        IntegerVariable(300, 1800), # maxWaitTime
        
        # maxTravelTimeAlpha (facteur multiplicatif du temps de trajet direct)
        FloatVariable(1.001, 2.0),  # maxTravelTimeAlpha
        
        # Note : maxTravelTimeBeta est fixé pour être égal à maxWaitTime,
        # il n'est donc pas inclus dans l'espace de recherche (c'est une constante dérivée).
        
        # --- 3. Scoring / Utilité de l'AMoD ---
        # Modificateur du score (Utility) de l'AMoD par rapport à la voiture
        FloatVariable(-0.5, 0.5),   # score_modifier
    ])
    
    return ds

from scipy.spatial.distance import cdist

def _reorder_nested(small, large):
    """
    Réorganise le tableau 'large' pour que ses premières lignes correspondent exactement
    aux lignes de 'small' (dans le même ordre).
    """
    dist = cdist(small, large)
    idx_small_in_large = np.argmin(dist, axis=1)
    
    idx_large_only = [i for i in range(len(large)) if i not in idx_small_in_large]
    
    reordered_large = np.vstack((large[idx_small_in_large], large[idx_large_only]))
    return reordered_large

def generate_nested_doe(ds, hifi_size=10):
    """
    Génère un Design of Experiments (DoE) imbriqué (Nested) pour le Multi-Fidelity Kriging (MFK)
    en utilisant l'outil NestedLHS de SMT.
    
    Les échantillons sont réordonnés de sorte que :
    - Les 10 premiers points de LF1 soient exactement HF.
    - Les 20 premiers points de LF0 soient exactement LF1.
    
    Args:
        ds: L'espace de conception SMT (DesignSpace)
        hifi_size: La taille du plan Haute-Fidélité (HF). (défaut: 10)
               
    Returns:
        Un dictionnaire contenant les tableaux d'échantillons LF0 (80), LF1 (20) et HF (10).
    """
    # L'outil SMT calcule nt = [80, 40, 20, 10] si on met nlevel=4 et hifi_size=10
    sampler = NestedLHS(nlevel=4, design_space=ds,seed=31)
    
    # La méthode __call__ prend en argument le nombre de points haute fidélité (HF)
    doe_list = sampler(hifi_size) 
    
    # On filtre pour ne garder que 80, 20 et 10
    hf = doe_list[3]
    lf1_raw = doe_list[2]
    lf0_raw = doe_list[0]
    
    # Réorganisation pour un alignement parfait au début des tableaux
    lf1 = _reorder_nested(hf, lf1_raw)
    lf0 = _reorder_nested(lf1, lf0_raw)
    
    nested_doe = {
        "LF0": lf0,
        "LF1": lf1,
        "HF":  hf
    }
    
    return nested_doe

import pandas as pd

if __name__ == "__main__":
    # Test de l'espace de conception et du Nested DoE
    ds = create_amod_design_space()
    print(f"L'espace de conception SMT a été initialisé avec {len(ds.design_variables)} variables.")
    
    nested_samples = generate_nested_doe(ds, hifi_size=10)
    
    print("\nGénération du Nested DoE pour MFK avec l'outil NestedLHS :")
    for level, sample in nested_samples.items():            
        print(f" - {level} (Taille : {sample.shape[0]} points)")
        if sample.shape[0] == 80 :
            print("for 1%, 5%, 10%")  
        if sample.shape[0] == 20 :
            print("for 20%, 25%")  
        if sample.shape[0] == 10 :
            print("for 50%")
            
    # --- EXPORT CSV AVEC PANDAS ---
    column_names = [
        "4_seats", "6_seats", "15_seats", "20_seats", 
        "WaitTime", "Alpha", "Score"
    ]
    
    print("\nExport des plans d'expériences en CSV...")
    for level, sample in nested_samples.items():
        filename = f"doe_{level}_{sample.shape[0]}.csv"
        df = pd.DataFrame(sample, columns=column_names)
        
        # S'assurer que les variables discrètes sont castées en entiers dans le CSV
        for col in ["4_seats", "6_seats", "15_seats", "20_seats", "WaitTime"]:
            df[col] = df[col].astype(int)
            
        df.to_csv(filename, index=False)
        print(f" - Fichier généré : {filename}")
    
    print("\nL'imbrication stricte au début des fichiers est garantie !")
