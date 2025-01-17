import numpy as np
from scipy.stats import t
from itertools import permutations

def gauss_ci_test(X, Y, Z, data):
    """
    Test d'indépendance conditionnelle gaussien.
    
    Parameters:
    - X: int, l'indice de la première variable dans les données.
    - Y: int, l'indice de la deuxième variable dans les données.
    - Z: list, indices des variables conditionnantes dans les données.
    - data: numpy.ndarray, matrice de données (observations x variables).
    
    Returns:
    - p_value: float, la p-valeur du test.
    - stat: float, la statistique de test.
    """
    n, p = data.shape
    
    if not Z or (isinstance(Z, list) and (len(Z)==0)):  # Si Z est vide, on calcule la corrélation simple
        r = np.corrcoef(data[:, X], data[:, Y])[0, 1]
        dll_Z = 0
    else:
        # Construction de la matrice de corrélation partielle
        assert X < p and Y < p and all(z < p for z in Z), "Indices hors limites"

        Z_data = data[:, Z]
        P_XZ = np.linalg.lstsq(Z_data, data[:, X], rcond=None)[0]
        P_YZ = np.linalg.lstsq(Z_data, data[:, Y], rcond=None)[0]
        
        resid_X = data[:, X] - Z_data @ P_XZ
        resid_Y = data[:, Y] - Z_data @ P_YZ
        
        r = np.corrcoef(resid_X, resid_Y)[0, 1]
        dll_Z = len(Z)
    
    # Calcul de la statistique de test
    dof = n - dll_Z - 2  # Degrés de liberté
    stat = np.sqrt(dof) * r / np.sqrt(1 - r**2)
    
    # Calcul de la p-valeur
    p_value = 2 * t.sf(np.abs(stat), dof)
    
    return p_value, stat


def skeleton(nodes, test_func, data_frame, alpha=0.05, verbose=False):
    """
    Génère le squelette d'un graphe basé sur des tests d'indépendance conditionnelle.

    Args:
        nodes (list): Liste des nœuds du graphe.
        test_func (callable): Fonction de test d'indépendance conditionnelle. 
                              Signature attendue : test_func(i, j, cond_set, data_array) -> list[float].
        data_frame (pandas.DataFrame): Données utilisées pour les tests (format DataFrame).
        alpha (float): Seuil de significativité pour rejeter l'indépendance conditionnelle.
        verbose (bool): Si True, affiche les détails du processus.

    Returns:
        list: Une liste des arêtes possibles avec leur validité.
              Chaque élément est une liste [nœud1, nœud2, valide].
    """
    edges = []  # Liste des arêtes avec leur validité

    # Parcourir toutes les paires de nœuds
    for node_i in nodes:
        for node_j in nodes:
            if node_j == node_i:  # Éviter les boucles sur un même nœud
                break

            # Ajouter une arête initiale entre node_i et node_j
            edges.append([node_i, node_j])
            is_edge_valid = True  # Supposition initiale : l'arête est valide

            # Construire l'ensemble conditionnel des nœuds restants
            conditional_set = []
            for node_k in nodes:
                if node_k not in [node_i, node_j] and node_k > max(node_i, node_j):
                    conditional_set.append(node_k)

                # Tester l'indépendance conditionnelle
                if node_k in nodes[max(node_i, node_j):]:
                    pval, _ = test_func(node_i, node_j, conditional_set, data_frame.to_numpy())

                    if verbose:
                        print(f"Test pour ({node_i}, {node_j} | {conditional_set}): {pval}")

                    if pval < alpha:
                        if verbose:
                            print(f"==> ({node_i}, {node_j} | {conditional_set}) est valide.")
                    else:
                        is_edge_valid = False  # L'arête est invalide si une condition échoue

            # Ajouter la validité à l'arête courante
            edges[-1].append(is_edge_valid)

            if verbose:
                print()

    return edges

def get_triplet_with_permutations(g):
    """
    Trouve tous les triplets connectés avec permutations dans un graphe non orienté.

    Args:
        g (networkx.Graph): Un graphe non orienté.

    Returns:
        list: Une liste de triplets connectés respectant les contraintes.
    """
    triplets = []
    for u, v, w in permutations(g.nodes, 3):
        if (u in g.neighbors(v) and w in g.neighbors(v)) and\
            (w, v, u) not in triplets:
            triplets.append((u, v, w))
    
    return triplets

def get_oriented_edges(triplets):
    """
    Prend une liste de triplets et renvoie une liste d'arêtes orientées vers le nœud central.

    Args:
        triplets (list of tuples): Liste de triplets (node1, central_node, node2).

    Returns:
        list of tuples: Liste des arêtes orientées sous la forme [(node1, central_node), (node2, central_node)].
    """
    oriented_edges = set()

    for triplet in triplets:
        if len(triplet) != 3:
            raise ValueError(f"Triplet invalide : {triplet}. Chaque élément doit être un triplet (node1, central_node, node2).")
        
        # Extraire les nœuds du triplet
        node1, central_node, node2 = triplet
        
        # Ajouter les arêtes orientées vers le nœud central
        oriented_edges.add((node1, central_node))
        oriented_edges.add((node2, central_node))

    return list(oriented_edges)

def get_all_oriented_edges(triplets):
    """
    Prend une liste de triplets et renvoie toutes les orientations possibles des arêtes, 
    en évitant les doublons.

    Args:
        triplets (list of tuples): Liste de triplets (node1, central_node, node2).

    Returns:
        list of tuples: Liste des arêtes orientées uniques sous la forme [(node1, central_node), ...].
    """
    oriented_edges = set()  # Utiliser un ensemble pour éviter les doublons

    for triplet in triplets:
        if len(triplet) != 3:
            raise ValueError(f"Triplet invalide : {triplet}. Chaque élément doit être un triplet (node1, central_node, node2).")
        
        # Extraire les nœuds du triplet
        node1, central_node, node2 = triplet
        
        # Ajouter les orientations possibles : 
        # 1. De node1 vers le central_node
        # 2. De central_node vers node1
        # 3. De node2 vers le central_node
        # 4. De central_node vers node2
        oriented_edges.add((node1, central_node))
        oriented_edges.add((central_node, node1))
        oriented_edges.add((node2, central_node))
        oriented_edges.add((central_node, node2))

    # Convertir l'ensemble en liste avant de retourner le résultat
    return list(oriented_edges)