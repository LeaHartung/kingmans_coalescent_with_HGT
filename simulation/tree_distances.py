import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os

from ete3 import Tree
from itertools import combinations

import dendropy
from phylodm import PhyloDM
from dendropy.calculate import treecompare

tns = dendropy.TaxonNamespace()


def normalise_bl_dist(tree: dendropy.Tree) -> dendropy.Tree:
    """
    Function to normalise the edge lengths by the Euclidean norm of the edge length vector
    :param tree: tree to be normalised
    :return: normalised tree
    """
    t = tree.clone()

    edge_legths = []
    for e in t.edges():
        if e.length is not None:
            edge_legths += [e.length]

    normalising_factor = np.linalg.norm(edge_legths)

    for e in t.edges():
        if e.length is not None:
            e.length = e.length / normalising_factor

    return t


def rf_distance(nwk1, nwk2, normalise=True):
    """
    Function to calculate the Robinson-Foulds distance between two trees.
    :param nwk1: First tree in Newick format
    :param nwk2: Second tree in Newick format
    :param normalise: whether to normalise the distance to [0,1]
    :return: RF distance between the two trees
    """
    tree_1 = Tree(nwk1)
    tree_2 = Tree(nwk2)
    if normalise:
        n_leaves = len(tree_1.get_leaves())
        return tree_1.robinson_foulds(tree_2)[0]/((n_leaves - 2)*2)
    else:
        return tree_1.robinson_foulds(tree_2)[0]


def bl_distance(nwk1, nwk2, normalise: bool = False):
    """
    Function to calculate the branch length distance between two trees.
    :param nwk1: First tree in Newick format
    :param nwk2: Second tree in Newick format
    :param normalise: whether to normalise the edge lengths
    :return: Branch length distance between the two trees
    """
    tree_1 = dendropy.Tree.get(data=nwk1,
                               schema="newick",
                               taxon_namespace=tns
                               )
    tree_2 = dendropy.Tree.get(data=nwk2,
                               schema="newick",
                               taxon_namespace=tns
                               )
    if normalise:
        tree_1 = normalise_bl_dist(tree_1)
        tree_2 = normalise_bl_dist(tree_2)
    return treecompare.euclidean_distance(tree_1, tree_2)


def dist_matr_distance(nwk1, nwk2, normalise: bool = False):
    """
    Function to calculate the distance matrix based distance between two trees.
    :param nwk1: First tree in Newick format
    :param nwk2: Second tree in Newick format
    :param normalise: whether to normalise the distance matrices
    :return: Distance matrix based distance between the two trees
    """
    tree_1_dp = dendropy.Tree.get(data=nwk1,
                               schema="newick",
                               taxon_namespace=tns
                               )
    tree_1 = PhyloDM.load_from_dendropy(tree_1_dp)
    dm_tree_1 = tree_1.dm(norm=False)
    tree_2_dp = dendropy.Tree.get(data=nwk2,
                               schema="newick",
                               taxon_namespace=tns
                               )
    tree_2 = PhyloDM.load_from_dendropy(tree_2_dp)
    dm_tree_2 = tree_2.dm(norm=False)
    if normalise:
        dm_tree_1 = np.divide(dm_tree_1, np.linalg.norm(dm_tree_1))
        dm_tree_2 = np.divide(dm_tree_2, np.linalg.norm(dm_tree_2))
    return np.linalg.norm(dm_tree_1 - dm_tree_2)


def distances_species_gene(nwk_dataframe: pd.DataFrame,
                           distance_function,
                           ):
    """
    Function to calculate all distances between the first column (species tree) and all other columns (gene trees)
    in a dataframe holding Newick strings.
    :param nwk_dataframe: pandas dataframe holding Newick strings
    :param distance_function: The distance function to apply to the data.
    :return: A dataframe that holds the distances between the first column and all other columns.
    """
    n_genes = nwk_dataframe.shape[1] - 1
    assert nwk_dataframe.shape[1] > 1, "Only species trees provided."
    distances_sp = pd.DataFrame(columns=list(range(n_genes)))
    for idx, row in nwk_dataframe.iterrows():
        nwk_species = row.iloc[0]
        d = []
        for g in range(n_genes):
            nwk_gene = row.iloc[g+1]
            d.append(distance_function(nwk_species, nwk_gene))
        distances_sp.loc[idx] = d

    return distances_sp


def distances_between_genes(nwk_dataframe: pd.DataFrame,
                           distance_function,
                           ):
    """
    Function to calculate all pairwise distances between columns 2:end (gene trees) from a pandas DataFrame holding
    Newick strings.
    :param nwk_dataframe: pandas dataframe holding Newick strings
    :param distance_function: The distance function to apply to the data.
    :return: A dataframe that holds the pairwise distances between the columns 2:end.
    """
    n_genes = nwk_dataframe.shape[1] - 1
    assert nwk_dataframe.shape[1] > 2, "Less than two gene trees per species tree provided."

    combs = list(combinations(range(1, n_genes + 1), 2))
    distances_genes= pd.DataFrame(columns=list(combs))
    for idx, row in nwk_dataframe.iterrows():
        d = []
        for pair in combs:
            nwk_gene_1 = row.iloc[pair[0]]
            nwk_gene_2 = row.iloc[pair[1]]
            d.append(distance_function(nwk_gene_1, nwk_gene_2))
        distances_genes.loc[idx] = d

    return distances_genes

def average_distances(input_dir,
                               task: str,
                               distance_function,
                               n_genes: int,
                               n_individuals: list[int],
                               HGT_rate: list[float],
                               speciation_rate: float,
                               save_file: str,
                              ):
    """
    Funktion to loop over different sets of simulations ad calculate the respective average distances.
    :param input_dir: Directory where the simulation results are stores (presumably from `scripts/run_simulations.py`)
    :param task: 'species_vs_genes' or 'gene_vs_gene', defining which trees we want to compare
    :param distance_function: The distance function to apply to the data.
    :param n_genes: Number of gene trees corresponding to each species tree.
    :param n_individuals: List of number of leaves in the trees.
    :param HGT_rate: List of HGT rates from the simulation.
    :param speciation_rate: Speciation rate of the species process.
    :param save_file: File where to save the results.
    :return: Dataframe holding the average distances per n_individual, HGT_rate, and gene process
    """
    parent_dir = os.path.dirname(save_file)
    assert os.path.exists(parent_dir), "The parent directory of the save file does not exist: {}".format(parent_dir)

    if task == 'species_vs_gene':
        distance_retrieval_function = distances_species_gene
        avrg_distances = pd.DataFrame(
            columns=["n_individuals", "HGT_rate"] + [f"distance_gene_{i}" for i in range(n_genes)]
        )
    elif task == 'gene_vs_gene':
        distance_retrieval_function = distances_between_genes
        combs = list(combinations(range(n_genes), 2))
        avrg_distances = pd.DataFrame(
            columns=["n_individuals", "HGT_rate"] + [f"distance_gene_{x}" for x in combs]
        )
    else:
        raise ValueError("'task' must be either 'species_vs_gene' or 'genes_vs_gene'.")


    counter = 0
    for i in range(len(n_individuals)):
        for k in range(len(HGT_rate)):
            print(f"Processing Individuals: {n_individuals[i]}, HGT-rate: {HGT_rate[k]}")
            nwk_dataframe = pd.read_csv(os.path.join(input_dir,
                                       f"ind_{n_individuals[i]}_srate_{speciation_rate}_HGTrate_{HGT_rate[k]}.csv"))
            distances = distance_retrieval_function(nwk_dataframe,distance_function)

            avrg_distances.loc[counter] = [n_individuals[i], np.round(HGT_rate[k], 2)] + list(
                distances.mean().values)
            counter += 1

    avrg_distances.to_csv(save_file, index=False)
    return avrg_distances



if __name__ == '__main__':
    nwk1 = "((3:0.40423157232759777, (4:0.2478364463156182, 0:0.2478364463156182):0.15639512601197958):0.21134272078739158, (1:0.6059330781358645, 2:0.6059330781358645):0.009641214979124846);"
    nwk2 = "((3:0.40423157232759777, (4:0.2478364463156182, 0:0.2478364463156182):0.15639512601197958):0.21134272078739158, (1:0.6059330781358645, 2:0.6059330781358645):0.009641214979124846);"
    nwk3 = "((3:0.40423157232759777, (4:0.2478364463156182, (0:0.1775268810369201, 2:0.1775268810369201):0.07030956527869808):0.15639512601197958):0.21134272078739158, 1:0.6155742931149893);"
    nwk4 = "((3:0.40423157232759777, (4:0.2478364463156182, 0:0.2478364463156182):0.15639512601197958):0.21134272078739158, (1:0.6059330781358645, 2:0.6059330781358645):0.009641214979124846);"

    print(rf_distance(nwk1, nwk3))
    print(bl_distance(nwk1, nwk3))
    print(bl_distance(nwk1, nwk3, normalise=True))
    print(dist_matr_distance(nwk1, nwk3))
    print(dist_matr_distance(nwk1, nwk3, normalise=True))

    nwk_df = pd.DataFrame(data=[[nwk1, nwk2, nwk3, nwk4]],
                          columns=["species_tree", "gene_tree_1", "gene_tree_2", "gene_tree_3"]
                          )

    distances_to_species = distances_species_gene(nwk_df,
                                       rf_distance,
                                       )
    print(distances_to_species)

    distances_genes = distances_between_genes(nwk_df,
                                              rf_distance,
                                              )


