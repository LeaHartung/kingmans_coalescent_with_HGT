import pandas as pd
import numpy as np
import os

from datetime import datetime
from itertools import product
from parallelbar import progress_imap

import dendropy
from ete3 import Tree
from phylodm import PhyloDM
tns = dendropy.TaxonNamespace()

from src.kingmans_coalescent import iterative_tree_build
from src.gene_process import iterative_gene_tree_build
from src.distance_dependent_gene_process import iterative_dd_gene_tree_build_ihpp


def wrapper(
        n_individuals: int,
        speciation_rate: float,
        HGT_rate: float,
        n_genes: int,
        HGT_type: str,
):
    if HGT_type == 'uniform':
        gene_simulation_function = iterative_gene_tree_build
    elif HGT_type == 'distance_dependent':
        gene_simulation_function = iterative_dd_gene_tree_build_ihpp
    else:
        raise ValueError(f'HGT_type must be either \'uniform\' or \'distance_dependent\'')

    output = []
    output_trees = []

    tree_dict, tree_string, realised_coalescent_events, surviving_lineages = iterative_tree_build(
        n_individuals=n_individuals,
        rate=speciation_rate,
    )

    newick_string = tree_string[[*tree_string][0]][0] + ';'
    t_species = Tree(newick_string, format=5)
    MRCA_species = t_species.get_common_ancestor([str(i) for i in range(n_individuals)])
    time_MRCA_species = MRCA_species.get_distance(t_species.get_leaves_by_name('0')[0])

    output.append(time_MRCA_species)
    output_trees.append(newick_string)

    for i in range(n_genes):
        gene_simulation_arguments = {
            'n_individuals': n_individuals,
            'HGT_rate': HGT_rate,
            'realised_coalescent_events': realised_coalescent_events,
        }

        if HGT_type == 'uniform':
            gene_simulation_arguments['surviving_lineages'] = surviving_lineages
        elif HGT_type == 'distance_dependent':
            species_tree_dendro = dendropy.Tree.get(
                data=newick_string,
                schema='newick',
                taxon_namespace=tns,
            )

            species_tree_phylo = PhyloDM.load_from_dendropy(species_tree_dendro)
            species_dist_matrix = species_tree_phylo.dm(norm=False)
            gene_simulation_arguments['species_dist_matrix'] = species_dist_matrix


        gene_tree_dict, gene_tree_string, n_HGTs = gene_simulation_function(**gene_simulation_arguments)

        newick_string_alleles = gene_tree_string[[*gene_tree_string][0]][0] + ';'
        t_alleles = Tree(newick_string_alleles, format=5)
        MRCA_alleles = t_alleles.get_common_ancestor([str(i) for i in range(n_individuals)])
        time_MRCA_alleles = MRCA_alleles.get_distance(t_alleles.get_leaves_by_name('0')[0])

        output.append(time_MRCA_alleles)
        output_trees.append(newick_string_alleles)

        # asserting that the gene tree converges faster than the species tree
        if round(time_MRCA_species, 10) < round(time_MRCA_alleles, 10):
            time_now = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
            new_path = os.path.join('log', time_now)
            os.makedirs(new_path)
            with open('trees.txt', 'w', encoding='utf-8') as f:
                f.write(tree_string + '\n')
                f.write(gene_tree_string + '\n')
            raise AssertionError(f'Allele tree has larger convergence time than species tree, '
                                 f'input saved in {new_path}. \n'
                                 f'{round(time_MRCA_species, 15)} !< {round(time_MRCA_alleles, 15)}')

    return output + output_trees


def single_input_wrapper(input: list[int, float, float, int, str]):
    n_individuals, speciation_rate, HGT_rate, n_genes, HGT_type = input
    return wrapper(n_individuals, speciation_rate, HGT_rate, n_genes, HGT_type)


if __name__ == '__main__':
    # model parameters
    speciation_rate = [1]
    HGT_rate = np.concatenate([np.linspace(0.1,1,10),np.linspace(2, 10, 9)])
    n_genes = 3
    n_individuals = [5, 10, 20, 50]
    HGT_type = 'uniform'  # must be either 'uniform' or 'distance_dependent'
    assert HGT_type in ['uniform', 'distance_dependent'], f'HGT_type must be either \'uniform\' or \'distance_dependent\', got {HGT_type}'

    # simulation parameters
    sample_size = 10000
    num_processes = 6  # number of parallel processes

    # create the folders to save the simulation results
    time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    new_path = os.path.join('../results', time_now)
    os.makedirs(new_path)

    tree_path = os.path.join(new_path, 'tree_strings')
    os.makedirs(tree_path)

    t_MRCA_path = os.path.join(new_path, 't_MRCA')
    os.makedirs(t_MRCA_path)

    # save the metadata of the simulation
    idx = [
        'speciation_rate',
        'HGT_rate',
        'n_genes',
        'n_individuals',
        'sample_size',
        'num_processes',
        'HGT_type',
    ]
    data = [
        speciation_rate,
        HGT_rate,
        n_genes,
        n_individuals,
        sample_size,
        num_processes,
        HGT_type,
    ]
    meta_data = pd.Series(index=idx, data=data)
    meta_data.to_csv(os.path.join(new_path, 'simulation_parameters.csv'), header=False)

    # start simulation iteration
    for variables in product(n_individuals, speciation_rate, HGT_rate, [n_genes], [HGT_type]):
        print(f'Start simulation with {variables[0]} ind, speciation {variables[1]}, HGT {variables[2]}, genes {variables[3]}.')
        inputs = [list(variables)] * sample_size

        results = progress_imap(single_input_wrapper, inputs, chunk_size=1, n_cpu=num_processes)

        column_names_times = ['time_species'] + [f'time_alleles_{i}' for i in range(n_genes)]
        column_names_tree_strings = ['tree_string_species'] + [f'tree_string_alleles_{i}' for i in range(n_genes)]
        result_df = pd.DataFrame(results, columns=column_names_times + column_names_tree_strings)

        result_df[column_names_times].to_csv(
            os.path.join(t_MRCA_path, f'ind_{variables[0]}_srate_{variables[1]}_HGTrate_{variables[2]}.csv'),
            index=False)

        result_df[column_names_tree_strings].to_csv(
            os.path.join(tree_path, f'ind_{variables[0]}_srate_{variables[1]}_HGTrate_{variables[2]}.csv'),
            index=False)

