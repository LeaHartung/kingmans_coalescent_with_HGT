import pandas as pd
import numpy as np
from itertools import product

import dendropy
from phylodm import PhyloDM

from .kingmans_coalescent import iterative_tree_build, update_tree_string

tns = dendropy.TaxonNamespace()


def draw_involved_allele_lines(dist_matrix: pd.DataFrame) -> [int, int]:
    """
    Draw involved allele lines in a potential HGT event from a distance matrix.

    :param dist_matrix: distance matrix of the species tree
    :return: pair of birth and death labels
    """
    df_inv = 1 / dist_matrix
    df_inv = df_inv.replace(np.inf, 0)

    pairs = []
    probs = []
    for pair in product(df_inv.index, df_inv.columns):
        pairs.append(pair)
        probs.append(df_inv.loc[pair])
    probs = probs / sum(probs)

    ij = np.random.choice(range(len(pairs)), size=1, p=probs)

    return pairs[ij[0]]  # = birth, death


def iterative_dd_gene_tree_build(
        n_individuals: int,
        HGT_rate: float,
        realised_coalescent_events: pd.DataFrame,
        surviving_lineages: pd.DataFrame,
        species_dist_matrix: np.array,
):
    """
    Function that iteratively builds the phylogenetic tree of a gene with distance dependent HGT
    given the underlying species tree.

    :param n_individuals: number of individuals in the population
    :param HGT_rate: Rate at which the HGT events happen
    :param realised_coalescent_events: The time and direction of the coalescent events in the species tree
    :param surviving_lineages: pd.DatFrame holding the time of each coalescence event in the species tree and which
                                species lineages were still active afterward
    :param species_dist_matrix: np.array holding the species distance matrix

    :return:
        - A dictionary holding the final surviving lineage and the grouping of species in subtrees
        - A dictionary holding the final surviving lineage and the tree in Newick format, including branch lengths
    """
    individuals = range(n_individuals)
    allele_tree_dict = dict([(i, i) for i in individuals])
    allele_tree_string = dict([(i, [str(i), 0]) for i in individuals])

    original_species_dist_matrix = pd.DataFrame(species_dist_matrix, columns=list(individuals), index=list(individuals))
    current_species_dist_matrix = original_species_dist_matrix.copy()

    time = 0
    t_next_realised_speciation_event = realised_coalescent_events.iloc[0]['time']
    last_species_colaescent_time = realised_coalescent_events.iloc[-1]['time']

    event_type = None
    while time < last_species_colaescent_time:
        number_of_surviving_alleles = len(allele_tree_dict.keys())
        if number_of_surviving_alleles == 1:
            break

        inv_dist_mat = current_species_dist_matrix.rdiv(1)  # get reciprocal of dist mat entries
        inv_dist_mat = inv_dist_mat.replace(np.inf, 0)
        rate_next_HGT_event = 0.5 * HGT_rate * inv_dist_mat.sum().sum()
        try:
            assert rate_next_HGT_event > 0
        except AssertionError:
            print(f"rate below 0: {rate_next_HGT_event}")
            raise AssertionError
        time_to_next_HGT_event = np.random.default_rng().exponential(scale=1 / rate_next_HGT_event, size=1)[0]
        time = time + time_to_next_HGT_event

        if time < t_next_realised_speciation_event:
            # HGT happens
            # draw which lines merge and direction
            event_type = 'HGT'
            copy_mat = current_species_dist_matrix.copy()

            birth, death = draw_involved_allele_lines(copy_mat)
        else:
            # speciation happens
            event_type = "speciation"
            merging_species = realised_coalescent_events[
                realised_coalescent_events['time'] == t_next_realised_speciation_event]

            birth = int(merging_species['from'].iloc[0])
            death = int(merging_species['to'].iloc[0])

            time = t_next_realised_speciation_event
            try:
                t_next_realised_speciation_event = \
                    realised_coalescent_events[realised_coalescent_events['time'] > time].iloc[0]['time']
            except IndexError:
                pass

        if birth in allele_tree_dict and death in allele_tree_dict:  # coalescent event with two active allele lineages
            allele_tree_string = update_tree_string(
                tree_string=allele_tree_string,
                birth=birth,
                death=death,
                time=time
            )

            allele_tree_dict[birth] = (allele_tree_dict[birth], allele_tree_dict[death])
            allele_tree_dict.pop(death)

            current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=1)  # delete recipient column
            if event_type == 'speciation':
                current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0)  # delete recipient row

        elif death in allele_tree_dict:  # transfer of an allele lineage
            allele_tree_dict[birth] = allele_tree_dict.pop(death)
            allele_tree_string[birth] = allele_tree_string.pop(death)

            current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=1)  # delete recipient column
            # reintroduce origin column
            current_species_dist_matrix[birth] = original_species_dist_matrix[birth].loc[
                current_species_dist_matrix.index]
            if event_type == 'speciation':
                current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0)  # delete recipient row

        else:
            assert event_type == "speciation"
            current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0)  # delete recipient row
    return allele_tree_string, allele_tree_string


if __name__ == '__main__':
    n_individuals = 5
    speciation_rate = 1

    tree_dict, tree_string, realised_coalescent_events, surviving_lineages = iterative_tree_build(
                n_individuals=n_individuals,
                rate=speciation_rate,
    )

    species_tree = dendropy.Tree.get(
        data=list(tree_string.values())[0][0] + ";",
        schema="newick",
        taxon_namespace=tns,
    )

    species_tree_phylo = PhyloDM.load_from_dendropy(species_tree)
    dm_species_tree = species_tree_phylo.dm(norm=False)

    HGT_rate = 1

    allele_tree_dict, allele_tree_string = iterative_dd_gene_tree_build(
        n_individuals=n_individuals,
        HGT_rate=HGT_rate,
        realised_coalescent_events=realised_coalescent_events,
        surviving_lineages=surviving_lineages,
        species_dist_matrix=dm_species_tree,
    )

    print(allele_tree_string)

