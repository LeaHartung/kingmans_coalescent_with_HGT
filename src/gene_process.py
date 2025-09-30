import pandas as pd
import numpy as np
import random

from itertools import product, permutations

from src.kingmans_coalescent import iterative_tree_build, update_tree_string


def iterative_gene_tree_build(
        n_individuals: int,
        HGT_rate: float,
        realised_coalescent_events: pd.DataFrame,
        surviving_lineages: pd.DataFrame,
):
    """
    Function that iteratively builds the phylogenetic tree of a gene with HGT given the underlying species tree.

    :param n_individuals: number of individuals in the population
    :param HGT_rate: Rate at which the HGT events happen
    :param realised_coalescent_events: The time and direction of the coalescent events in the species tree
    :param surviving_lineages: pd.DatFrame holding the time of each coalescence event in the species tree and which
                                species lineages were still active afterward
    :return:
        - A dictionary holding the final surviving lineage and the grouping of species in subtrees
        - A dictionary holding the final surviving lineage and the tree in Newick format, including branch lengths
    """
    individuals = range(n_individuals)
    allele_tree_dict = dict([(i, i) for i in individuals])
    allele_tree_string = dict([(i, [str(i), 0]) for i in individuals])

    time = 0
    t_next_realised_speciation_event = realised_coalescent_events.iloc[0]['time']
    last_species_coalescent_time = realised_coalescent_events.iloc[-1]['time']

    HGT_count = 0
    event_type = None
    while time < last_species_coalescent_time:
        active_species = surviving_lineages[surviving_lineages['time'] <= time].iloc[-1, 1]
        number_of_surviving_species = len(active_species)
        number_of_surviving_alleles = len(allele_tree_dict.keys())
        if number_of_surviving_alleles == 1:
            break
        rate_next_HGT_event = (HGT_rate / 2) * number_of_surviving_alleles * (number_of_surviving_species - 1)
        time_to_next_HGT_event = np.random.default_rng().exponential(scale=1 / rate_next_HGT_event, size=1)[0]
        time = time + time_to_next_HGT_event

        if time < t_next_realised_speciation_event:
            # HGT happens
            # draw which lines merge and direction
            event_type = 'HGT'

            HGT_between_active_alleles = list(permutations(list(allele_tree_dict.keys()), 2))
            species_without_allele = [x for x in active_species if x not in allele_tree_dict.keys()]
            HGT_species_transfer = list(product(species_without_allele, list(allele_tree_dict.keys())))
            possible_HGT_combinations = HGT_between_active_alleles + HGT_species_transfer
            merging_lines = random.sample(possible_HGT_combinations,1)[0]
            birth = merging_lines[0]
            death = merging_lines[1]
        else:
            # speciation happens
            event_type = 'speciation'

            merging_species = realised_coalescent_events[
                realised_coalescent_events['time'] == t_next_realised_speciation_event]

            birth = merging_species['from'].iloc[0]
            death = merging_species['to'].iloc[0]

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
            if event_type == 'HGT':
                HGT_count += 1
        elif death in allele_tree_dict:  # transfer of an allele lineage
            allele_tree_dict[birth] = allele_tree_dict.pop(death)
            allele_tree_string[birth] = allele_tree_string.pop(death)
            if event_type == 'HGT':
                HGT_count += 1

    return allele_tree_string, allele_tree_string, HGT_count


if __name__ == '__main__':
    n_individuals = 5
    speciation_rate = 1

    tree_dict, tree_string, realised_coalescent_events, surviving_lineages = iterative_tree_build(
                n_individuals=n_individuals,
                rate=speciation_rate,
    )

    HGT_rate = 1

    allele_tree_dict, allele_tree_string, n_HGTs = iterative_gene_tree_build(
        n_individuals=n_individuals,
        HGT_rate=HGT_rate,
        realised_coalescent_events=realised_coalescent_events,
        surviving_lineages=surviving_lineages,
    )

    print(allele_tree_string)

