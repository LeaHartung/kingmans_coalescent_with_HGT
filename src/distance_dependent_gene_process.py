import pandas as pd
import numpy as np
import decimal
from itertools import product

import dendropy
from phylodm import PhyloDM

from src.kingmans_coalescent import iterative_tree_build, update_tree_string

tns = dendropy.TaxonNamespace()
rng = np.random.default_rng()


def draw_next_HGT_event_ihpp(
        start: float,
        stop: float,
        species_distance: float,
        HGT_rate: float = 1,
        threshold: float = 0,
):
    """Function to draw the next HGT event according to an inhomogeneous Poisson process
    for a given time frame and distance between two species.

    We use the intensity function HGT_rate*e^(x - 0.5*species_distance+ 0.5*threshold) for the inhomogeneous PP.
    Where threshold is an optional x-offset, that gives a threshold, for which distances the intensity function should
    be larger than the uniform HGT rate.

    If no event happens before the time horizon, this function will return a value larger
    than the time horizon. This is intentional as it represent the next HGT happening
    after the next event in the species tree.

    If the given distance is 0, np.inf (infinity) is returned.

    :param start: the lower bound of the time interval for the simulation
    :param stop: the upper bound of the time interval for the simulation
                        (in our cas the time of next event in the species tree)
    :param species_distance: Distance of the species in the species tree
    :param HGT_rate: optional factor to rescale the amount of HGT events happening
    :param threshold: optional threshold for the intensity function of the inhomogeneous PP
    :return: time to the next HGT event between two species.
    """
    if species_distance == 0:
        return np.inf

    intensity_function = lambda x: HGT_rate*float(np.exp(decimal.Decimal(x - 0.5*species_distance + 0.5*threshold)))

    upper_bound = intensity_function(stop)

    event_times = []
    current_time = start
    while current_time < stop:
        new_point = rng.exponential(1 / (HGT_rate * upper_bound), 1)[0]
        current_time += new_point
        event_times += [current_time]

    for event_time in event_times:
        u = rng.uniform(0, 1)
        if u < intensity_function(event_time) / upper_bound:
            return event_time


def iterative_dd_gene_tree_build_ihpp(
        n_individuals: int,
        HGT_rate: float,
        realised_coalescent_events: pd.DataFrame,
        species_dist_matrix: np.array,
        threshold: float = 0,
):
    """
    Function that iteratively builds the phylogenetic tree of a gene with distance dependent HGT
    following an inhomogeneous Poisson process and an underlying species tree.

    :param n_individuals: number of individuals in the population
    :param HGT_rate: Rate at which the HGT events happen
    :param realised_coalescent_events: The time and direction of the coalescent events in the species tree
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

    HGT_count = 0
    event_type = None
    while time < last_species_colaescent_time:
        number_of_surviving_alleles = len(allele_tree_dict.keys())
        if number_of_surviving_alleles == 1:
            break

        sampling_function = lambda x: draw_next_HGT_event_ihpp(start=time,
                                                               stop=t_next_realised_speciation_event,
                                                               species_distance=x,
                                                               HGT_rate=HGT_rate,
                                                               threshold=threshold,
                                                               )
        next_HGT_events = current_species_dist_matrix.map(sampling_function)

        time_of_next_HGT_event = next_HGT_events.min().min()

        if time_of_next_HGT_event < t_next_realised_speciation_event:
            # HGT happens
            # draw which lines merge and direction
            event_type = 'HGT'

            min_idx = np.where(next_HGT_events == time_of_next_HGT_event)
            birth = current_species_dist_matrix.index[min_idx[0][0]]
            death = current_species_dist_matrix.columns[min_idx[1][0]]

            time = time_of_next_HGT_event
        else:
            # speciation happens
            event_type = 'speciation'
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
            if event_type == 'HGT':
                HGT_count += 1
            elif event_type == 'speciation':
                current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0)  # delete recipient row

        elif death in allele_tree_dict:  # transfer of an allele lineage
            allele_tree_dict[birth] = allele_tree_dict.pop(death)
            allele_tree_string[birth] = allele_tree_string.pop(death)

            current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=1)  # delete recipient column
            # reintroduce origin column
            current_species_dist_matrix[birth] = original_species_dist_matrix[birth].loc[
                current_species_dist_matrix.index]
            if event_type == 'HGT':
                HGT_count += 1
            elif event_type == 'speciation':
                current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0)  # delete recipient row

        else:
            assert event_type == 'speciation'
            current_species_dist_matrix = current_species_dist_matrix.drop(death, axis=0) # delete recipient row

    return allele_tree_string, allele_tree_string, HGT_count

if __name__ == '__main__':
    n_individuals = 5
    speciation_rate = 1

    tree_dict, tree_string, realised_coalescent_events, surviving_lineages = iterative_tree_build(
                n_individuals=n_individuals,
                rate=speciation_rate,
    )

    species_tree = dendropy.Tree.get(
        data=list(tree_string.values())[0][0] + ';',
        schema='newick',
        taxon_namespace=tns,
    )

    species_tree_phylo = PhyloDM.load_from_dendropy(species_tree)
    dm_species_tree = species_tree_phylo.dm(norm=False)

    HGT_rate = 1

    allele_tree_dict, allele_tree_string, n_HGTs = iterative_dd_gene_tree_build_ihpp(
        n_individuals=n_individuals,
        HGT_rate=HGT_rate,
        realised_coalescent_events=realised_coalescent_events,
        species_dist_matrix=dm_species_tree,
    )

    print(allele_tree_string)

