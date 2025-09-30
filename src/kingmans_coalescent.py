import pandas as pd
import numpy as np
import random

from scipy.special import binom
from itertools import permutations


def update_tree_string(
        tree_string: dict,
        birth: float,
        death: float,
        time: float
) -> dict:
    """
    Function to update the tree defining string after a coalescent event.

    :param tree_string: dictionary, holding the current sub-trees
        ; format: {leaf_state_i: (sub-tree string, last coalescent time), ...}
    :param birth: key of tree_string that splits
    :param death: key of tree_string that dies
    :param time: time of the event
    :return: updated tree_string dictionary
    """
    tree_string[birth][0] = f'({tree_string[birth][0]}:{time - tree_string[birth][1]},' \
                            f' {tree_string[death][0]}:{time - tree_string[death][1]})'
    tree_string[birth][1] = time
    tree_string.pop(death)

    return tree_string


def iterative_tree_build(
        n_individuals: int,
        rate: float,
):
    """
    Function to iteratively build a kingman's coalescent tree

    :param n_individuals: number of individuals in the population
    :param rate: rate of the Kingma's coalescent
    :return:
        - A dictionary holding the final surviving lineage and the grouping of species in subtrees
        - A dictionary holding the final surviving lineage and the tree in Newick format, including branch lengths
        - A data frame holding the time and direction of the coalescent events
        - A dataframe holding the time of each coalescence event and which species lineages were still active afterward
    """
    individuals = range(n_individuals)
    tree_dict = dict([(i, i) for i in individuals])
    tree_string = dict([(i, [str(i), 0]) for i in individuals])

    realised_coalescent_events = pd.DataFrame(columns=['time', 'from', 'to'])
    surviving_lineages = pd.DataFrame(columns=['time', 'surviving_lineages'])
    surviving_lineages.loc[0] = [0, list(individuals)]

    # loop over the number of blocks left in the partition process
    time = 0
    for k in range(n_individuals, 1, -1):
        # draw time to the next coalescent event
        rate_next_coalescence = rate * binom(k, 2)
        t_next_coalescence = np.random.default_rng().exponential(scale=1 / rate_next_coalescence, size=1)[0]

        time += t_next_coalescence

        # draw coalescing lineages
        active_lineages = list(tree_dict.keys())
        possible_merging_lines = list(permutations(active_lineages, 2))
        merging_lines = random.sample(possible_merging_lines,1)[0]

        birth = merging_lines[0]
        death = merging_lines[1]

        # update the tree string and dict
        tree_string = update_tree_string(
            tree_string=tree_string,
            birth=birth,
            death=death,
            time=time
        )

        tree_dict[birth] = (tree_dict[birth], tree_dict[death])
        tree_dict.pop(death)

        # fill additional output
        realised_coalescent_events.loc[k] = [time, birth, death]
        surviving_lineages.loc[k] = [time, list(tree_dict.keys())]

    realised_coalescent_events = realised_coalescent_events.reset_index(drop=True)
    surviving_lineages = surviving_lineages.reset_index(drop=True)

    return tree_dict, tree_string, realised_coalescent_events, surviving_lineages


if __name__ == '__main__':
    n_individuals = 5
    speciation_rate = 1

    tree_dict, tree_string, realised_coalescence_events, surviving_lineages = iterative_tree_build(
        n_individuals=n_individuals,
        rate=speciation_rate,
    )

    print(tree_dict)
    print(tree_string)
    print(realised_coalescence_events)
    print(surviving_lineages)