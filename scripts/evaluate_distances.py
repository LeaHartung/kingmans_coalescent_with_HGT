from src.tree_distances import average_distances, rf_distance, bl_distance, dist_matr_distance

import os
import pandas as pd
import ast

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def try_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


if __name__ == '__main__':
    results_dir = '../results'
    time_string = '2025-09-23_14-44-35'

    input_dir = os.path.join(results_dir, time_string)
    tree_dir = os.path.join(input_dir, 'tree_strings')

    distance_functions = {
        'rf_dist': rf_distance,
        'bl_dist': bl_distance,
        'matr_dist': dist_matr_distance,
    }

    output_dir = os.path.join(input_dir, 'avrg_distances')
    os.makedirs(output_dir, exist_ok=True)

    simulation_parameters = pd.read_csv(os.path.join(input_dir, 'simulation_parameters.csv'), header=None)
    simulation_parameters = pd.Series(simulation_parameters[1].values, index=simulation_parameters[0].values)
    simulation_parameters = simulation_parameters.apply(try_eval).to_dict()

    for distance_name, distance_function in distance_functions.items():
        for task in ['species_vs_gene', 'gene_vs_gene']:
            for normalise in [True, False]:
                file_name = f'avrg_{distance_name}_{task}.csv'
                if normalise:
                    file_name = 'normalised_' + file_name
                save_file = os.path.join(output_dir, file_name)

                print(f'Start evaluating {task} in {distance_name} (normalised:{normalise}).')

                average_distances(
                    input_dir=tree_dir,
                    task=task,
                    distance_function=lambda x, y: distance_function(x, y, normalise=normalise),
                    n_genes=simulation_parameters['n_genes'],
                    n_individuals=simulation_parameters['n_individuals'],
                    HGT_rate=simulation_parameters['HGT_rate'],
                    speciation_rate=simulation_parameters['speciation_rate'][0],
                    save_file=save_file,
                )
