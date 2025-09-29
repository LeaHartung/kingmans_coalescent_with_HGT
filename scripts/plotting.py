import pandas as pd
import numpy as np
from scipy.special import binom
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

from typing import Optional, Tuple

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def try_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


# implementation of bounded KDE from https://towardsdatascience.com/bounded-kernel-density-estimation-2082dff3f47f/
def silverman_bandwidth(x_data: np.ndarray) -> float:
    ma_x_data = np.ma.masked_array(x_data, ~np.isfinite(x_data))
    return (4/(3*x_data.shape[0]))**0.2 * np.std(ma_x_data)


def basic_kde(x_data: np.ndarray, x_prediction: np.ndarray) -> np.ndarray:
    """Perform Gaussian Kernel Density Estimation.
    Args:
        x_data: Sample points drawn from the distribution to estimate.
          Numpy array of shape (N,)
        x_prediction: Points at which to evaluate the density estimate.
          Numpy array of shape (N,)
    Returns:
        Densities evaluated at `x_prediction` by averaging gaussian kernels
          around `x_data`. Numpy array of shape (N,)
    """
    h = silverman_bandwidth(x_data)
    pdf_values = norm.pdf(x_prediction.reshape(1, -1),
                          loc=x_data.reshape(-1, 1),
                          scale=h)
    densities = np.mean(pdf_values, axis=0)
    return densities

def weighted_kde(x_data: np.ndarray, x_prediction: np.ndarray) -> np.ndarray:
    h = silverman_bandwidth(x_data)  # Required to evaluate CDF
    area_values = norm.cdf(1.0, x_prediction, h) - norm.cdf(0.0, x_prediction, h)
    basic_densities = basic_kde(x_data, x_prediction)
    return basic_densities / area_values


def time_MRCA_alleles(sample_size: int, n_individuals: int, HGT_rate: float, speciation_rate: float = 1):
    """
    Function to draw from the theoretical distribution of the time to the MRCA in the gene process.

    :param sample_size: number of samples drawn
    :param n_individuals: number of alleles/individuals at time 0
    :param HGT_rate: rate of an HGT event happening
    :param speciation_rate: rate of a coalescence event happening in the species process
    :return: Series of length sample size
    """
    t_MRCA = np.zeros(sample_size)

    for k in range(2, n_individuals + 1):
        rate = (HGT_rate + speciation_rate) * binom(k, 2)
        t_next_coalescent = np.random.default_rng().exponential(scale=1 / rate, size=sample_size)
        t_MRCA = t_MRCA + t_next_coalescent

    return pd.Series(t_MRCA, name='theoretic_time')


def plotting_average_distances(
        input_file: str,
        output_file: str,
        y_label: str,
        x_label: str = 'HGT rate',
        legent_title: str = 'Number of \nSpecies',
        y_lim: Optional[Tuple[float,float]] = None,
):
    data = pd.read_csv(input_file)
    data['n_individuals'] = data['n_individuals'].astype(int)
    data_melted = data.melt(
        id_vars=['n_individuals', 'HGT_rate'],
        var_name='gene',
        value_name='distance',
    )

    sns.set_theme(style='whitegrid', font_scale=1.3)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sns.scatterplot(data=data_melted, x='HGT_rate', y='distance', hue='n_individuals', palette='crest')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(title=legent_title)

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    plt.savefig(output_file, transparent=True, dpi=300)
    plt.close()


if __name__ == '__main__':
    results_dir = '../results'
    time_string = '2025-09-23_14-44-35'

    # plotting T_MRCA
    # average per HGT rate
    simulation_parameters = pd.read_csv(os.path.join(results_dir, time_string, 'simulation_parameters.csv'), header=None)
    simulation_parameters = pd.Series(simulation_parameters[1].values, index=simulation_parameters[0].values)
    simulation_parameters = simulation_parameters.apply(try_eval).to_dict()

    input_dir = os.path.join(results_dir, time_string, 't_MRCA')
    output_dir = os.path.join('../plots', time_string, 't_MRCA')
    os.makedirs(output_dir, exist_ok=True)

    average_t_MRCA_species = pd.DataFrame(
        columns=simulation_parameters['n_individuals'],
        index=simulation_parameters['HGT_rate'],
    )
    average_t_MRCA_alleles = pd.DataFrame(
        columns=simulation_parameters['n_individuals'],
        index=simulation_parameters['HGT_rate'],
    )

    for i in range(len(simulation_parameters['n_individuals'])):
        for k in range(len(simulation_parameters['HGT_rate'])):
            results = pd.read_csv(os.path.join(input_dir,
                                               f'ind_{simulation_parameters['n_individuals'][i]}'
                                               f'_srate_{simulation_parameters['speciation_rate'][0]}'
                                               f'_HGTrate_{simulation_parameters['HGT_rate'][k]}.csv'))
            average_t_MRCA_species.iloc[k, i] = results['time_species'].mean()
            average_t_MRCA_alleles.iloc[k, i] = results[f'time_alleles_0'].mean()

    # plot average t_MRCA for the allele process
    data = average_t_MRCA_alleles
    data = data.reset_index(names=['HGT_rate']).melt(id_vars='HGT_rate', var_name='n_individuals')

    sns.set_theme(style='whitegrid', font_scale=1.3)
    ax = sns.relplot(
        kind='line',
        data=data,
        x='HGT_rate', y='value',
        hue='n_individuals',
        palette='crest',
    )
    ax.set_axis_labels('HGT_rate', 'T_MRCA(alleles)')
    ax._legend.set_title('Number of \nspecies')

    # if we used uniform HGT, add theoretical points
    if simulation_parameters['HGT_type'] == 'uniform':
        theoretical_t_MRCA = pd.DataFrame(columns=['HGT_rate', 'T_MRCA', 'n_individuals'])
        for idx, row in data.iterrows():
            t_theo = (2 / (simulation_parameters['speciation_rate'][0] + row['HGT_rate'])) * (1 - 1 / row['n_individuals'])
            theoretical_t_MRCA.loc[idx] = [row['HGT_rate'], t_theo, row['n_individuals']]

        sns.scatterplot(theoretical_t_MRCA, x='HGT_rate', y='T_MRCA', color='red', marker='x', zorder=7, legend=False)

    plt.xlabel('HGT rate')
    plt.ylabel('Time to the MRCA (allele process)')

    plt.savefig(os.path.join(output_dir, 'Talleles_vs_HGTrate.png'), transparent=True, dpi=300)
    plt.close()

    # distribution of T_MRCA(genes)/T_MRCA(species)
    sns.set_theme(style='whitegrid', font_scale=1.3)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey='row')
    axes = axes.flatten()

    prediction_points = np.linspace(0, 1, 200)
    df_p_1 = pd.DataFrame(columns=simulation_parameters['n_individuals'])
    for ax, n_indiv in zip(axes, simulation_parameters['n_individuals']):  # n_indiv in n_individuals:
        data = pd.DataFrame(columns=['time_species'] + [f'time_alleles_{i}' for i in range(simulation_parameters['n_genes'])] + ['HGT_rate'])
        for rate_HGT in simulation_parameters['HGT_rate']:
            results = pd.read_csv(
                os.path.join(input_dir, f'ind_{n_indiv}_srate_{simulation_parameters['speciation_rate'][0]}_HGTrate_{rate_HGT}.csv'))
            results['HGT_rate'] = str(round(rate_HGT, 2))
            data = pd.concat([data, results], ignore_index=True)
        data.iloc[:, 1:-1] = data.iloc[:, 1:-1].div(data.time_species, axis=0)

        bounded_kdes = pd.DataFrame(columns=['x', 'density', 'HGT_rate'])
        p_1 = pd.Series()
        for rate_HGT in data['HGT_rate'].unique():
            observations = data[data['HGT_rate'] == rate_HGT]['time_alleles_0'].values
            p_1[rate_HGT] = sum(observations >= 1) / len(observations)
            prediction = weighted_kde(observations[observations < 1], prediction_points) * (1 - p_1[rate_HGT])
            kd_estimate = pd.DataFrame({'x': prediction_points, 'density': prediction})
            kd_estimate['HGT_rate'] = rate_HGT
            bounded_kdes = pd.concat([bounded_kdes, kd_estimate])
        df_p_1[n_indiv] = p_1

        sns.lineplot(data=bounded_kdes, x='x', y='density', hue='HGT_rate', palette='crest', ax=ax)
        ax.set_xlabel('time')
        ax.set_title(f'Number of species: {n_indiv}')
        ax.legend().remove()

        # Reorder all lines
        for i, line in enumerate(ax.lines[::-1]):  # reverse the line order
            line.set_zorder(i + 1)

    # Get legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Apply updated legend
    fig.legend(
        handles,
        labels,
        title='HGT rate',
        bbox_to_anchor=(0.95, 0.5),
        loc='center right',
        frameon=False,
        # fontsize='small',
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, f'kdes_ratio_all_indiv.png'), transparent=True, bbox_inches='tight',
                dpi=300)
    plt.close()

    # plotting the fraction of gene processes where T_MRCA(genes)/T_MRCA(species) = 1
    sns.set_theme(style='whitegrid', font_scale=1.3)
    df_p_1_long = df_p_1.reset_index().rename(columns={'index': 'HGT_rate'})
    df_p_1_long['HGT_rate'] = df_p_1_long['HGT_rate'].astype(float)
    df_p_1_long = df_p_1_long.melt(id_vars='HGT_rate', var_name='n_indiv', value_name='fraction')
    sns.lineplot(data=df_p_1_long, x='HGT_rate', y='fraction', hue='n_indiv', palette='crest')
    plt.ylabel('')
    plt.xlabel('HGT rate')
    plt.legend(title='Number of\n Species')

    plt.savefig(os.path.join(output_dir, f'fraction_equal_T_MRCA.png'), transparent=True, bbox_inches='tight',
                dpi=300)


    # qq-plots for the T_MRCA(genes), if we used uniform HGT
    if simulation_parameters['HGT_type'] == 'uniform':
        qq_dir = os.path.join(output_dir, 'qq_plots')
        os.makedirs(qq_dir, exist_ok=True)

        n_quantiles = simulation_parameters['sample_size']
        mses = pd.DataFrame(
            columns=simulation_parameters['n_individuals'],
            index=np.round(simulation_parameters['HGT_rate'], 2),
        )
        for n_indiv in simulation_parameters['n_individuals']:
            for rate_HGT in simulation_parameters['HGT_rate']:
                emp_times = \
                pd.read_csv(os.path.join(input_dir, f'ind_{n_indiv}_srate_{simulation_parameters['speciation_rate'][0]}_HGTrate_{rate_HGT}.csv'))[
                    f'time_alleles_0']
                theo_times = time_MRCA_alleles(sample_size=len(emp_times), n_individuals=n_indiv, HGT_rate=rate_HGT)

                mse = mean_squared_error(theo_times.sort_values(), emp_times.sort_values())
                mses.loc[round(rate_HGT, 2), n_indiv] = round(mse, 6)

                theoretic_quantiles = np.quantile(theo_times, np.linspace(start=0, stop=1, num=n_quantiles + 1))
                empirical_quantiles = np.quantile(emp_times, np.linspace(start=0, stop=1, num=n_quantiles + 1))

                a, b = np.polyfit(theo_times.sort_values(), emp_times.sort_values(), 1)

                fig, ax = plt.subplots(figsize=(6, 6))
                sns.scatterplot(x=theoretic_quantiles, y=empirical_quantiles, size=1, linewidth=0, legend=False, zorder=3)
                plt.xlim(0, theo_times.max() + 0.1)
                plt.ylim(0, emp_times.max() + 0.1)
                # plt.axline((0, b), slope=a, color='lightblue') #regression fit
                plt.axline((0, 0), slope=1, linestyle='--', color='black', zorder=5)  # identity
                plt.xlabel('theoretic quantiles')
                plt.ylabel('empirical quantiles')
                plt.savefig(os.path.join(qq_dir,
                            f'qq_plot_ind_{n_indiv}_srate_{simulation_parameters['speciation_rate'][0]}_HGTrate_{round(rate_HGT, 2)}.png'),
                            transparent=True, bbox_inches='tight', dpi=300)
                plt.close()
        mses.to_csv(os.path.join(qq_dir, 'MSE_qq_T_MRCA_latex.csv'), sep='&')


    # plotting average tree distances
    input_dir = os.path.join(results_dir, time_string, 'avrg_distances')
    output_dir = os.path.join('../plots', time_string, 'avrg_distances')
    os.makedirs(output_dir, exist_ok=True)

    y_labels = {
        'rf_dist': 'RF distance',
        'bl_dist': 'branch length distance',
        'matr_dist': 'matrix distance',
    }

    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    for file in input_files:
        input_file = os.path.join(input_dir, file)

        output_file = os.path.join(output_dir, file.replace('.csv', '_vs_HGT.png'))

        distance = file.removeprefix('normalised_').removeprefix('avrg_')
        distance = distance.removesuffix('_gene_vs_gene.csv').removesuffix('_species_vs_gene.csv')
        y_label = y_labels[distance]
        if file.startswith('normalised'):
            y_label = 'normalised ' + y_label
        y_label = 'Average ' + y_label

        plotting_average_distances(
            input_file=input_file,
            output_file=output_file,
            y_label=y_label,
        )