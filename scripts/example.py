from src.kingmans_coalescent import iterative_tree_build
from src.gene_process import iterative_gene_tree_build
from src.distance_dependent_gene_process import iterative_dd_gene_tree_build_ihpp

import pandas as pd

from src.tree_distances import rf_distance, bl_distance, dist_matr_distance

from ete3 import Tree, TextFace
from ete3.treeview import TreeStyle

import dendropy
from phylodm import PhyloDM

import os
import svgutils.transform as sg

# define the plotting style of ete3
ts = TreeStyle()
ts.show_leaf_name = True  # show leaf names
ts.show_branch_length = True  # show branch lengths
ts.show_branch_support = False  # show support values
ts.rotation = 90
ts.show_scale = False
ts.aligned_foot = True

# define and create the plot directory
plotting_dir = '../plots/examples/'
if not os.path.exists(plotting_dir):
    os.makedirs(plotting_dir)

# define TyxonNamespace for dendropy
tns = dendropy.TaxonNamespace()


def combine_svgs(file_name_1, file_name_2, file_name_3):
    ''' Function to combine three tree plots in one svg files. This does not rescale them to fit a uniform
    scale for the edge lengths.

    Inputs are the paths to the tree plots in SVG format.
    '''
    # Load your three SVG files
    fig1 = sg.fromfile(file_name_1)
    fig2 = sg.fromfile(file_name_2)
    fig3 = sg.fromfile(file_name_3)

    # Extract the root <svg> elements
    plot1 = fig1.getroot()
    plot2 = fig2.getroot()
    plot3 = fig3.getroot()

    # Move plots horizontally (shift them by x,y)
    plot1.moveto(0, 40)
    plot2.moveto(600, 40)  # shift 2nd svg 500px to the right
    plot3.moveto(1200, 40)  # shift 3rd svg 1000px to the right

    # Titles (x positions also doubled)
    title1 = sg.TextElement(100, 25, "Species Tree", size=24, weight="bold")
    title2 = sg.TextElement(700, 25, "Gene Tree (Uniform HGT)", size=24, weight="bold")
    title3 = sg.TextElement(1300, 25, "Gene Tree (dd HGT)", size=24, weight="bold")

    # Create a new SVG canvas large enough to fit all three
    combined = sg.SVGFigure("1800", "400")  # width x height

    # Append all three
    combined.append([plot1, plot2, plot3, title1, title2, title3])

    # Save to file
    combined.save(os.path.join(plotting_dir, "combined.svg"))

if __name__ == '__main__':
    # set parameters for the simulated tree
    speciation_rate = 1
    HGT_rate = 1
    n_species = 5

    # build the underlying species tree
    species_tree_dict, species_tree_string, realised_species_coalescence_events, surviving_species_lineages = iterative_tree_build(
        n_individuals=n_species,
        rate=speciation_rate,
    )

    species_nwk_str, species_T_MRCA = next(iter(species_tree_string.values()))
    species_nwk_str += ';'
    species_tree = Tree(species_nwk_str)

    species_ts = ts
    #species_tree.show(tree_style=ts)
    species_tree.render(os.path.join(plotting_dir, 'species_tree.svg'),
                             h=180,
                             w=160,
                             units='mm',
                             tree_style=species_ts,
                             )

    # build gene tree with uniform HGT
    uniform_gene_tree_dict, uniform_gene_tree_string, n_HGTs_uniform = iterative_gene_tree_build(
        n_individuals=n_species,
        HGT_rate=HGT_rate,
        realised_coalescent_events=realised_species_coalescence_events,
        surviving_lineages=surviving_species_lineages,
    )

    uniform_gene_nwk_str, uniform_gene_T_MRCA = next(iter(uniform_gene_tree_string.values()))
    uniform_gene_nwk_str += ';'
    uniform_gene_tree = Tree(uniform_gene_nwk_str)

    uniform_gene_ts  = ts
    # uniform_gene_tree.show(tree_style=ts)
    uniform_gene_tree.render(os.path.join(plotting_dir, 'uniform_gene_tree.svg'),
                             h=180,
                             w=160,
                             units='mm',
                             tree_style=uniform_gene_ts,
                             )

    # build gene tree with distance dependent HGT

    # find the distance matrix of the species tree
    species_tree = dendropy.Tree.get(
        data=species_nwk_str,
        schema='newick',
        taxon_namespace=tns,
    )
    species_tree_phylo = PhyloDM.load_from_dendropy(species_tree)
    distance_matrix_species_tree = species_tree_phylo.dm(norm=False)

    dd_gene_tree_dict, dd_gene_tree_string, n_HGTs_dd = iterative_dd_gene_tree_build_ihpp(
        n_individuals=n_species,
        HGT_rate=HGT_rate,
        realised_coalescent_events=realised_species_coalescence_events,
        species_dist_matrix=distance_matrix_species_tree,
    )

    dd_gene_nwk_str, dd_gene_T_MRCA = next(iter(dd_gene_tree_string.values()))
    dd_gene_nwk_str += ';'
    dd_gene_tree = Tree(dd_gene_nwk_str)

    dd_gene_ts = ts
    #dd_gene_tree.show(tree_style=ts)
    dd_gene_tree.render(os.path.join(plotting_dir, 'dd_gene_tree.svg'),
                             h=180,
                             w=160,
                             units='mm',
                             tree_style=dd_gene_ts,
                             )

    # calculate the distances between the species tree and gene trees
    distances_df = pd.DataFrame(columns=['uniform HGT', 'dd HGT'])
    idx = 0
    for distance in [rf_distance, bl_distance, dist_matr_distance]:
        dists = []
        for gene_tree_nwk in [uniform_gene_nwk_str, dd_gene_nwk_str]:
            d = distance(species_nwk_str, gene_tree_nwk)
            dists.append(d)
        distances_df.loc[idx] = dists
        idx += 1
    distances_df.index = ['RF dist.', 'Branch length dist.', 'Dist. matrix dist.']

    # print the results
    print(f'Time to the MRCA:\n\n'
          f'species tree: {species_T_MRCA}\n'
          f'gene tree (unif. HGT): {uniform_gene_T_MRCA}\n'
          f'gene tree (dd HGT): {dd_gene_T_MRCA}\n')

    print(f'\nNumber of HGT events:\n\n'
          f'uniform: {n_HGTs_uniform}\n'
          f'distance dependent: {n_HGTs_dd}\n')

    print('\nDistances between species and gene tree (for both uniform and distance dependent HGT):\n')
    print(distances_df)


    # plot all three trees next to each other and save to combined.svg (the trees are not on the same scale!)
    combine_svgs(
        os.path.join(plotting_dir, 'species_tree.svg'),
        os.path.join(plotting_dir, 'uniform_gene_tree.svg'),
        os.path.join(plotting_dir, 'dd_gene_tree.svg'),
    )