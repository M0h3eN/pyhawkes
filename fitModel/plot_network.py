import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_network(G, figname):
    # Creates a copy of the graph
    H = G.copy()

    # crates a list for edges and for the weights
    edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())

    # calculates the degree of each node
    d = nx.degree(H)
    # creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*dict(d).items())

    # positions
    positions = nx.circular_layout(H)

    # Figure size
    plt.figure(figsize=(15, 15))

    # draws nodes
    nx.draw_networkx_nodes(H, positions, node_color='#DA70D6', nodelist=nodelist,
                           # the node size will be now based on its degree
                           node_size=tuple([x ** 3 for x in node_sizes]), alpha=0.8)

    # Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=8,
                            font_family='sans-serif')

    # draws the edges
    nx.draw_networkx_edges(H, positions, edge_list=edges, style='solid',
                           # adds width=weights and edge_color = weights
                           # so that edges are based on the weight parameter
                           # edge_cmap is for the color scale based on the weight
                           # edge_vmin and edge_vmax assign the min and max weights for the width
                           width=weights, edge_color=weights, edge_cmap=plt.cm.PuRd,
                           edge_vmin=min(weights), edge_vmax=max(weights))

    dc = nx.degree_centrality(G)
    a1, a2 = zip(*dc.items())

    # displays the graph without axis
    plt.axis('off')
    # saves image
    plt.savefig(figname + '.svg', format='svg', dpi=1200)
    # , plt.show()
    return np.argmax(a2)
