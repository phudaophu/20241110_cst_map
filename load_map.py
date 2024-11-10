import networkx as nx
import osmnx as ox
import math
import numpy as np
from extract import getDir

ox.settings.use_cache = True

def removeNodes(graph):
    # Select all nodes with only 2 neighbors
    nodes_to_remove = [n for n in graph.nodes if len(list(graph.neighbors(n))) == 2]

    # For each of those nodes
    for node in nodes_to_remove:
        # get the number of neigbors for each node
        neghbors = len(list(graph.neighbors(node)))
        # case 1: 1 neighbors -> remove
        if neghbors == 1:
            graph.remove_node(node)
        # case 2: 2 neighbors -> remove, connect the neighbors
        elif neghbors == 2:
            # We add an edge between neighbors (len == 2 so it is correct)
            graph.add_edge(*graph.neighbors(node))
            # And delete the node
            graph.remove_node(node)
    
    #single_nodes = []
    #for node in graph.nodes(data=True):
    #    if 'street_count' in node[1]:
    #        #print(node[1])
    #        if node[1]['street_count'] == 0:
    #            single_nodes.append(node[0])
    #print(single_nodes)
    #for node in single_nodes:
    #    graph.remove_node(node)
    
    return graph

def loadMap(map_name):
    # load map from file
    G = ox.io.load_graphml(filepath=getDir(map_name))
    #print('G: ', G)
    #G = removeNodes(G)
    #print('G: ', G)
    #G = nx.DiGraph(G)
    #G = ox.convert.to_digraph(G, weight='length')
    #print('G: ', G)
    #G = removeNodes(G)
    #print('G: ', G)
    
    '''
    # impute edge (driving) speeds and calculate edge travel times
    G = ox.speed.add_edge_speeds(G)
    G = ox.speed.add_edge_travel_times(G)

    # you can convert MultiDiGraph to/from GeoPandas GeoDataFrames
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    G = ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=G.graph)

    # convert MultiDiGraph to DiGraph to use nx.betweenness_centrality function
    # choose between parallel edges by minimizing travel_time attribute value
    D = ox.utils_graph.get_digraph(G, weight="travel_time")

    # calculate node betweenness centrality, weighted by travel time
    bc = nx.betweenness_centrality(D, weight="travel_time", normalized=True)
    nx.set_node_attributes(G, values=bc, name="bc")

    # plot the graph, coloring nodes by betweenness centrality
    #nc = ox.plot.get_node_colors_by_attr(G, "bc", cmap="plasma")
    #fig, ax = ox.plot.plot_graph(
    #    G, bgcolor="k", node_color=nc, node_size=50, edge_linewidth=2, edge_color="#333333"
    #)
    '''
    return G

if __name__ == "__main__":
    map_name = 'newgraph_conso.osm'
    # load the graph from map file
    G = loadMap(map_name)
    ox.plot.plot_graph(G, node_size=1)
