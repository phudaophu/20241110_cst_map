import networkx as nx
import osmnx as ox
import math
import numpy as np
from extract import getDir
from load_map import removeNodes

#lon, lat, lon_dist, lat_dist = 106.70199935, 10.782499699999999, 0.16599475000000297, 0.10649849999999983
lon, lat, lon_dist, lat_dist = 106.65416, 10.782499699999999, 0.16599475000000297, 0.10649849999999983
# lon_dist, lat_dist = (18458, 11842) m
# ratio = 111196.2878
distance = 750
ox.settings.use_cache = True

def splitMap(m_dist_from_center=2000, intersection=0.3, max_lon=106.8679941, min_lon=106.5360046, max_lat=10.8889982, min_lat=10.6760012):
    coord_to_m_ratio = 111196.2878
    max_dist = m_dist_from_center/coord_to_m_ratio
    print(max_dist)
    border = 0.5*intersection*max_dist
    new_max_dist = max_dist - border
    print(new_max_dist)
    max_lon, min_lon, max_lat, min_lat = max_lon-border, min_lon+border, max_lat-border, min_lat+border
    lon_dist = max_lon-min_lon
    lat_dist = max_lat-min_lat
    ratio = [math.ceil(0.5*lon_dist/new_max_dist),math.ceil(0.5*lat_dist/new_max_dist)]
    print(ratio)
    lon_ranges = [min_lon+(i+0.5)*lon_dist/ratio[0] for i in range(ratio[0])]
    lat_ranges = [min_lat+(j+0.5)*lat_dist/ratio[1] for j in range(ratio[1])]
    lon_ranges = np.asarray(lon_ranges)
    lat_ranges = np.asarray(lat_ranges)
    print(lon_ranges)
    print(lat_ranges)
    return lon_ranges, lat_ranges, m_dist_from_center

def getMap(m_dist_from_center=2000, intersection=0.3, max_lon=106.8679941, min_lon=106.5360046, max_lat=10.8889982, min_lat=10.6760012):
    lon_ranges, lat_ranges, m_dist_from_center = splitMap(m_dist_from_center=m_dist_from_center, intersection=intersection, max_lon=max_lon, min_lon=min_lon, max_lat=max_lat, min_lat=min_lat)
    graphs = [ox.graph.graph_from_point((lat, lon), dist=m_dist_from_center, network_type="drive") for lat in lat_ranges for lon in lon_ranges]
    #print(graphs)
    full_graph = nx.compose_all(graphs)
    #print(full_graph)
    ox.io.save_graphml(full_graph, filepath=getDir('./fullgraph.osm'))
    return


#getMap()

#G = ox.graph_from_bbox(bbox=(10.81754,10.76729,106.65416,106.71553), network_type='drive', retain_all=False)
#ox.io.save_graphml(G, filepath=getDir('./') + 'map_3_areas.osm')
#print('G: ', G)
G_1 = ox.io.load_graphml(filepath=getDir('./map_3_areas.osm'))
# print('G_1: ', G_1)
# G_proj = ox.project_graph(G_1)

fig, ax = ox.plot_graph(G_1, node_color="r")

#G_1 = ox.io.load_graphml(filepath=getDir('./newgraph_conso.osm'))
#print('G_1: ', G_1)
## simplify the network
#G_3 = removeNodes(G_1)
#print('G_3: ', G_3)
#ox.plot.plot_graph(G_3, node_size=1)
'''
G_1 = ox.io.load_graphml(filepath=getDir('./map_3_areas.osm'))
print('G_1: ', G_1)
G_proj = ox.project_graph(G_1)

# consolidate the network
G_2 = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)
ox.io.save_graphml(G_2, filepath=getDir('./') + 'newgraph_conso.osm')
print('G_2: ', G_2)

# simplify the network
G_3 = ox.simplify_graph(G_1)
ox.io.save_graphml(G_3, filepath=getDir('./') + 'newgraph_simpl_strict.osm')
print('G_3: ', G_3)

# simplify network with strict mode turned off
G_4 = ox.simplify_graph(G_1.copy(), edge_attrs_differ=["osmid"])
ox.io.save_graphml(G_4, filepath=getDir('./') + 'newgraph_simpl.osm')
print('G_4: ', G_4)
'''

'''
G = ox.graph.graph_from_point((lat, lon), dist=distance, network_type="drive")
ox.io.save_graphml(G, filepath=getDir('./fullgraph.osm'))
# test load
print('G: ', G)
G_1 = ox.io.load_graphml(filepath=getDir('./fullgraph.osm'))
print('G_1: ', G_1)
'''
#ox.save_load.save_graph_osm(G, filename='fullgraph.osm')
'''
# download street network data from OSM and construct a MultiDiGraph model
G = ox.graph.graph_from_point((lat, lon), dist=distance, network_type="drive") # distance in meters

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

# save graph as a geopackage or graphml file
ox.io.save_graph_geopackage(G, filepath="./graph.gpkg")
ox.io.save_graphml(G, filepath="./graph.graphml")

#highways_to_keep = ['motorway', 'trunk', 'primary']
#H = nx.MultiDiGraph()
#for u,v,attr in G.edges(data=True):
#    if attr['highway'] in highways_to_keep:
#        H.add_edge(u,v,attr_dict=attr)
#        H.node[u] = G.node[u]
#        H.node[v] = G.node[v]
#
#H.graph = G.graph
##ox.plot_graph(H)
#
#fig, ax = ox.plot.plot_graph(
#    G, bgcolor="k", node_color=nc, node_size=50, edge_linewidth=2, edge_color="#333333"
#)

nodes = G._node
nodes = G.edges
print("G: ", len(nodes))
#for i in nodes:
#    print(i)
#    print(nodes[i]) # y, x, street_count, bc (betweenness centrality)
'''