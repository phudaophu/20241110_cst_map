from extract import extractData
import time
from load_map import loadMap
import random
#import numpy as np
from algorithms import bruteForce, dijkstra, bellman_ford

def benchmark(algo_func, paths, start, end):
    start_time = time.time()
    result = algo_func(paths, start, end)
    end_time = time.time()
    return result, (end_time-start_time)

paths_FloydWarshall_example = [
    {'start': 1, 'dest': 2, 'travel_time': 1},
    {'start': 1, 'dest': 4, 'travel_time': 4},
    {'start': 2, 'dest': 1, 'travel_time': 2},
    {'start': 2, 'dest': 3, 'travel_time': -2},
    {'start': 3, 'dest': 1, 'travel_time': 3},
    {'start': 4, 'dest': 1, 'travel_time': 4},
    {'start': 4, 'dest': 2, 'travel_time': -5},
    {'start': 4, 'dest': 3, 'travel_time': -1}
]

#print(*bruteForce(paths_FloydWarshall_example, 1, 3), sep='\n')

def edgeConverter(edges_format_1):
    #for i in edges_format_1:
    #    print(i)
    #    print(edges_format_1[i])
    
    edges_format_2 = [{'start': i[0], 'dest': i[1], 'travel_time': edges_format_1[i]['length']} for i in edges_format_1]
    return edges_format_2

def runAlgorithm(algorithm, graph):
    # pick start and end point randomly (seeded)
    random.seed(10)
    start, end = random.sample(sorted(graph._node), 2)
    print('start: ', start)
    print('end: ', end)
    
    # convert edge to our format
    edges_format_ours = edgeConverter(graph.edges)
    found_paths, run_time = benchmark(algorithm, edges_format_ours, start, end)
    print('==============================================')
    print('Found paths:')
    print(*found_paths, sep='\n')
    print('\nTime elapsed:')
    print(run_time)
    '''
    nodes, edges = extractData()
    #print(edges)
    found_paths, run_time = benchmark(bruteForce, edges, 4639027499, 366430703)
    print('==============================================')
    print('Found paths:')
    print(*found_paths, sep='\n')
    print('\nTime elapsed:')
    print(run_time)
    '''
    return

if __name__ == "__main__":
    map_name = 'newgraph_conso.osm'
    # load the graph from map file
    graph = loadMap(map_name)
    
    runAlgorithm(bruteForce, graph)
    
    
