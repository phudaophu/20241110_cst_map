from load_map import loadMap
import numpy as np
from edmonds_karp import find_maximum_flow_using_edmonds_karp

def getDestList(paths, start, total_travel_time = 0):
    destinations = []
    for i in range(len(paths)):
        if (paths[i]['start'] == start):
            destinations.append({
                'dest': paths[i]['dest'], 
                'travel_time': paths[i]['travel_time'], 
                'total_travel_time': total_travel_time + paths[i]['travel_time'],
                'stop': False 
                })
#    print(destinations)
    return destinations

def bruteForce(paths, start, end, total_travel_time = 0, history = {}):
    #print(history)
    #print(start)
    if (start in history):
        #print('looped')
        return []
    histories = []
    destinations = getDestList(paths, start, total_travel_time)
    #print(np.asarray(destinations))
    for each_dest in destinations:
        new_hist = history.copy()
        new_hist.update({start: total_travel_time})
        if (end in new_hist):
            print('reached dest')
            print(new_hist)
            return [new_hist]    
        histories += bruteForce(paths, each_dest['dest'], end, each_dest['total_travel_time'], new_hist)
        
    return histories


def dijkstra(paths, start, dest):
    # Step 1: Create a graph as an adjacency list
    graph = {}
    for path in paths:
        if path['start'] not in graph:
            graph[path['start']] = []
        graph[path['start']].append((path['dest'], path['travel_time']))

    # Step 2: Initialize data structures
    unvisited_nodes = {node: float('inf') for node in graph}  # Set all nodes' distances to infinity
    unvisited_nodes[start] = 0  # Start node has a distance of zero
    visited_nodes = {}  # Track visited nodes and shortest path times
    previous_nodes = {}  # Track the path taken to each node

    # Step 3: Loop until we visit all reachable nodes or find the destination
    while unvisited_nodes:
        # Select the node with the smallest known distance
        current_node = min(unvisited_nodes, key=unvisited_nodes.get)
        current_distance = unvisited_nodes[current_node]

        # If we reach the destination node, stop and construct the path
        if current_node == dest:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous_nodes.get(current_node)
            return {'path': path, 'total_time': current_distance}

        # Update the distances to each neighbor of the current node
        for neighbor, travel_time in graph.get(current_node, []):
            if neighbor in visited_nodes:
                continue  # Skip visited neighbors
            new_distance = current_distance + travel_time
            # If a shorter path to the neighbor is found, update the distance and path
            if new_distance < unvisited_nodes.get(neighbor, float('inf')):
                unvisited_nodes[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

        # Mark the current node as visited
        visited_nodes[current_node] = current_distance
        unvisited_nodes.pop(current_node)

    # If we exit the loop without reaching the destination, return no path
    return {'path': [], 'total_time': float('inf')}

def bellman_ford(paths, start, dest):
    # Step 1: Initialize distances from start to all other nodes as infinity
    distances = {}
    previous_nodes = {}
    for path in paths:
        distances[path['start']] = float('inf')
        distances[path['dest']] = float('inf')
    distances[start] = 0

    # Step 2: Relax edges V-1 times
    for _ in range(len(distances) - 1):
        for path in paths:
            u, v, weight = path['start'], path['dest'], path['travel_time']
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                previous_nodes[v] = u

    # Step 3: Reconstruct the shortest path from start to dest
    path = []
    current = dest
    while current in previous_nodes:
        path.insert(0, current)
        current = previous_nodes[current]
    if path:
        path.insert(0, start)

    return {
        'path': path if distances[dest] != float('inf') else [],
        'total_time': distances[dest] if distances[dest] != float('inf') else None
    }

#===============================================================================
def dinics_node(node, level=-1):
    # Add dinics attributes to a node
    #print(f'old: {node}')
    node['level'] = level
    #node.get('level', level)
    #print(f'new: {node}')
    #print(node)
    return node

def dinics_edge(edge, flow=0, capacity=10):
    # Add dinics attributes to a edge
    edge['flow'] = flow
    if 'capacity' not in edge:
        edge['capacity'] = capacity
    #edge.get('flow', flow)
    #edge.get('capacity', capacity)
    #print(edge)
    return edge

def condition(current_node, next_node, end_node):
    print(next_node)
    if next_node['osmid_original'] == end_node['osmid_original']:
        return True
    
    curr_end = ((current_node['x'] - end_node['x'])**2 + (current_node['y'] - end_node['y'])**2)**0.5
    curr_next = ((current_node['x'] - next_node['x'])**2 + (current_node['y'] - next_node['y'])**2)**0.5
    next_end = ((next_node['x'] - end_node['x'])**2 + (next_node['y'] - end_node['y'])**2)**0.5
    
    cond_1 = curr_end > (4/5)*(curr_next + next_end)
    #cond_2 = curr_end < (curr_next + next_end)
    
    return cond_1# and cond_2 

def BFS_buildLevelMap(graph, start_id, end_id, level_graph=None, shortest_dist=False):
    '''
    graph: the graph
    start_id: id of start node
    end_id: id of end node
    level_graph: the constructed level graph, return by reference
    '''
    start_node = graph.nodes[start_id]
    end_node = graph.nodes[end_id]
    # Level of source vertex = 0
    dinics_node(start_node, level=0)
    # save nodes in level graph to reset later
    if level_graph:
        # reset node levels
        for node_id in level_graph['nodes']:
            node = graph.nodes[node_id]
            node['level'] = -1
        pass
    else:    
        level_graph={'nodes': set(), 'edges': set()}
    level_graph['nodes'].add(start_id)
    
    # Create a queue, enqueue source vertex and mark source vertex as visited
    queue = []
    queue.append(start_id)
    
    while queue:
        current_id = queue.pop(0) # pop the 1st id
        #print(f'current id: {current_id}')
        current_node = graph.nodes[current_id]
        # get current_node's edges
        for edge in graph.edges(current_id, data=True):
            #print(f'edge: {edge}')
            edge_data = edge[2]
            next_id = edge[1] # get the end node of this edge
            next_node = graph.nodes[next_id]
            #print(f'next node: {next_id}, {next_node}')
            if 'level' not in next_node:
                next_node['level'] = -1
            if 'flow' not in edge_data:
                edge_data['flow'] = 0
            if 'capacity' not in edge_data:
                edge_data['capacity'] = 10
            
            # condition to put node in level map
            # condition: next node is closer to end node than current node
            if shortest_dist:
                match = condition(current_node, next_node, end_node)
                if not match:
                    print('not match')
                    continue
            if next_node['level'] < 0 and edge_data['flow'] < edge_data['capacity']:
                queue.append(next_id)
                # add ids to level graph
                level_graph['nodes'].add(next_id)
                level_graph['edges'].add((edge[:2]))
                # Level of current vertex is level of parent + 1
                dinics_node(next_node, level=current_node['level']+1)
                dinics_edge(edge_data)
    reached_sink = False if ('level' not in end_node or end_node['level'] == -1) else True
    return reached_sink, level_graph

# A DFS based function to send flow after BFS has
# figured out that there is a possible flow and
# constructed levels. This functions called multiple
# times for a single call of BFS.
# 
# flow : Current flow send by parent function call
# start[] : To keep track of next edge to be explored
#           start[i] stores count of edges explored
#           from i
# u : Current vertex
# t : Sink

def DFS_sendFlow(graph, current_id, end_id, flow_in, path=[], paths=[]):
    # Sink reached
    if current_id == end_id:
        path.append(flow_in)
        paths.append(path.copy())
        print(f'reached end: {path}')
        return flow_in
    total_flow = 0

    current_node = graph.nodes[current_id]    
    # Traverse all adjacent nodes/edges one -by -one
    for edge in graph.edges(current_id, data=True):
        edge_data = edge[2]
        next_id = edge[1] # get the end node of this edge
        next_node = graph.nodes[next_id]
        residual_capacity = edge_data['capacity'] - edge_data['flow']
        
        # prunes dead ends by ensuring that:
        # 1. follow the level condition (level of the destination node = current node's level + 1).
        # 2. only explores edges where the residual capacity is positive.
        if (next_node['level'] == (current_node['level']+1)) and residual_capacity > 0:
            path.append({'start': current_id, 'end': next_id})
            
            # find minimum flow from u to t
            curr_flow_to_send = min(flow_in, residual_capacity)
            flow_sent = DFS_sendFlow(graph, next_id, end_id, curr_flow_to_send, path, paths)
            
            # only continue if flow is greater than zero
            if not (flow_sent and flow_sent > 0):
                continue
            
            path.pop()
            
            # add flow to current edge
            edge_data['flow'] += flow_sent
            
            # find the reverse edge
            for rev_edge in graph.edges(next_id, data=True):
                if rev_edge[1] == current_id:
                    rev_edge_data = rev_edge[2]
                    # subtract flow from reverse edge of current edge
                    if 'flow' not in rev_edge_data:
                        rev_edge_data['flow'] = -flow_sent
                        break
                    rev_edge_data['flow'] -= flow_sent
                    break
            
            flow_in -= flow_sent
            total_flow += flow_sent
            if flow_in == 0:
                break
    return total_flow

def reset_map(graph, true_level_graph):
    for current_id in true_level_graph['nodes']:
        current_node = graph.nodes[current_id]
        current_node['level'] = -1        
        for edge in graph.edges(current_id, data=True):
            #print(f'edge: {edge}')
            edge_data = edge[2]
            next_id = edge[1] # get the end node of this edge
            next_node = graph.nodes[next_id]
            next_node['level'] = -1
            edge_data['flow'] = 0
    return
    
def dinics(graph, start_id, end_id, shortest_dist=False):
    """Find the maximum flow from source to sink"""
    max_flow = 0
    paths = []
    level_graph = None
    true_level_graph = {'nodes': set(), 'edges': set()}
    while True:
        # 1. Build the level graph using BFS
        reached_sink, level_graph = BFS_buildLevelMap(graph, start_id, end_id, level_graph=level_graph, shortest_dist=shortest_dist)
        #print(f'old true paths list: {true_paths_list}, {true_paths}')
        true_level_graph['nodes'] |= level_graph['nodes']
        true_level_graph['edges'] |= level_graph['edges']
        #print(f'new true paths list: {true_paths_list}')
        if not reached_sink:
            break
        # 2. Find augmenting paths using DFS with dead-end pruning
        flow = DFS_sendFlow(graph, start_id, end_id, float('Inf'), paths=paths)
        if flow == 0:
            break
        max_flow += flow
    return max_flow, paths, true_level_graph

#================================================================================
def buildResidualGraph(graph):
    residual_graph = {}
    for edge in graph.edges(data=True):
        #print(edge[2])
        if 'capacity' not in edge[2]:
            edge[2]['capacity'] = 10
        u, v, capacity = edge[0], edge[1], edge[2]['capacity']

        # Initialize forward edge
        if u not in residual_graph:
            residual_graph[u] = {}
        if v not in residual_graph[u]:
            residual_graph[u][v] = capacity
        else:
            residual_graph[u][v] += capacity  # Sum capacities if multiple edges

        # Initialize reverse edge with 0 capacity
        if v not in residual_graph:
            residual_graph[v] = {}
        if u not in residual_graph[v]:
            residual_graph[v][u] = 0

    return residual_graph

def dfs(residual_graph, start, dest, visited, path):
    if start == dest:
        return path

    visited.add(start)

    for neighbor, capacity in residual_graph[start].items():
        if neighbor not in visited and capacity > 0:  # Only consider edges with positive capacity
            result = dfs(residual_graph, neighbor, dest, visited, path + [(start, neighbor, capacity)])
            if result is not None:
                return result
    return None

def fordFulkerson(paths, start, dest):
    residual_graph = buildResidualGraph(paths)
    max_flow = 0
    all_routes = []
    min_travel_times = []
    if start not in residual_graph:
        return 0, [], []
    while True:
        # Find an augmenting path using DFS
        visited = set()
        augmenting_path = dfs(residual_graph, start, dest, visited, [])

        if augmenting_path is None:
            break  # No more augmenting paths

        # Calculate the minimum travel time (bottleneck) along the path
        path_flow = min(edge[2] for edge in augmenting_path)

        # Store the path and its minimum travel time
        all_routes.append(augmenting_path)
        min_travel_times.append(path_flow)

        # Augment the flow along the path
        for u, v, capacity in augmenting_path:
            residual_graph[u][v] -= path_flow  # Reduce capacity in forward direction
            residual_graph[v][u] += path_flow  # Increase capacity in reverse direction

        max_flow += path_flow  # Increase the total max flow by the path's bottleneck capacity

    return max_flow, all_routes, min_travel_times,None

import networkx as nx

def create_test_graph():
    G = nx.DiGraph()
    edges = [
        (1, 2, {'flow': 0, 'capacity': 10}),
        (1, 3, {'flow': 0, 'capacity': 10}),
        (2, 5, {'flow': 0, 'capacity': 4}),
        (2, 3, {'flow': 0, 'capacity': 2}),
        (3, 4, {'flow': 0, 'capacity': 9}),
        (2, 4, {'flow': 0, 'capacity': 8}),
        (4, 5, {'flow': 0, 'capacity': 6}),
        (5, 6, {'flow': 0, 'capacity': 10}),
        (4, 6, {'flow': 0, 'capacity': 10})
    ]
    G.add_edges_from(edges)
    return G


np.random.seed(10)
def estimate_max_capacity(data):
    max_capacity = 0
    if data['highway']  == 'trunk':
      max_capacity = 500
    elif data['highway']  == ['trunk', 'primary'] or data['highway'] == ['tertiary', 'secondary']:
      max_capacity = 400
    elif data['highway'] == 'primary':
      max_capacity = 300
    elif data['highway'] == 'primary_link' or data['highway'] == 'secondary':
      max_capacity = 200
    elif data['highway'] == 'secondary_link':
      max_capacity = 150
    elif data['highway'] == 'tertiary':
      max_capacity = 100
    elif data['highway'] == 'tertiary_link' or data['highway'] == 'living_street':
      max_capacity = 70
    elif data['highway'] == 'residential':
      max_capacity = 30
    else:
      max_capacity = 20
    return round(max_capacity*(1+np.random.uniform(-0.1, 0.1)))

if __name__ == "__main__":
    #G = loadMap('minigraph.osm')
    #G = loadMap('newgraph_conso.osm')
    G = create_test_graph()
    print(G)
    #for node in G.nodes:
    #    print(f'node: {G.nodes[node]}')
    #    print(f'edge: {G.edges(node, data=True)}')
    #    print(f'edge i: {G.in_edges(node)}')
    #    print(f'edge o: {G.out_edges(node)}')
    
    #source, sink = 533, 352
    source, sink = 1, 6
    '''
    max_flow, paths_list, true_paths_list, true_level_graph = dinics(G, source, sink)
    #print(f'true paths list: {true_paths_list}')
    for path in paths_list:
        print(f'paths found: {path}')
    print(f'level graph: {true_level_graph}')
    # reset the map for later runs
    reset_map(G, true_level_graph)
    
    print('===============================')
    max_flow, paths_list, true_paths_list, true_level_graph = dinics(G, source, sink)
    #print(f'true paths list: {true_paths_list}')
    for path in paths_list:
        print(f'paths found: {path}')
        
    print(max_flow)
    '''

    # create capacity for edges
    # for edge in G.edges(data=True):
    #     #print(f'old edge: {edge}')
    #     edge[2]['capacity'] = estimate_max_capacity(edge[2])
    #     #print(f'new edge: {edge}')
        
    max_flow, paths, true_level_graph = dinics(G, source, sink)
    print(max_flow)
    print(paths)
    print(true_level_graph)
    #print(level_graph)
