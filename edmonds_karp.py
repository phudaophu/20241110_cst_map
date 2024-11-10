import networkx as nx
from collections import deque

def find_maximum_flow_using_edmonds_karp(graph, source, sink):
    """
    Finds the maximum flow in a graph from a source node to a sink node using the Edmonds-Karp algorithm.
    
    Args:
        graph (networkx.DiGraph): A directed graph with capacity attributes on edges.
        source (int): The source node.
        sink (int): The sink node.
    
    Returns:
        tuple: (flow_value, all_routes)
            - flow_value: The maximum flow from the source to the sink
            - all_routes: A list of routes, where each route is a list of nodes
    """
    
    def bfs(residual_graph, source, sink, parent):
        visited = {node: False for node in residual_graph}
        queue = deque()
        queue.append(source)
        visited[source] = True
        

        while queue:
            u = queue.popleft()
            
            for v, data in residual_graph[u].items():
                capacity = data['capacity']
                if not visited[v] and capacity > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u
        
        return visited[sink]
    
    def get_path(parent, source, sink):
        """
        Reconstructs the path from source to sink using parent dictionary.
        
        Args:
            parent (dict): Dictionary containing the path information
            source (int): Source node
            sink (int): Sink node
            
        Returns:
            list: The path from source to sink
        """
        path = []
        current = sink
        while current != source:
            path.append(current)
            current = parent[current]
        path.append(source)
        return list(reversed(path))

    if not (graph.has_node(source) and graph.has_node(sink)):
        return 0, []

    residual_graph = {node: {} for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        residual_graph[u][v] = {'capacity': data['capacity']}
        if v not in residual_graph[u]:
            residual_graph[u][v] = {'capacity': 0}
        if u not in residual_graph[v]:
            residual_graph[v][u] = {'capacity': 0}

    max_flow = 0
    paths = [] 
    parent = {node: None for node in graph.nodes()}

    while bfs(residual_graph, source, sink, parent):
        path_flow = float('inf')
        path = get_path(parent, source, sink)
        
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            path_flow = min(path_flow, residual_graph[u][v]['capacity'])
        
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            residual_graph[u][v]['capacity'] -= path_flow
            residual_graph[v][u]['capacity'] += path_flow
        
        paths.append(path)
        max_flow += path_flow
        
        parent = {node: None for node in graph.nodes()}

    return max_flow, paths, []


def find_maximum_flow_using_edmonds_karp_multidigraph(graph, source, sink):
    """
    Finds the maximum flow in an OSM MultiDiGraph from source node to sink node using Edmonds-Karp algorithm.
    
    Args:
        graph (networkx.MultiDiGraph): An OSM MultiDiGraph with capacity attributes on edges
        source (int): The source node OSM ID
        sink (int): The sink node OSM ID
    
    Returns:
        tuple: (flow_value, all_routes)
            - flow_value: The maximum flow from source to sink
            - all_routes: List of all routes from source and sink suitable for OSMnx visualization
    """
    def get_edge_capacity(u, v):
        if v not in graph[u]:
            return 0
        return sum(data.get('capacity', 0) for data in graph[u][v].values())

    def get_residual_graph():
        residual = {}
        for u, v, data in graph.edges(data=True):
            if u not in residual:
                residual[u] = {}
            if v not in residual:
                residual[v] = {}
            
            # Add forward edge
            if v not in residual[u]:
                residual[u][v] = 0
            residual[u][v] += data.get('capacity', 0)
            
            # Add reverse edge if it doesn't exist
            if u not in residual[v]:
                residual[v][u] = 0
                
        return residual

    def bfs(residual):
        """
        Use BFS to find a path from source to sink in the residual graph.
        
        Returns:
            list: Path from source to sink, or None if no path exists
        """
        parent = {source: None}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            for v in residual[u]:
                if v not in parent and residual[u][v] > 0:
                    parent[v] = u
                    if v == sink:
                        # Reconstruct path
                        path = []
                        curr = sink
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        return list(reversed(path))
                    queue.append(v)
        return None

    def get_path_flow(path, residual):
        flow = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow = min(flow, residual[u][v])
        return flow

    residual_graph = get_residual_graph()
    max_flow = 0
    all_routes = []

    while True:
        path = bfs(residual_graph)
        if not path:
            break
            
        path_flow = get_path_flow(path, residual_graph)
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual_graph[u][v] -= path_flow 
            residual_graph[v][u] += path_flow
        
        max_flow += path_flow
        
        if path_flow > 0.1:
            all_routes.append(path)

    if len(all_routes) == 0:
        return 0, []
    elif len(all_routes) == 1 and all_routes[0]:
        all_routes.append(all_routes[0])

    return max_flow, all_routes