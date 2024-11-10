from collections import deque
from networkx.algorithms.flow import dinitz

# BFS from given source s
def bfs(adj, s):
  
    # Create a queue for BFS
    q = deque()
    
    # Initially mark all the vertices as not visited When we push a node into the q, we mark it as visited
    visited = [False] * len(adj)

    # Mark the source node as visited and enqueue it
    visited[s] = True
    q.append(s)

    # Iterate over the queue
    while q:
      
        # Dequeue a node from queue and print it
        curr = q.popleft()
        print(curr, end=" ")

        # Get all adjacent vertices of the dequeued node. 
        # If an adjacent has not been visited, mark it visited and enqueue it
        for x in adj[curr]:
            if not visited[x]:
                visited[x] = True
                q.append(x)
    
    # queue's view each loop:
    # pop left <- [node_A, node_B,..., node_N]
    # => [node_B..., node_N, node_A_child_1, node_A_child_2,...,node_A_child_M] <- push A's children
    

from collections import deque

# A class to represent the Dinic's algorithm for maximum flow
class Dinic:
    def __init__(self, n):
        self.n = n  # Number of vertices
        self.graph = [[] for _ in range(n)]  # Adjacency list to store the graph
        self.level = [-1] * n  # Level of each node
        self.ptr = [0] * n  # Pointer for DFS

    # Adds a directed edge from u to v with a given capacity
    def add_edge(self, u, v, cap):
        # Forward edge from u to v with capacity 'cap'
        self.graph[u].append([v, cap, len(self.graph[v])])
        # Backward edge from v to u with 0 initial capacity
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    # Performs BFS to build the level graph
    def bfs(self, source, sink):
        queue = deque([source])
        self.level = [-1] * self.n
        self.level[source] = 0

        while queue:
            u = queue.popleft()
            for v, cap, rev in self.graph[u]:
                if self.level[v] == -1 and cap > 0:  # Not yet visited and has residual capacity
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[sink] != -1  # If the sink is reachable

    # Performs DFS to send flow along augmenting paths
    def dfs(self, u, sink, flow):
        if u == sink:
            return flow

        while self.ptr[u] < len(self.graph[u]):
            v, cap, rev = self.graph[u][self.ptr[u]]
            if self.level[v] == self.level[u] + 1 and cap > 0:
                # The available flow is the minimum of current flow and the edge's capacity
                pushed = self.dfs(v, sink, min(flow, cap))
                if pushed > 0:
                    # Augment the flow along the path
                    self.graph[u][self.ptr[u]][1] -= pushed  # Reduce capacity in forward edge
                    self.graph[v][rev][1] += pushed  # Increase capacity in backward edge
                    return pushed
            self.ptr[u] += 1

        return 0

    # Main function to calculate the maximum flow
    def max_flow(self, source, sink):
        flow = 0
        while True:
            if not self.bfs(source, sink):  # If there is no augmenting path, we're done
                break
            self.ptr = [0] * self.n  # Reset the pointer for DFS
            while True:
                pushed = self.dfs(source, sink, float('Inf'))
                if pushed == 0:
                    break
                flow += pushed

        return flow


# Python implementation of Dinic's Algorithm
class Edge:
    def __init__(self, v, flow, C, rev):
        self.v = v
        self.flow = flow
        self.C = C
        self.rev = rev
 
# Residual Graph
 
 
class Graph:
    def __init__(self, V):
        self.adj = [[] for i in range(V)]
        self.V = V
        self.level = [0 for i in range(V)]
 
    # add edge to the graph
    def addEdge(self, u, v, C):
 
        # Forward edge : 0 flow and C capacity
        a = Edge(v, 0, C, len(self.adj[v]))
 
        # Back edge : 0 flow and 0 capacity
        b = Edge(u, 0, 0, len(self.adj[u]))
        self.adj[u].append(a)
        self.adj[v].append(b)
 
    # Finds if more flow can be sent from s to t
    # Also assigns levels to nodes
    def BFS(self, s, t):
        for i in range(self.V):
            self.level[i] = -1
 
        # Level of source vertex
        self.level[s] = 0
 
        # Create a queue, enqueue source vertex
        # and mark source vertex as visited here
        # level[] array works as visited array also
        q = []
        paths = []
        q.append(s)
        paths.append([s])
        print(paths)
        idx = 0
        while q:
            u = q.pop(0)
            path = paths[idx]
            #print(idx, path)
            for i in range(len(self.adj[u])):
                e = self.adj[u][i]
                if self.level[e.v] < 0 and e.flow < e.C:
 
                    # Level of current vertex is
                    # level of parent + 1
                    self.level[e.v] = self.level[u]+1
                    q.append(e.v)
                    paths.append(path + [e.v])
            idx+=1
        print(paths)
 
        # If we can not reach to the sink we
        # return False else True
        return False if self.level[t] < 0 else True
 
# A DFS based function to send flow after BFS has
# figured out that there is a possible flow and
# constructed levels. This functions called multiple
# times for a single call of BFS.
# flow : Current flow send by parent function call
# start[] : To keep track of next edge to be explored
#           start[i] stores count of edges explored
#           from i
# u : Current vertex
# t : Sink
    def sendFlow(self, u, flow, t, start):
        print(f'u: {u} | t: {t} | flow: {flow} | start: {start}')
        # Sink reached
        if u == t:
            return flow
 
        # Traverse all adjacent edges one -by -one
        while start[u] < len(self.adj[u]):
            # Pick next edge from adjacency list of u
            e = self.adj[u][start[u]]
            print(e.v)
            # if reached a node of next level that still have flow < max capacity
            if self.level[e.v] == self.level[u]+1 and e.flow < e.C:
 
                # find minimum flow from u to t
                curr_flow = min(flow, e.C-e.flow)
                temp_flow = self.sendFlow(e.v, curr_flow, t, start)
 
                # flow is greater than zero
                if temp_flow and temp_flow > 0:
 
                    # add flow to current edge
                    e.flow += temp_flow
 
                    # subtract flow from reverse edge
                    # of current edge
                    print(f'reverse edge {e.v}: {self.adj[e.v][e.rev].flow}')
                    self.adj[e.v][e.rev].flow -= temp_flow
                    return temp_flow
            start[u] += 1
 
    # Returns maximum flow in graph
    def DinicMaxflow(self, s, t):
 
        # Corner case
        if s == t:
            return -1
 
        # Initialize result
        total = 0
 
        # Augument the flow while there is path
        # from source to sink
        while self.BFS(s, t) == True:
 
            # store how many edges are visited
            # from V { 0 to V }
            start = [0 for i in range(self.V+1)]
            while True:
                flow = self.sendFlow(s, float('inf'), t, start)
                if not flow:
                    break
 
                # Add path flow to overall flow
                total += flow
 
        # return maximum flow
        return total
 
 
# This code is contributed by rupasriachanta421.

# Driver code
if __name__ == '__main__':
    '''
    n, m = map(int, input().split())  # Number of nodes and edges
    dinic = Dinic(n)

    for _ in range(m):
        u, v, cap = map(int, input().split())  # Add edge u -> v with capacity cap
        dinic.add_edge(u, v, cap)

    source, sink = map(int, input().split())  # Source and sink nodes
    print("Maximum Flow:", dinic.max_flow(source, sink))
    '''
    g = Graph(6)
    g.addEdge(0, 1, 10)
    g.addEdge(0, 2, 10)
    g.addEdge(1, 2, 2)
    g.addEdge(1, 3, 4)
    g.addEdge(1, 4, 8)
    g.addEdge(2, 4, 9)
    g.addEdge(3, 5, 10)
    g.addEdge(4, 3, 6)
    g.addEdge(4, 5, 10)
    print("Maximum flow", g.DinicMaxflow(0, 5))
 

