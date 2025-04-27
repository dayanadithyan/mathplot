# algorithms/graph_algorithms.py - Graph theory algorithms

import heapq
import numpy as np
import math

def shortest_path_dijkstra(graph, start, end):
    """Find shortest path between two nodes using Dijkstra's algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        start: Starting node
        end: Ending node
        
    Returns:
        tuple: (distance, path) - Total distance and node path
    """
    # Initialize
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    unvisited = set(graph.keys())
    
    while unvisited:
        # Find the unvisited node with the smallest distance
        current = min(unvisited, key=lambda x: distances[x])
        
        # If we reached the end node or if the smallest distance is infinity, stop
        if current == end or distances[current] == float('inf'):
            break
        
        # Remove the current node from unvisited
        unvisited.remove(current)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current]:
            distance = distances[current] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
    
    # Reconstruct the path
    if distances[end] == float('inf'):
        return float('inf'), []  # No path found
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return distances[end], path

def minimum_spanning_tree_kruskal(graph):
    """Find minimum spanning tree using Kruskal's algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        list: List of (start, end, weight) edges in the MST
    """
    # Extract edges and sort by weight
    edges = []
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            if node < neighbor:  # Avoid duplicates for undirected graph
                edges.append((node, neighbor, weight))
    
    edges.sort(key=lambda x: x[2])
    
    # Initialize disjoint set data structure
    parent = {node: node for node in graph}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Run Kruskal's algorithm
    mst_edges = []
    
    for edge in edges:
        start, end, weight = edge
        if find(start) != find(end):
            union(start, end)
            mst_edges.append(edge)
    
    return mst_edges

def color_graph_greedy(graph):
    """Color a graph using a greedy algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        dict: Node colors {node: color_index}
    """
    # Get nodes sorted by degree (number of neighbors)
    nodes = sorted(graph.keys(), key=lambda x: len(graph[x]), reverse=True)
    
    # Initialize colors
    colors = {}
    
    for node in nodes:
        # Get colors of neighbors
        neighbor_colors = set()
        for neighbor, _ in graph[node]:
            if neighbor in colors:
                neighbor_colors.add(colors[neighbor])
        
        # Find the first available color
        color = 0
        while color in neighbor_colors:
            color += 1
        
        # Assign color
        colors[node] = color
    
    return colors

def detect_cycles(graph):
    """Detect cycles in a graph using DFS.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        list: List of cycles as node lists
    """
    visited = set()
    parent = {}
    cycles = []
    
    def dfs(node, current_path):
        visited.add(node)
        current_path.add(node)
        
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                dfs(neighbor, current_path)
            elif neighbor in current_path and parent.get(node) != neighbor:
                # Found a cycle
                cycle = []
                current = node
                while current != neighbor:
                    cycle.append(current)
                    current = parent[current]
                cycle.append(neighbor)
                cycle.append(node)  # Complete the cycle
                cycles.append(cycle)
        
        current_path.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node, set())
    
    return cycles

def force_directed_layout(graph, iterations=100, k=0.1, progress_callback=None):
    """Generate force-directed layout for a graph.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        iterations (int): Number of iterations
        k (float): Spring constant
        progress_callback (callable): Function to report progress
        
    Returns:
        dict: Node positions {node: (x, y, z)}
    """
    # Initialize random positions
    positions = {node: (np.random.uniform(-1, 1), 
                      np.random.uniform(-1, 1), 
                      np.random.uniform(-1, 1)) 
               for node in graph}
    
    # Number of nodes
    n = len(graph)
    
    # Optimal distance between nodes
    optimal_distance = k * math.sqrt(3.0 / n)
    
    # Run iterations
    for iter in range(iterations):
        # Report progress
        if progress_callback:
            progress = (iter + 1) / iterations
            if not progress_callback(progress, f"Layout iteration {iter+1}/{iterations}"):
                break
        
        # Initialize displacement vectors
        displacement = {node: [0, 0, 0] for node in graph}
        
        # Calculate repulsive forces between all pairs of nodes
        for node1 in graph:
            for node2 in graph:
                if node1 != node2:
                    # Calculate vector from node2 to node1
                    dx = positions[node1][0] - positions[node2][0]
                    dy = positions[node1][1] - positions[node2][1]
                    dz = positions[node1][2] - positions[node2][2]
                    
                    # Distance (avoid division by zero)
                    distance = max(0.01, math.sqrt(dx*dx + dy*dy + dz*dz))
                    
                    # Repulsive force is inversely proportional to distance
                    force = optimal_distance * optimal_distance / distance
                    
                    # Add to displacement
                    displacement[node1][0] += dx / distance * force
                    displacement[node1][1] += dy / distance * force
                    displacement[node1][2] += dz / distance * force
        
        # Calculate attractive forces along edges
        for node1 in graph:
            for node2, _ in graph[node1]:
                # Calculate vector from node1 to node2
                dx = positions[node2][0] - positions[node1][0]
                dy = positions[node2][1] - positions[node1][1]
                dz = positions[node2][2] - positions[node1][2]
                
                # Distance (avoid division by zero)
                distance = max(0.01, math.sqrt(dx*dx + dy*dy + dz*dz))
                
                # Attractive force is proportional to distance
                force = distance * distance / optimal_distance
                
                # Add to displacement
                displacement[node1][0] += dx / distance * force
                displacement[node1][1] += dy / distance * force
                displacement[node1][2] += dz / distance * force
        
        # Apply displacements with temperature (gradually reduced)
        temperature = 1.0 - iter / iterations
        for node in graph:
            disp = displacement[node]
            
            # Limit maximum displacement
            disp_length = max(0.01, math.sqrt(disp[0]*disp[0] + disp[1]*disp[1] + disp[2]*disp[2]))
            scaling = min(temperature, disp_length) / disp_length
            
            # Update position
            x, y, z = positions[node]
            positions[node] = (
                x + disp[0] * scaling,
                y + disp[1] * scaling,
                z + disp[2] * scaling
            )
    
    # Scale positions to a reasonable range
    max_distance = 0
    for pos in positions.values():
        distance = math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
        max_distance = max(max_distance, distance)
    
    scale_factor = 1.0 / max(0.01, max_distance)
    for node in positions:
        positions[node] = tuple(coord * scale_factor for coord in positions[node])
    
    return positions

# Registration functions
def register():
    """Register graph theory algorithms"""
    print("Math Playground: Graph theory algorithms registered")

def unregister():
    """Unregister graph theory algorithms"""
    print("Math Playground: Graph theory algorithms unregistered")