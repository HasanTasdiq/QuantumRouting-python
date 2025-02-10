import numpy as np
import networkx as nx
import random

class MultiLinkACO:
    def __init__(self, graph, n_ants, n_iterations, alpha=1, beta=2, rho=0.5, q=1):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Heuristic influence
        self.rho = rho  # Pheromone evaporation rate
        self.q = q  # Pheromone deposit factor
        self.pheromone = {}  # Pheromone levels for each link
        
        # Initialize pheromone values for each link in multi-link edges
        for u, v, key in self.graph.edges(keys=True):
            self.pheromone[(u, v, key)] = 1.0  

    def run_episode(self, num_requests):
        """Runs an episode with random (source, destination) pairs."""
        link_availability = {edge: True for edge in self.graph.edges(keys=True)}  # Track available links

        sources, destinations = self.generate_random_requests(num_requests)
        best_paths = {}
        best_lengths = {}

        for source, destination in zip(sources, destinations):
            best_path, best_length = self.aco_search(source, destination, link_availability)
            if best_path:
                best_paths[(source, destination)] = best_path
                best_lengths[(source, destination)] = best_length
                # Mark links as consumed
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    # Find the first available link and consume it
                    for key in self.graph[u][v]:
                        if link_availability.get((u, v, key), False):
                            link_availability[(u, v, key)] = False
                            break  # Consume only one link per edge

        return best_paths, best_lengths

    def generate_random_requests(self, num_requests):
        """Generates random (source, destination) pairs."""
        nodes = list(self.graph.nodes)
        sources = random.sample(nodes, num_requests)
        destinations = random.sample(nodes, num_requests)
        return sources, destinations

    def aco_search(self, start, end, link_availability):
        """ACO search for a single source-destination pair while respecting link constraints."""
        best_path = None
        best_length = float("inf")

        for _ in range(self.n_iterations):
            paths = []
            lengths = []

            for _ in range(self.n_ants):
                path, length = self.construct_solution(start, end, link_availability)
                if path:
                    paths.append(path)
                    lengths.append(length)

                    if length < best_length:
                        best_path, best_length = path, length

            self.update_pheromone(paths, lengths)

        return best_path, best_length

    def construct_solution(self, start, end, link_availability):
        """Constructs a routing path for an ant while avoiding used links."""
        path = [start]
        current = start
        total_length = 0

        while current != end:
            neighbors = list(self.graph.neighbors(current))
            valid_neighbors = []

            for neighbor in neighbors:
                available_links = [key for key in self.graph[current][neighbor] if link_availability.get((current, neighbor, key), False)]
                if available_links:
                    valid_neighbors.append((neighbor, available_links))

            if not valid_neighbors:
                return None, float("inf")  # No valid path found

            probabilities = self.calculate_probabilities(current, valid_neighbors)
            chosen_neighbor, chosen_key = self.select_next_hop(valid_neighbors, probabilities)

            path.append(chosen_neighbor)
            total_length += self.graph[current][chosen_neighbor][chosen_key]['weight']
            current = chosen_neighbor

        return path, total_length

    def calculate_probabilities(self, current, valid_neighbors):
        """Calculates probabilities for the next node selection."""
        pheromone_values = []
        heuristic_values = []

        for neighbor, available_links in valid_neighbors:
            link = available_links[0]  # Choose the first available link for now
            pheromone_values.append(self.pheromone[(current, neighbor, link)])
            heuristic_values.append(1 / (self.graph[current][neighbor][link]['weight'] + 1e-6))

        pheromone_values = np.array(pheromone_values)
        heuristic_values = np.array(heuristic_values)

        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities /= probabilities.sum()
        return probabilities

    def select_next_hop(self, valid_neighbors, probabilities):
        """Selects the next node based on probability."""
        index = np.random.choice(len(valid_neighbors), p=probabilities)
        return valid_neighbors[index]

    def update_pheromone(self, paths, lengths):
        """Updates pheromone levels based on successful paths."""
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)  # Evaporation

        for path, length in zip(paths, lengths):
            pheromone_to_add = self.q / (length + 1e-6)
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                for key in self.graph[u][v]:
                    self.pheromone[(u, v, key)] += pheromone_to_add
                    break  # Update only one link per edge

# Generate a Large Random Network
num_nodes = 100
num_edges = 500  # High connectivity
graph = nx.MultiGraph()

# Add random weighted edges (multiple links per pair)
for _ in range(num_edges):
    u, v = random.sample(range(num_nodes), 2)
    weight = random.randint(1, 10)  # Random weight for cost
    graph.add_edge(u, v, weight=weight)  # MultiGraph allows multiple edges

aco = MultiLinkACO(graph, n_ants=20, n_iterations=10)

# Run an episode with 50 random (source, destination) pairs
num_requests = 50
best_paths, best_lengths = aco.run_episode(num_requests)

print("\nRouting Results (Large Network):")
for key in list(best_paths.keys())[:10]:  # Print only first 10 results
    print(f"  Best Path {key}: {best_paths[key]}, Length: {best_lengths[key]}")
