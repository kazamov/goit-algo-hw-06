import networkx as nx
import matplotlib.pyplot as plt


def create_graph():
    graph = nx.Graph()

    graph.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"])

    edges_with_weights = [
        ("A", "B", 3),
        ("B", "C", 2),
        ("C", "D", 1),
        ("D", "E", 4),
        ("E", "F", 2),
        ("F", "G", 3),
        ("G", "H", 5),
        ("H", "I", 2),
        ("I", "J", 4),
        ("J", "K", 1),
        ("K", "L", 3),
        ("L", "A", 5),
        ("B", "E", 2),
        ("D", "G", 1),
        ("F", "I", 3),
        ("H", "K", 4),
        ("A", "C", 2),
        ("L", "B", 1),
    ]

    graph.add_weighted_edges_from(edges_with_weights)

    return graph


def describe_graph(graph):
    print("Кількість вершин графа:", len(graph.nodes()))
    print("Кількість ребер графа:", graph.edges())
    print("Ваги ребер графа:", nx.get_edge_attributes(graph, "weight"))

    degree_sequence = dict(graph.degree())
    print("Ступені вершин графа:")
    for node, degree in degree_sequence.items():
        print(f"    Ступінь вершини {node}: {degree}")


def visualize_graph(graph):
    pos = nx.circular_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        font_weight="bold",
        node_size=700,
        node_color="skyblue",
        font_color="black",
        font_size=8,
        edge_color="gray",
        width=[float(d["weight"]) for (u, v, d) in graph.edges(data=True)],
    )
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()


def dijkstra(graph, start):
    distances = {node: float("inf") for node in graph.nodes()}
    distances[start] = 0

    visited = {node: False for node in graph.nodes()}

    while False in visited.values():
        current_node = min(
            [node for node in graph.nodes() if visited[node] is False],
            key=lambda node: distances[node],
        )
        visited[current_node] = True

        for neighbour, weight in graph[current_node].items():
            if distances[current_node] + weight["weight"] < distances[neighbour]:
                distances[neighbour] = distances[current_node] + weight["weight"]

    return distances


if __name__ == "__main__":
    # Завдання 1
    G = create_graph()
    describe_graph(G)

    # Завдання 2
    print("DFS-дерево:")
    dfs_tree = nx.dfs_tree(G, source="A")
    print(list(dfs_tree.edges()))

    print("BFS-дерево:")
    bfs_tree = nx.bfs_tree(G, source="A")
    print(list(bfs_tree.edges()))

    # Завдання 3
    print("Найкоротші відстані від вершини A:")
    print(dijkstra(G, "A"))

    visualize_graph(G)
