import numpy as np


def find_central_hub(graph, graphSize, n):
    from collections import defaultdict, deque

    # Build adjacency list
    adj_list = defaultdict(list)
    for u, v in graph:
        adj_list[u].append(v)
        adj_list[v].append(u)

    def bfs_paths_count(start):
        # BFS to find shortest paths and path counts from start node
        queue = deque([start])
        distances = {start: 0}
        paths_count = {start: 1}
        parents = defaultdict(list)

        while queue:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if neighbor not in distances:
                    queue.append(neighbor)
                    distances[neighbor] = distances[node] + 1
                if distances[neighbor] == distances[node] + 1:
                    paths_count[neighbor] = paths_count.get(neighbor, 0) + paths_count[node]
                    parents[neighbor].append(node)

        return distances, paths_count, parents

    centrality = [0] * n

    for node in range(n):
        distances, paths_count, parents = bfs_paths_count(node)
        dependencies = {v: 0 for v in distances}

        nodes_by_distance = sorted(distances, key=lambda x: distances[x], reverse=True)

        for w in nodes_by_distance:
            for v in parents[w]:
                dependencies[v] += (paths_count[v] / paths_count[w]) * (1 + dependencies[w])
            if w != node:
                centrality[w] += dependencies[w]

    # Find the node with the highest centrality
    max_centrality = max(centrality)
    central_hub = min(node for node in range(n) if centrality[node] == max_centrality)

    return central_hub


input_str = input().strip()


def parse_graph_from_string(input_str):
    edges = []
    max_value = 0
    input_str = input_str.strip()
    while input_str:
        start = input_str.find('(')
        end = input_str.find(')', start)
        if start == -1 or end == -1:
            break
        edge_str = input_str[start + 1:end]
        x, y = map(int, edge_str.split(','))
        edges.append([x, y])
        max_value = max(max_value, x, y)
        input_str = input_str[end + 1:]
    return edges, len(edges), max_value + 1


# Call candidate function
graph, n, maxValue = parse_graph_from_string(input_str)
result = find_central_hub(graph, n, maxValue)

# Part 4 - Print result
print(result)
