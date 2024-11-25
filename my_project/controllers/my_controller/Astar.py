import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = 0  # Coste desde el inicio hasta el nodo actual
        self.h = 0  # Heurística (distancia estimada al objetivo)
        self.f = 0  # Costo total (g + h)

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Calcula la distancia Manhattan entre dos puntos a y b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(map_2d, start, end):
    """Algoritmo A* para encontrar el camino más corto en un mapa 2D."""
    open_list = []
    closed_list = set()

    start_node = Node(start)
    end_node = Node(end)
    heapq.heappush(open_list, start_node)

    while open_list:
        # Selecciona el nodo con el menor costo f
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        # Si llegamos al objetivo, reconstruimos el camino
        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Camino de inicio a fin

        # Genera vecinos (arriba, abajo, izquierda, derecha)
        neighbors = [
            (0, 1), (0, -1), (1, 0), (-1, 0)  # Derecha, izquierda, abajo, arriba
        ]
        for dx, dy in neighbors:
            neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

            # Verifica si el vecino está dentro del mapa y es transitable
            if (0 <= neighbor_pos[0] < len(map_2d) and
                    0 <= neighbor_pos[1] < len(map_2d[0]) and
                    map_2d[neighbor_pos[0]][neighbor_pos[1]] == 0 and
                    neighbor_pos not in closed_list):

                neighbor_node = Node(neighbor_pos, current_node)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = heuristic(neighbor_pos, end_node.position)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                # Añadir a la open list si no está ya con mejor f
                if all(open_node.position != neighbor_node.position or neighbor_node.f < open_node.f for open_node in open_list):
                    heapq.heappush(open_list, neighbor_node)

    return None  # No se encontró un camino
