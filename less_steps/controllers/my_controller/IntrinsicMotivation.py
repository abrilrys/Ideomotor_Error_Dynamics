import numpy as np
import random
import tools
import csv
import heapq


motor_data = []
gps_data = []

# Leer los datos del motor del CSV
with open("motor_angles.csv", "r", newline='') as motor_csvfile:
    motor_reader = csv.reader(motor_csvfile)
   
    for row in motor_reader:
        motor_data.append([float(value) for value in row[1:]])  # Pasar la columna del indice
    
# Leer los datos del GPS del CSV
with open("gps_hand.csv", "r", newline='') as gps_csvfile:
    gps_reader = csv.reader(gps_csvfile)
   
    for row in gps_reader:
        gps_data.append([float(value) for value in row[1:]])  # Pasar la columna del indice
            
            
class IntrinsicMotivation:
    """
    Designed to manage tasks and policies in a self-organizing map (SOM) learning environment.
    
    """    
    #self.lengthOfBuffers
    #self.task_dictionary
    #self.slopes
    #self.buffers
    
    def __init__(self, somVisual, somAngles, hebbianTable, robot):
        self.robot=robot
        self.lengthOfBuffers=3
        self.somVisual=somVisual
        self.somAngles=somAngles
        self.hebbian_table=hebbianTable
        # self.task_dictionary=self.initialize_task_dictionary()
        # self.slopes=self.get_slopes()
        #self.print_task_dict()
        #print(self.buffers)
    
    # def initDict(self): 
    #     #iterate until the task dictionary is correctly initialized (with executable goals and starting points)
    #     while(1):
    #         self.task_dictionary=self.initialize_task_dictionary()
    #         if(self.task_dictionary!= None):
    #             break
            
    #     #get the overall task performance
    #     self.updateTaskPerformance()
        
    def initDict(self): 
        self.task_dictionary=self.initialize_task_dictionary()
        self.updateTaskPerformance()
        
    # def initialize_task_dictionary(self):
    #     """
    #      Initializes a dictionary containing task-related data for the robot's SOM-based learning process.

    #         The method performs the following steps:
            
    #         1. Randomly selects 10 unique pairs of coordinates to represent tasks.
    #         2. For each task, initializes 4 policies, where each policy contains:
    #         - A set of 10 coordinates (the first being one of the task's pair coordinates).
    #         - A buffer of length 11, initialized with increasing values.
    #         3. For each task and policy, calculates predictive error.
    #         5. Updates the policy's buffer with the calculated predictive error for each coordinate.

    #         Returns:
    #         A dictionary containing task data, where each task has its selected coordinate pair, policies and associated buffers.
    #     """    
        
    #     # Define the dimensions of the SOM
    #     som_height = self.somVisual.get_weights().shape[0]  # Number of rows
    #     som_width = self.somVisual.get_weights().shape[1]   # Number of columns
        
    #     # Generate all possible coordinates
    #     all_coordinates = [(row, col) for row in range(som_height) for col in range(som_width)]
        
    #     # Shuffle the list of coordinates
    #     random.shuffle(all_coordinates)
        
    #     # Select 10 unique pairs of coordinates without repetition
    #     selected_pairs = []
    #     selected_pairs_count = 0
        
    #     self.numberOfTasks = 10
        
        
    #     while selected_pairs_count < self.numberOfTasks:
    #         # Select two random coordinates
    #         coord1 = self.find_executable_neuron(all_coordinates)
    #         coord2 = self.find_executable_neuron(all_coordinates)
            
    #         # Ensure the pair is unique and not repeated
    #         if coord1 != coord2 and (coord1, coord2) not in selected_pairs and (coord2, coord1) not in selected_pairs:
    #             selected_pairs.append((coord1, coord2))
    #             selected_pairs_count += 1
        
    #     # Create a dictionary to store the data
    #     data_dict = {}
        
    #     self.numberOfPolicies = 4
    #     self.lengthOfPolicies = self.lengthOfBuffers-1
        
    #     lengthOfBuffers = self.lengthOfBuffers
        
    #     # Populate the dictionary with selected pairs, associated sets, and buffers
    #     for task_index, pair in enumerate(selected_pairs, start=0):
    #         sets_and_buffers = {}
    #         for policy_index in range(0, self.numberOfPolicies):
    #             set_pairs = []
    #             # Add random coordinates not equal to the pair coordinates
    #             while len(set_pairs) < self.lengthOfPolicies:
    #                 random_coord = random.choice(all_coordinates)
    #                 if random_coord != pair[0] and random_coord != pair[1]:
    #                     set_pairs.append(random_coord)
                        
                
    #             set_pairs.insert(0, pair[0])
    #             #print(set_pairs)
    #             # Initialize buffer with increasing values
    #             buffer = []
    #             valor = 10
    #             while len(buffer) < self.lengthOfBuffers:
    #                 buffer.append(valor)
    #                 valor += 10
    #             # Store the policy with its index
    #             sets_and_buffers[f"Policy_{policy_index}"] = {
    #                 "Set": tuple(set_pairs),
    #                 "Buffer": buffer
    #             }
    #             #print(sets_and_buffers)
    #         # Store the task with its index
    #         data_dict[f"Task_{task_index}"] = {
    #             "Coordinates": pair,
    #             "Sets_and_Buffers": sets_and_buffers
    #         }
        
    #     # # Calculate predictive error and update the buffer
    #     # for task, values in data_dict.items():
    #     #     pair = values["Coordinates"]
    #     #     #visual_goal = tools.denormalize_vector(self.somVisual.get_weights()[pair[1][0], pair[1][1]], gps_data)
    #     #     visual_goal = self.somVisual.get_weights()[pair[1][0], pair[1][1]]
    #     #     visual_goal=np.round(visual_goal,4)
    #     #     motor_angles_goal_coord = self.hebbian_table.getConectionsFromSOM1(visual_goal)
                    
    #     #     if motor_angles_goal_coord is not None:
    #     #         rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_goal_coord[0], motor_angles_goal_coord[1]], motor_data)
    #     #         real_goal = self.robot.getRealGpsGoal(rotation_angles)
    #     #     else:
    #     #         print("Goal not reachable, changing task")
    #     #         return None
                        
                        
    #     #     for policy, policy_data in values["Sets_and_Buffers"].items():
    #     #         set_pairs = policy_data["Set"]
    #     #         buffer = policy_data["Buffer"]
                
    #     #         # Calculate predictive error for each coordinate in set_pairs
    #     #         for idx, coord in enumerate(set_pairs):
    #     #             #visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
    #     #             visual_input=self.somVisual.get_weights()[coord[0], coord[1]]
    #     #             visual_input=np.round(visual_input,4)
    #     #             motor_angles_coord = self.hebbian_table.getConectionsFromSOM1(visual_input)
    #     #             if motor_angles_coord is not None:
    #     #                 rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
    #     #                 # Assume the final goal is the second coordinate of the pair
    #     #                 predictive_error = self.robot.GetPredError(rotation_angles, real_goal)
                        
    #     #                 # Store predictive error in the buffer
    #     #                 if idx < len(buffer):  # Ensure we don't go out of bounds
    #     #                     buffer[idx] = predictive_error  
        
    #     # List to store the buffers
    #     feature_vectors = []
        
    #     # Loop through data_dict to extract only the buffers
    #     for task, values in data_dict.items():
    #         for policy, policy_data in values["Sets_and_Buffers"].items():
    #             buffer = policy_data["Buffer"]
    #             feature_vectors.append(buffer)  # Append only the buffer
                
    #     self.buffers=feature_vectors
        
    #     # Convert the list of buffers to a NumPy array
    #     tasks_array = np.array(feature_vectors)
    #     # Save the NumPy array into a CSV file
    #     np.savetxt('tasks_train_dataset.csv', tasks_array, delimiter=',', fmt='%.6f')
        
    #     print("Buffers saved into 'tasks_train_dataset.csv'")
    #     return data_dict
    
    def expand_viable_area(self,start, end, current_area, som_height, som_width, executable_map, step=1):
        """Expande el área viable si no hay suficientes puntos."""
        min_row = max(min(start[0], end[0]) - step, 0)
        max_row = min(max(start[0], end[0]) + step, som_height - 1)
        min_col = max(min(start[1], end[1]) - step, 0)
        max_col = min(max(start[1], end[1]) + step, som_width - 1)

        expanded_area = set(
            (x, y)
            for x in range(min_row, max_row + 1)
            for y in range(min_col, max_col + 1)
            if executable_map[x][y] == 0 and (x, y) not in current_area and (x,y) != start and (x,y)!= end
        )
        return expanded_area
    
    def initialize_task_dictionary(self):
        som_height = self.somVisual.get_weights().shape[0]
        som_width = self.somVisual.get_weights().shape[1]
        executable_map = self.generate_executable_map(som_height, som_width)
        all_coordinates = [(row, col) for row in range(som_height) for col in range(som_width)]

        data_dict = {}
        self.numberOfTasks = 10
        self.numberOfPolicies = 4
        self.lengthOfPolicies = 2  # 10 puntos intermedios


        for task_index in range(self.numberOfTasks):
            while True:
                start = self.find_executable_neuron(all_coordinates)
                end = self.find_executable_neuron(all_coordinates)

                if start != end:
                    base_path = self.astar_with_length(executable_map, start, end, self.lengthOfPolicies+2)
                    print(type(base_path))
                    if base_path and len(base_path) == self.lengthOfPolicies + 2:  # A* incluye inicio y fin
                        base_path = base_path[1:-1]  # Excluir inicio y fin
                        break

            sets_and_buffers = {}
            for policy_index in range(self.numberOfPolicies):
                if policy_index == 0:
                    # Primera política: exactamente el camino A*
                    policy_path = [start] + base_path 
                else:
                    # Variar ligeramente las demás políticas
                    policy_path = [start]  # Añadir inicio
                    viable_points = set(
                        (x, y)
                        for x in range(min(start[0], end[0]), max(start[0], end[0]) + 1)
                        for y in range(min(start[1], end[1]), max(start[1], end[1]) + 1)
                        if executable_map[x][y] == 0 and (x, y) not in policy_path and (x,y) != end
                    )

                    # Expandir el área hasta encontrar suficientes puntos
                    step = 1
                    while len(viable_points) < self.lengthOfPolicies:
                        new_points = self.expand_viable_area(start, end, policy_path, som_height, som_width, executable_map, step)
                        viable_points.update(new_points)
                        step += 1
                        if step > som_height + som_width:  # Prevenir ciclos infinitos
                            raise ValueError(
                                f"No se encontraron suficientes puntos viables para la tarea {task_index}."
                            )

                    # Seleccionar puntos únicos al azar
                    policy_path += random.sample(viable_points, self.lengthOfPolicies)

                sets_and_buffers[f"Policy_{policy_index}"] = {
                    "Set": tuple(policy_path),
                    "Buffer": [10 * (i + 1) for i in range(self.lengthOfBuffers)]  # Buffer inicial
                }

            data_dict[f"Task_{task_index}"] = {
                "Coordinates": (start, end),
                "Sets_and_Buffers": sets_and_buffers
            }

        return data_dict



    def generate_policies_with_unique_points(self,map_2d, start, end, num_policies):
        """
        Genera varias políticas basadas en A*, asegurando que los puntos son únicos,
        excepto el primero, que debe coincidir con el inicio.
        """
        policies = []

        # Generar la política A* pura
        astar_path = self.astar_with_length(map_2d, start, end, self.lengthOfPolicies)
        if not astar_path:
            raise ValueError("No se pudo generar un camino válido con A*.")
        policies.append(astar_path)

        # Generar políticas variantes
        max_attempts = 100  # Límite de intentos para evitar bucles infinitos
        for _ in range(num_policies - 1):
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                variant = [start]  # Inicia con el punto inicial
                used_positions = set(variant)  # Mantiene un registro de los puntos usados
                for point in astar_path[1:-1]:  # Iterar entre el inicio (excluido) y el final (excluido)
                    # Generar un vecino válido dentro del rango de la meta
                    valid_neighbors = [
                        (point[0] + dx, point[1] + dy)
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        if (0 <= point[0] + dx < len(map_2d) and
                            0 <= point[1] + dy < len(map_2d[0]) and
                            map_2d[point[0] + dx][point[1] + dy] == 0 and
                            (point[0] + dx, point[1] + dy) not in used_positions and
                            (point[0] + dx, point[1] + dy) != end)  # No permitir puntos repetidos ni el final
                    ]
                    if valid_neighbors:
                        chosen_neighbor = random.choice(valid_neighbors)
                        variant.append(chosen_neighbor)
                        used_positions.add(chosen_neighbor)
                    else:
                        # Si no hay vecinos válidos, usar el punto original (pero debe ser único)
                        if point not in used_positions and point != end:
                            variant.append(point)
                            used_positions.add(point)
                        else:
                            break  # Salir si no se puede generar un camino único

                # Si la longitud es la deseada y todos los puntos son únicos, aceptar
                if len(variant) == self.lengthOfPolicies and len(set(variant)) == len(variant):
                    policies.append(variant)  # Añadir el fin al final
                    break

            # Si no se generó una política válida, usar A* como respaldo
            if attempt == max_attempts:
                print(f"Advertencia: No se pudo generar una variante única. Usando A* como política {len(policies)}.")
                policies.append(astar_path)

        return policies


    def astar_with_length(self,map_2d, start, end, path_length):
        """A* que detiene la búsqueda si encuentra un camino de longitud específica."""
        open_list = []
        closed_list = set()
        start_node = Node(start)
        heapq.heappush(open_list, start_node)

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)

            # Verificar si el camino es de la longitud deseada
            if len(self.reconstruct_path(current_node)) == path_length:
                return self.reconstruct_path(current_node)

            # Generar vecinos
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dx, dy in neighbors:
                neighbor_pos = (current_node.position[0] + dx, current_node.position[1] + dy)

                if (0 <= neighbor_pos[0] < len(map_2d) and
                    0 <= neighbor_pos[1] < len(map_2d[0]) and
                    map_2d[neighbor_pos[0]][neighbor_pos[1]] == 0 and
                    neighbor_pos not in closed_list):

                    neighbor_node = Node(neighbor_pos, current_node)
                    neighbor_node.g = current_node.g + 1
                    neighbor_node.h = heuristic(neighbor_pos, end)
                    neighbor_node.f = neighbor_node.g + neighbor_node.h

                    heapq.heappush(open_list, neighbor_node)

        return None  # No se encontró camino de longitud exacta


    def reconstruct_path(self,node):
        """Reconstruye el camino desde el nodo actual hasta el inicio."""
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    
    def find_executable_neuron(self, all_coordinates):
        """
        Attempts to find an executable neuron goal given an initial coordinate.
        
        Args:
            -initial_goal(pair): A neuron representing the goal coordinate
            -all_coordinates (list): The list of neurons 
        
        Returns:
            -initial_goal(pair): A neuron that has a conection in the hebbian table to a motor command
        """
        initial_goal=random.choice(all_coordinates)
        while True:
            visual_goal = self.somVisual.get_weights()[initial_goal[0], initial_goal[1]]
            motor_angles_goal_coord = self.hebbian_table.getConectionsFromSOM1(visual_goal)
            if motor_angles_goal_coord is not None:
                return initial_goal
            # Pick a new coordinate if the goal is not executable
            initial_goal = random.choice(all_coordinates)
            
    def find_executable_neuron_with_exception(self, all_coordinates, no_neuron):
        """
        Attempts to find an executable neuron goal given an initial coordinate.
        
        Args:
            -initial_goal(pair): A neuron representing the goal coordinate
            -all_coordinates (list): The list of neurons 
        
        Returns:
            -initial_goal(pair): A neuron that has a conection in the hebbian table to a motor command
        """
        initial_goal=random.choice(all_coordinates)
        while True:
            visual_goal = self.somVisual.get_weights()[initial_goal[0], initial_goal[1]]
            motor_angles_goal_coord = self.hebbian_table.getConectionsFromSOM1(visual_goal)
            if motor_angles_goal_coord is not None and initial_goal!=no_neuron:
                return initial_goal
            # Pick a new coordinate if the goal is not executable
            initial_goal = random.choice(all_coordinates)
            
    def generate_executable_map(self, som_height, som_width):
        """Genera un mapa de ejecutabilidad donde 0 es transitable y 1 es obstáculo."""
        executable_map = np.ones((som_height, som_width), dtype=int)
        for row in range(som_height):
            for col in range(som_width):
                visual_goal = self.somVisual.get_weights()[row, col]
                if self.hebbian_table.getConectionsFromSOM1(visual_goal) is not None:
                    executable_map[row, col] = 0  # Es ejecutable
        return executable_map

    def updateTaskPerformance(self):
        overallTaskPerformance = []

        for task, values in self.task_dictionary.items():
            task_performance = []

            for policy, policy_data in values["Sets_and_Buffers"].items():
                buffer = policy_data["Buffer"]
                
                # Sum 1 each time the buffer is decreasing
                decrease_count = sum(1 for i in range(1, len(buffer)) if buffer[i] < buffer[i - 1])
                decrease_score = decrease_count / (len(buffer) - 1)

                # Measure 2: Convergence to final goal (last element close to zero)
                final_goal_closeness = 1 / (1 + abs(buffer[-1]))  # Scales closer values to a higher score
                
                # Combined score: Weighted average of both measures
                policy_score = 0.7 * decrease_score + 0.3 * final_goal_closeness
                task_performance.append(policy_score)

            # Average performance across all policies for the task
            overall_task_score = sum(task_performance) / len(task_performance)
            overallTaskPerformance.append(overall_task_score)
        
        
        # print("Overall Task Performance calculated:", overallTaskPerformance)
        self.overall_task_performance=overallTaskPerformance
        
        
    def update_buffer(self, policy_idx, task_idx):
        """
        Updates the buffer for a specific policy in a specific task.
        
        Args:
        - policy_idx: Index of the policy to update.
        - task_idx: Index of the task to update.
        - new_buffer_values: List of new values to replace the buffer.
        """
        # Construct the task and policy keys
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        # Access the buffer for the specific task and policy
        buffer = self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"]
        set_pairs = self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"]
        
        
        coordinates = self.task_dictionary[task_key]["Coordinates"]
        #visual_goal = tools.denormalize_vector(self.somVisual.get_weights()[coordinates[1][0], coordinates[1][1]], gps_data)
        #print(f"Goal: {coordinates[1][0]}, { coordinates[1][1]}")
        visual_goal = self.somVisual.get_weights()[coordinates[1][0], coordinates[1][1]]
        #visual_goal=np.round(visual_goal,4)
        motor_angles_goal_coord = self.hebbian_table.getConectionsFromSOM1(visual_goal)
            
        if motor_angles_goal_coord is not None:
            rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_goal_coord[0], motor_angles_goal_coord[1]], motor_data)
            realGoal=self.robot.getRealGpsGoal(rotation_angles)
        else:
            print("Goal not reachable, changing task")
            return None
            
        # Calculate predictive error for each coordinate in set_pairs
        for idx, coord in enumerate(set_pairs):
            #visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
            visual_input = self.somVisual.get_weights()[coord[0], coord[1]]
            #visual_input=np.round(visual_input,4)
            #print(f"Visual input= {coord[0]}, {coord[1]}")
            motor_angles_coord = self.hebbian_table.getConectionsFromSOM1(visual_input)
            
            if motor_angles_coord is not None:
                rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                
                # Assume the final goal is the second coordinate of the pair
                predictive_error = self.robot.GetPredError(rotation_angles, realGoal)
                
                # Store predictive error in the buffer
                if idx < len(buffer):  # Ensure we don't go out of bounds
                    buffer[idx] = predictive_error 
            elif idx==0 and motor_angles_coord==None:
                print("Unstartable task (first coordinate in policy not reachable), changing task")
                return None

            # Preserve the first element of buffer and set_pairs
        first_buffer_element = buffer[0]
        first_set_pair = set_pairs[0]
        
        # Sort buffer (excluding the first element) in descending order and reorder set_pairs accordingly
        sorted_pairs = sorted(zip(buffer[1:], set_pairs[1:]), key=lambda x: x[0], reverse=True)
        sorted_buffer, sorted_set_pairs = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Reconstruct the buffer and set_pairs with the first elements intact
        buffer = [first_buffer_element] + list(sorted_buffer)
        set_pairs = [first_set_pair] + list(sorted_set_pairs)
        
        # Update the task dictionary with the sorted buffer and set_pairs
        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"] = buffer
        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"] = set_pairs
        
        #self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"] = buffer
        return 1
        
        


    def print_task_dict(self):
        """
        Print the data from the task dictionary.
        """        
        for task, content in self.task_dictionary.items():
            print(task, content)

    def get_min_distance_to_neighbors(self, node_coord):
        """
        Calculate the minimum Euclidean distance between a node in the SOM and its neighbors.
        
        Args:
        - node_coord: A tuple (x, y) representing the coordinates of the node.
        
        Returns:
        - The minimum distance between the node and its neighbors.
        """
        som_height= self.somVisual.get_weights().shape[0] 
        som_width = self.somVisual.get_weights().shape[1]

        som_weights=self.somVisual.get_weights()
        #som_weights=np.round(som_weights,4)
        x, y = node_coord
        
        # List of relative positions of 8 neighbors
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  
            (0, -1),          (0, 1),   
            (1, -1), (1, 0), (1, 1)     
        ]
        
        min_distance = float('inf')

        node_vector = som_weights[x, y]

        for offset in neighbor_offsets:
            neighbor_x, neighbor_y = x + offset[0], y + offset[1]
            
            # Ensure the neighbor is within the bounds of the SOM grid
            if 0 <= neighbor_x < som_height and 0 <= neighbor_y < som_width:
                neighbor_vector = som_weights[neighbor_x, neighbor_y]
                # Calculate the Euclidean distance to the neighbor
                distance = np.linalg.norm(node_vector - neighbor_vector)

                if distance < min_distance:
                    min_distance = distance
        
        return min_distance
    
    def evaluate_buffer(self,buffer):
        """
        Evaluates a buffer of PE based on three conditions:

        1. The last element's distance to the x-axis (closeness to zero).
        2. The slope of the linear regression over the buffer values, ensuring it's negative.
        3. Whether the buffer is strictly decreasing.

        Args:
            -buffer (list or array): A buffer of values representing the history of predictive errors.

        Returns:
            tuple:
                - distance_to_zero (float): The absolute value of the last buffer element (how close it is to zero).
                - slope (float): The slope of the linear regression for the buffer values over time.
                - is_decreasing (bool): A boolean indicating if the buffer is strictly decreasing.
        """        
        
        time_buffer = np.arange(self.lengthOfBuffers) 
        # Condition 1: Last element of the buffer close to x axis
        distance_to_zero = abs(buffer[-1])
        
        # Condition 2: Slope of the linear regression (must be negative)
        _, slope = self.estimate_coef(time_buffer, np.array(buffer))

        # Condition 3: Verify if the buffer is strictly decreasing 
        is_decreasing = all(buffer[i] >= buffer[i+1] for i in range(len(buffer)-1))
        
        return distance_to_zero, slope, is_decreasing
    
    def evaluate_best_task_buffers(self, best_task_index):
        """
        Evaluates only the buffers of the task with the highest overall performance.
        
        Returns:
            list: A sorted list of tuples, where each tuple contains:
                - buffer_index (int): The index of the buffer (calculated across all tasks).
                - buffer (list or array): The buffer of values representing predictive errors.
                - distance_to_zero (float): The absolute value of the last buffer element.
                - slope (float): The slope of the linear regression for the buffer values.
            
            The returned list is sorted first by distance to zero (ascending) and then by slope (ascending).
        """
        # Identify the index of the task with the best overall performance
        #best_task_index = self.overallTaskPerformance.index(max(self.overallTaskPerformance))
        
        evaluated_buffers = []
        
        # Retrieve the best-performing task from the task dictionary
        best_task = self.task_dictionary[f"Task_{best_task_index}"]
        
        # Evaluate only the buffers in this best-performing task
        for policy_index, (policy, policy_data) in enumerate(best_task["Sets_and_Buffers"].items()):
            buffer = policy_data["Buffer"]
            
            # Calculate the global buffer index
            buffer_index = (best_task_index * 4) + policy_index
            
            # Evaluate the buffer
            distance_to_zero, slope, _ = self.evaluate_buffer(buffer)
            
            # Store the buffer with its evaluation criteria
            evaluated_buffers.append((buffer_index, buffer, distance_to_zero, slope))

        # Sort buffers:
        # First by distance_to_zero
        # Then by slope (most negative)
        evaluated_buffers.sort(key=lambda x: (x[2], x[3]))  # Sort by (distance_to_zero, slope)
        
        return evaluated_buffers


  
    
    def get_worst_task(self):
        """
        Returns the index of the worst task based on the evaluated buffers.

        The method calls 'evaluate_all_buffers' to get a sorted list of buffers, then returns the index of the buffer deemed 
        worst (the last in the sorted list).

        Returns:
            The index of the worst task based on the evaluation criteria.
        """        
        buffer_sort=self.evaluate_all_buffers()
        worst_task_idx=buffer_sort[-1][0]
        return worst_task_idx

           
    # def get_best_goal(self):
    #     """
    #     Returns the index of the best goal based on the evaluated buffers.

    #     The method calls 'evaluate_all_buffers' to get a sorted list of buffers, then returns the index of the buffer deemed 
    #     best (the first in the sorted list).

    #     Returns:
    #         The index of the best goal based on the evaluation criteria.
    #     """        
    #     buffer_sort=self.evaluate_all_buffers()
    #     best_goal_idx=buffer_sort[0][0]
    #     return best_goal_idx
    
    def get_best_policy(self, best_task_idx):
        """
        Returns the index of the best policy based on the evaluated buffers of a given task.

        The method calls 'evaluate_best_task_buffers' to get a sorted list of buffers, then returns the index of the buffer deemed 
        best (the first in the sorted list).
        
        Args:
            -best_task_idx(int): The index of the best overall performance task
        Returns:
            The index of the best goal based on the evaluation criteria.
        """        
        buffer_sort=self.evaluate_best_task_buffers(best_task_idx)
        best_policy_idx=buffer_sort[0][0]
        return best_policy_idx
        
    # def get_random_goal(self):
    #     """
    #     Generates a random goal index within a specified range.

    #     This method returns a random integer between 0 and 39, representing a goal index.

    #     Returns:
    #         A randomly generated goal index.
    #     """        
    #     random_goal_idx=random.randint(0, 39)
    #     return random_goal_idx
    
    def get_random_policy(self, task_idx):
        """
        Generates a random policy index within a specified range.

        This method returns a random integer between 0 and 3, representing a policy index.

        Returns:
            A randomly generated policy index.
        """        
        random_policy_idx=(task_idx * 4) + random.randint(0, 3)
        return random_policy_idx

    def get_neighbors(self, coord):
        """
        Returns the valid neighbors for a given coordinate in the Self-Organizing Map (SOM).

        This method takes a coordinate as input and calculates its neighbors, considering up, down, left, righ and diagonal positions. 

        Args:
            -coord (tuple): A tuple representing the coordinate (row, column) for which to find neighbors.

        Returns:
            -list: A list of valid neighboring coordinates within the SOM.
        """        
        row, col = coord
        som_height = self.somVisual.get_weights().shape[0]  # Number of rows
        som_width = self.somVisual.get_weights().shape[1]   # Number of columns
        neighbors = []
        
        possible_neighbors = [
            (row - 1, col),       # Up
            (row + 1, col),       # Down
            (row, col - 1),       # Left
            (row, col + 1),       # Right
            (row - 1, col - 1),   # Top-left
            (row - 1, col + 1),   # Top-right
            (row + 1, col - 1),   # Bottom-left
            (row + 1, col + 1)    # Bottom-right
        ]
        
        for r, c in possible_neighbors:
            if 0 <= r < som_height and 0 <= c < som_width:
                neighbors.append((r, c))
        
        return neighbors

    def change_policy(self, policy_idx, task_idx, num_coords_change, goal_coord):
        """
        Change a specific number of coordinates in the given policy while keeping the first coordinate.
        
        Args:
        - policy_idx: Index of the policy to change.
        - task_idx: Index of the task to which the policy belongs.
        - num_coords_change: Number of coordinates to change in the policy.
        - goal_coord: The goal coordinate that the agent wants to reach
        """
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        buffer = self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"]
        set_pairs = list(self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"])
        
        print(f"Set pairs: {set_pairs}")
        first_coord = set_pairs[0]
        #first_buffer = buffer[0]
        
        coords_to_consider = set_pairs[1:]  # Skip the first coordinate
        #buffer_to_consider = buffer[1:]
        
        sorted_coords_by_error = sorted(coords_to_consider, key=lambda coord: buffer[set_pairs.index(coord)], reverse=True)
        
        for i in range(min(num_coords_change, len(sorted_coords_by_error))):
            coord_to_change = sorted_coords_by_error[i]
            
            neighbors = self.get_neighbors(coord_to_change)
            #print(f"Neighbors: {neighbors}")

            valid_neighbors = [n for n in neighbors if n not in set_pairs and n!= goal_coord]
            
            if not valid_neighbors:
                print(f"No valid neighbors found for {coord_to_change}. Skipping.")
                continue
            
            #new_coord = random.choice([n for n in valid_neighbors])
            
            
            if random.random() < 0.4:
                # Explore a random neighbor
                new_coord = random.choice(valid_neighbors)
            else:
                # Select the neighbor closest to the goal coordinate
                new_coord = min(valid_neighbors, key=lambda neighbor: np.linalg.norm(np.array(neighbor) - np.array(goal_coord)))

            # new_coord = min(valid_neighbors, key=lambda neighbor: np.linalg.norm(np.array(neighbor) - np.array(goal_coord)))
            print(f"Changed coord: {coord_to_change} for: {new_coord}")
            # Replace the old coordinate with the new one
            set_pairs[set_pairs.index(coord_to_change)] = new_coord
        
        #combined = sorted(zip(buffer_to_consider, coords_to_consider), key=lambda x: x[0], reverse=True)
        #sorted_buffer, sorted_coords = zip(*combined)
        
        #buffer = [first_buffer] + list(sorted_buffer)
        
        #set_pairs = [first_coord] + list(sorted_coords)
        
        set_pairs[0] = first_coord  
        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"] = set_pairs
        #self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"] = buffer
        
        print(f"New set pairs: {set_pairs}")
        #print( self.task_dictionary[task_key]["Sets_and_Buffers"])
        print(f"Updated policy {policy_idx} for task {task_idx}, keeping the first coordinate unchanged.")

        #self.print_task_dict()
        
    #linear regression
    def estimate_coef(self,x, y):
        """
        Estimates the coefficients of a linear regression model given two vectors.

        Args:
            -x (numpy.ndarray): A 1D array representing the independent variable values.
            -y (numpy.ndarray): A 1D array representing the dependent variable values.

        Returns:
            tuple: A tuple containing the intercept (b_0) and slope (b_1) of the regression line.
        """        
        n = np.size(x)
        #mean of x and y vector
        m_x = np.mean(x)
        m_y = np.mean(y)
        
        SS_xy = np.sum(y*x) - n*m_y*m_x
        SS_xx = np.sum(x*x) - n*m_x*m_x
        
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
        return (b_0, b_1)
    
    def get_goal_from_task(self,task_idx):
        """
        Returns the goal coordinate from a specified task.

        Args:
            -task_idx (int): The index of the task to get the goal coordinate from.

        Returns:
            -tuple: A tuple representing the goal coordinate (row, column) of the specified task.
        """        
        task_key = f"Task_{task_idx}"
        
        goal = self.task_dictionary[task_key]["Coordinates"][1]
        
        return goal

    def get_buffer_from_task_policy(self,task_idx, policy_idx):
        """
        Returns the buffer associated with a specific policy of a specified task.

        Args:
            -task_idx (int): The index of the task from which to retrieve the buffer.
            -policy_idx (int): The index of the policy associated with the task.

        Returns:
            -list: The buffer corresponding to the specified task and policy.
        """        
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        buffer = self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"]
        
        return buffer

    def get_slopes(self):
        """
        Calculates the slopes of linear regression for all buffers in the task dictionary.

        Returns:
            numpy.ndarray: An array of slopes calculated for each buffer.
        """        
        time_buffer = np.arange(self.lengthOfBuffers)
        # List to store slopes of each buffer
        slopes = []
        
        # Calculate slopes for each buffer in data_dict
        for task, values in self.task_dictionary.items():
            for policy, policy_data in values["Sets_and_Buffers"].items():
                buffer = policy_data["Buffer"]
                # Perform linear regression on the buffer
                lin_reg = self.estimate_coef(time_buffer, np.array(buffer))
                slope = lin_reg[1]
                
                # Store the slope in the dictionary (if desired)
                policy_data["Slope"] = slope
                
                # Also store it in the slopes list
                slopes.append(slope)

        slopes = np.array(slopes)
        return slopes	
    
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
