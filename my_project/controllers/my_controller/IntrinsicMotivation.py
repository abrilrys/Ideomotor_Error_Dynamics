import numpy as np
import random
import tools
import csv


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
        self.lengthOfBuffers=11
        self.somVisual=somVisual
        self.somAngles=somAngles
        self.hebbian_table=hebbianTable
        self.task_dictionary=self.initialize_task_dictionary()
        self.slopes=self.get_slopes()
        #self.print_task_dict()
        #print(self.buffers)
        
    def initialize_task_dictionary(self):
        """
         Initializes a dictionary containing task-related data for the robot's SOM-based learning process.

            The method performs the following steps:
            
            1. Randomly selects 10 unique pairs of coordinates to represent tasks.
            2. For each task, initializes 4 policies, where each policy contains:
            - A set of 10 coordinates (the first being one of the task's pair coordinates).
            - A buffer of length 11, initialized with increasing values.
            3. For each task and policy, calculates predictive error.
            5. Updates the policy's buffer with the calculated predictive error for each coordinate.

            Returns:
            A dictionary containing task data, where each task has its selected coordinate pair, policies and associated buffers.
        """    
        
        # Define the dimensions of the SOM
        som_height = self.somVisual.get_weights().shape[0]  # Number of rows
        som_width = self.somVisual.get_weights().shape[1]   # Number of columns
        
        # Generate all possible coordinates
        all_coordinates = [(row, col) for row in range(som_height) for col in range(som_width)]
        
        # Shuffle the list of coordinates
        random.shuffle(all_coordinates)
        
        # Select 10 unique pairs of coordinates without repetition
        selected_pairs = []
        selected_pairs_count = 0
        
        numberOfTasks = 10
        
        while selected_pairs_count < numberOfTasks:
            # Select two random coordinates
            coord1 = random.choice(all_coordinates)
            coord2 = random.choice(all_coordinates)
            
            # Ensure the pair is unique and not repeated
            if coord1 != coord2 and (coord1, coord2) not in selected_pairs and (coord2, coord1) not in selected_pairs:
                selected_pairs.append((coord1, coord2))
                selected_pairs_count += 1
        
        # Create a dictionary to store the data
        data_dict = {}
        
        numberOfPolicies = 4
        lengthOfPolicies = 10
        lengthOfBuffers = self.lengthOfBuffers
        
        # Populate the dictionary with selected pairs, associated sets, and buffers
        for task_index, pair in enumerate(selected_pairs, start=0):
            sets_and_buffers = {}
            for policy_index in range(0, numberOfPolicies):
                set_pairs = []
                # Add random coordinates not equal to the pair coordinates
                while len(set_pairs) < lengthOfPolicies:
                    random_coord = random.choice(all_coordinates)
                    if random_coord != pair[0] and random_coord != pair[1]:
                        set_pairs.append(random_coord)
                        
                
                set_pairs.insert(0, pair[0])
                #print(set_pairs)
                # Initialize buffer with increasing values
                buffer = []
                valor = 10
                while len(buffer) < self.lengthOfBuffers:
                    buffer.append(valor)
                    valor += 10
                # Store the policy with its index
                sets_and_buffers[f"Policy_{policy_index}"] = {
                    "Set": tuple(set_pairs),
                    "Buffer": buffer
                }
                #print(sets_and_buffers)
            # Store the task with its index
            data_dict[f"Task_{task_index}"] = {
                "Coordinates": pair,
                "Sets_and_Buffers": sets_and_buffers
            }
        
        # Calculate predictive error and update the buffer
        for task, values in data_dict.items():
            pair = values["Coordinates"]
            visual_goal = tools.denormalize_vector(self.somVisual.get_weights()[pair[1][0], pair[1][1]], gps_data)
            
            for policy, policy_data in values["Sets_and_Buffers"].items():
                set_pairs = policy_data["Set"]
                buffer = policy_data["Buffer"]
                
                # Calculate predictive error for each coordinate in set_pairs
                for idx, coord in enumerate(set_pairs):
                    #visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
                    visual_input=self.somVisual.get_weights()[coord[0], coord[1]]
                    
                    motor_angles_coord = self.hebbian_table.getConectionsFromSOM1(visual_input)
                    
                    if motor_angles_coord is not None:
                        rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                        
                        # Assume the final goal is the second coordinate of the pair
                        predictive_error = self.robot.executeMovement(rotation_angles, visual_goal)
                        
                        # Store predictive error in the buffer
                        if idx < len(buffer):  # Ensure we don't go out of bounds
                            buffer[idx] = predictive_error  
        
        # List to store the buffers
        feature_vectors = []
        
        # Loop through data_dict to extract only the buffers
        for task, values in data_dict.items():
            for policy, policy_data in values["Sets_and_Buffers"].items():
                buffer = policy_data["Buffer"]
                feature_vectors.append(buffer)  # Append only the buffer
                
        self.buffers=feature_vectors
        
        # Convert the list of buffers to a NumPy array
        tasks_array = np.array(feature_vectors)
        # Save the NumPy array into a CSV file
        np.savetxt('tasks_train_dataset.csv', tasks_array, delimiter=',', fmt='%.6f')
        
        print("Buffers saved into 'tasks_train_dataset.csv'")
        return data_dict
    
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
        visual_goal = tools.denormalize_vector(self.somVisual.get_weights()[coordinates[1][0], coordinates[1][1]], gps_data)

        # Calculate predictive error for each coordinate in set_pairs
        for idx, coord in enumerate(set_pairs):
            #visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
            visual_input = self.somVisual.get_weights()[coord[0], coord[1]]
            
            motor_angles_coord = self.hebbian_table.getConectionsFromSOM1(visual_input)
            
            if motor_angles_coord is not None:
                rotation_angles = tools.denormalize_vector(self.somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                
                # Assume the final goal is the second coordinate of the pair
                predictive_error = self.robot.executeMovement(rotation_angles, visual_goal)
                
                # Store predictive error in the buffer
                if idx < len(buffer):  # Ensure we don't go out of bounds
                    buffer[idx] = predictive_error 

        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"] = buffer
        
        


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
    
    def evaluate_all_buffers(self):
        """
        Evaluates all buffers in the task dictionary based on their distance to zero and the slope of their linear regression.

            The method iterates through each task and policy in the task dictionary, evaluates the corresponding buffer using 
            the evaluate_buffer method, and stores the evaluation results.

            Returns:
                list: A sorted list of tuples, where each tuple contains:
                    - buffer_index (int): The index of the buffer.
                    - buffer (list or array): The buffer of values representing predictive errors.
                    - distance_to_zero (float): The absolute value of the last buffer element (how close it is to zero).
                    - slope (float): The slope of the linear regression for the buffer values over time.
            
            The returned list is sorted first by distance to zero (ascending) and then by slope (ascending).
        """        
        evaluated_buffers = []
        
        buffer_index = 0
        # Loop through all buffers in the data_dict
        for task, values in self.task_dictionary.items():
            for policy, policy_data in values["Sets_and_Buffers"].items():
                buffer = policy_data["Buffer"]
                
                # Evaluate the buffer
                distance_to_zero, slope,_ = self.evaluate_buffer(buffer)
                
                # Store the buffer with its evaluation criteria
                evaluated_buffers.append((buffer_index, buffer, distance_to_zero, slope))
                
                buffer_index += 1
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
           
    def get_best_goal(self):
        """
        Returns the index of the best goal based on the evaluated buffers.

        The method calls 'evaluate_all_buffers' to get a sorted list of buffers, then returns the index of the buffer deemed 
        best (the first in the sorted list).

        Returns:
            The index of the best goal based on the evaluation criteria.
        """        
        buffer_sort=self.evaluate_all_buffers()
        best_goal_idx=buffer_sort[0][0]
        return best_goal_idx
        
    def get_random_goal(self):
        """
        Generates a random goal index within a specified range.

        This method returns a random integer between 0 and 39, representing a goal index.

        Returns:
            A randomly generated goal index.
        """        
        random_goal_idx=random.randint(0, 39)
        return random_goal_idx

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
        
        coords_to_consider = set_pairs[1:]  # Skip the first coordinate
        
        sorted_coords_by_error = sorted(coords_to_consider, key=lambda coord: buffer[set_pairs.index(coord)], reverse=True)
        
        for i in range(min(num_coords_change, len(sorted_coords_by_error))):
            coord_to_change = sorted_coords_by_error[i]
            
            neighbors = self.get_neighbors(coord_to_change)
            #print(f"Neighbors: {neighbors}")

            valid_neighbors = [n for n in neighbors if n not in set_pairs and n!= goal_coord]
            
            if not valid_neighbors:
                print(f"No valid neighbors found for {coord_to_change}. Skipping.")
                continue
            #new_coord = random.choice([n for n in neighbors if n != coord_to_change])
            
            # Select the neighbor closest to the goal coordinate
            new_coord = min(neighbors, key=lambda neighbor: np.linalg.norm(np.array(neighbor) - np.array(goal_coord)))
            print(f"Changed coord: {coord_to_change} for: {new_coord}")
            # Replace the old coordinate with the new one
            set_pairs[set_pairs.index(coord_to_change)] = new_coord
        
        set_pairs[0] = first_coord  
        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"] = set_pairs
        print(f"New set pairs: {set_pairs}")

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