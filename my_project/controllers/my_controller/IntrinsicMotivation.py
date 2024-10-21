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
                    visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
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
        
        Parameters:
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
            visual_input = tools.denormalize_vector(self.somVisual.get_weights()[coord[0], coord[1]], gps_data)
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
        for task, content in self.task_dictionary.items():
            print(task, content)

    def get_min_distance_to_neighbors(self, node_coord):
        """
        Calculate the minimum Euclidean distance between a node in the SOM and its neighbors.
        
        Parameters:
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
        time_buffer = np.arange(self.lengthOfBuffers) 
        # Condition 1: Last element of the buffer close to x axis
        distance_to_zero = abs(buffer[-1])
        
        # Condition 2: Slope of the linear regression (most negative)
        _, slope = self.estimate_coef(time_buffer, np.array(buffer))

        # Condition 3: Verify if the buffer is strictly decreasing 
        is_decreasing = all(buffer[i] >= buffer[i+1] for i in range(len(buffer)-1))
        
        return distance_to_zero, slope, is_decreasing
    
    def evaluate_all_buffers(self):
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
           
    def get_best_goal(self):
        buffer_sort=self.evaluate_all_buffers()
        best_goal_idx=buffer_sort[0][0]
        return best_goal_idx
        
    def get_random_goal(self):
        random_goal_idx=random.randint(0, 39)
        return random_goal_idx

    def get_neighbors(self, coord):
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

    def change_policy(self, policy_idx, task_idx, num_coords_change):
        """
        Change a specific number of coordinates in the given policy while keeping the first coordinate.
        
        Parameters:
        - policy_idx: Index of the policy to change.
        - task_idx: Index of the task to which the policy belongs.
        - num_coords_change: Number of coordinates to change in the policy.
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

            new_coord = random.choice([n for n in neighbors if n != coord_to_change])
            
            set_pairs[set_pairs.index(coord_to_change)] = new_coord
        
        set_pairs[0] = first_coord  
        self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Set"] = set_pairs
        print(f"New set pairs: {set_pairs}")

        print(f"Updated policy {policy_idx} for task {task_idx}, keeping the first coordinate unchanged.")

        #self.print_task_dict()
        
    #linear regression
    def estimate_coef(self,x, y):
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
        task_key = f"Task_{task_idx}"
        
        goal = self.task_dictionary[task_key]["Coordinates"][1]
        
        return goal

    def get_buffer_from_task_policy(self,task_idx, policy_idx):
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        buffer = self.task_dictionary[task_key]["Sets_and_Buffers"][policy_key]["Buffer"]
        
        return buffer

    def get_slopes(self):
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