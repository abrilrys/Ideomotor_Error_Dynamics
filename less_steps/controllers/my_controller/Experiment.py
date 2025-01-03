import os
import IntrinsicMotivation as intrinsic
import time
import numpy as np
import json
import tools
import random
import pickle
import HebbianTable as hebbian
import csv
from collections import deque



class Experiment:
    """
    Implements the experiment.
    """   
    
    def __init__(self, eps, duration, robot):
        """
        Initialize the parameters of the experiment
        Args:
            -eps (float): The epsilon value for the epsilon-greedy algorithm, which determines 
                        the probability of selecting a random task versus the best-rated task.
            -duration (float): The total duration of the experiment in seconds.
            -robot (object): An instance of the robot that the agent will control or interact with.
        """        
        self.robot=robot
        self.intrinsic_motivation = intrinsic.IntrinsicMotivation(somVisual, somAngles, hebbian_table, robot)
        self.eps=eps
        self.duration=duration
        self.max_len_buffer_behaviour=25
        self.buffer_agent=deque(maxlen=self.max_len_buffer_behaviour)
        self.current_goal_idx = -1
        

    def get_task_policy_from_index(self, index):
        """
        Calculate the task and policy index from a given index.

        Args:
            -index (int): The index representing a specific task-policy combination.

        Returns:
            -tuple: A tuple containing two integers:
                - task_idx (int): The calculated index of the task.
                - policy_idx (int): The calculated index of the policy within the task.
        """        
        # Calculate the task and policy index
        task_idx = index // 4  
        policy_idx = index % 4  
        
        return task_idx, policy_idx
    
    def run_exp(self):
        """
        Execute the learning experiment over a specified duration.
        """        
        self.clear_previous_file("task_dictionary.txt")
        self.clear_previous_file("learnt_policies.json")
        
        self.intrinsic_motivation.initDict()
        start_time = time.time()
        
        it=0
        treshold_bad_behaviour=5
        counter_bad_behaviour=0
        min_iterations_per_task=10
        counter_it_in_task=0
        
        
        buffer_time= deque(maxlen=self.max_len_buffer_behaviour)
        saved_policies_count = 0  # Track policies with AddCount = 3
        
        #get the best rated task on overall performance (explote)
        task_index = self.intrinsic_motivation.overall_task_performance.index(max(self.intrinsic_motivation.overall_task_performance)) 
                        
        #while time.time() - start_time < self.duration:
        while saved_policies_count < 4:
            
            print(f"######################################################\n Iteration {it}\n")
            #print(f"Previous task {task_index}\n")
            
            if(counter_it_in_task>min_iterations_per_task):
                # print(f"EVALUATING TASK CHANGE, COUNTER BAD BEHAVIOUR: {counter_bad_behaviour}")
                counter_it_in_task=0
                if(counter_bad_behaviour>treshold_bad_behaviour):
                    counter_bad_behaviour=0
                    prob=np.random.random() 
                    if prob < self.eps: 
                        # move to another random task (explore)
                        task_index = random.randint(0, (self.intrinsic_motivation.numberOfTasks-1))
                        # print(f"Changed to a random task {task_index}\n") 
                    else:
                        #get the best rated task on overall performance (explote)
                        task_index = self.intrinsic_motivation.overall_task_performance.index(max(self.intrinsic_motivation.overall_task_performance)) 
                        # print(f"Changed to the best rated task {task_index}\n")  
                    
                else:
                    #continue learning the same task
                    task_index=task_index
                    # print(f"Continuing with task {task_index} because of good performance\n")
                    
            
            # get the current task using the intrinsic motivation strategy and e-greedy algorithm
            p = np.random.random() 
            if p < self.eps: 
                #select random task
                # self.current_goal_idx=self.intrinsic_motivation.get_random_goal()
                self.current_goal_idx=self.intrinsic_motivation.get_random_policy(task_index)
                
                task_idx, policy_idx=self.get_task_policy_from_index(self.current_goal_idx)
                #get previous dynamic of said taks and policy
                prev_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))
            else: 
                #select best rated task
                #self.current_goal_idx=self.intrinsic_motivation.get_best_goal()
                self.current_goal_idx=self.intrinsic_motivation.get_best_policy(task_index)
                
                task_idx, policy_idx=self.get_task_policy_from_index(self.current_goal_idx)
                #get previous dynamic of said taks and policy
                prev_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))

            # print(f"EXECUTING TASK {task_idx}, POLICY {policy_idx}")

            # print(f"Previous dynamic: {prev_din}")
            
            goal_task=self.intrinsic_motivation.get_goal_from_task(task_idx) 
            #print(f"Goal coordinate: {goal_task}")
            #modify the policy
            self.intrinsic_motivation.change_policy(policy_idx, task_idx,1 , goal_task)
            
            
            #update the PE
            x=self.intrinsic_motivation.update_buffer(policy_idx, task_idx)
            if(x==None):
                self.remove_task(task_idx, self.intrinsic_motivation.task_dictionary, "learnt_policies.json")
                #self.learnt_task_remove(task_idx, self.intrinsic_motivation.task_dictionary, "learnt_policies.json")
                
            else:  
                #get current dynamic of current goal
                new_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))
                # print(f"New dynamic: {new_din}")
                
                #get PE of dynamic
                mse = np.mean((np.array(prev_din) - np.array(new_din)) ** 2)
                # print(f"Mean Squared Error: {mse}")
                
                self.buffer_agent.append(mse)
                
                buffer_time.append(it)
                # print(f"buffer time: {buffer_time}")
                
                b0, b1=self.intrinsic_motivation.estimate_coef(np.array(buffer_time), np.array(self.buffer_agent))
                # print(f"Agent behaviour: b0= {b0}, b1= {b1}")
                
                #check if the task is learnt
                evaluation_buffer=self.intrinsic_motivation.evaluate_buffer(new_din)
                min_distance_to_neighbors=self.intrinsic_motivation.get_min_distance_to_neighbors(tuple(goal_task))
                #print("min_distance_to_neighbors > ", min_distance_to_neighbors)
                if evaluation_buffer[0]<min_distance_to_neighbors and evaluation_buffer[1]<0 and evaluation_buffer[2]:
                    print("task learnt")
                    self.save_policy_to_json("learnt_policies.json",task_idx, policy_idx, self.intrinsic_motivation.task_dictionary)
                    self.learnt_task_remove(task_idx,self.intrinsic_motivation.task_dictionary,"learnt_policies.json" )
                    
                    with open("learnt_policies.json", 'r') as file:
                        policies = json.load(file)
                        saved_policies_count = sum(1 for p in policies if p["AddCount"] == 3)
                    
                    if saved_policies_count >= 4:
                        print(f"Experiment completed")
                        break  # Stop the loop

                if (b1 > 0):
                    counter_bad_behaviour= counter_bad_behaviour+1
                    # print(f"Counter bad behaviour: {counter_bad_behaviour}")
                
                self.intrinsic_motivation.updateTaskPerformance()
                # print(f"Overall task performance: {self.intrinsic_motivation.overall_task_performance}")
                it=it+1
                counter_it_in_task=counter_it_in_task+1
                    
            #self.save_task_dictionary_to_txt(self.intrinsic_motivation.task_dictionary, "task_dictionary.txt", it)      
    
    # def save_policy_to_json(self,file_name, task_idx, policy_idx, task_dictionary):
    #     """
    #     Save a specific policy's coordinates and set of coordinates to a JSON file.
        
    #     Args:
    #     - file_name: Name of the JSON file to save to.
    #     - task_idx: The index of the task.
    #     - policy_idx: The index of the policy within the task.
    #     - task_dictionary: The task dictionary containing the task and policy data.
    #     """
    #     task_key = f"Task_{task_idx}"
    #     policy_key = f"Policy_{policy_idx}"
        
    #     policy_data = task_dictionary[task_key]["Sets_and_Buffers"][policy_key]

    #     coordinates = task_dictionary[task_key]["Coordinates"]
        
    #     # Structure the data
    #     policy_to_save = {
    #         "Coordinates": coordinates,
    #         "SetPairs": list(policy_data["Set"]),  # Convert set to a list for JSON compatibility
    #         "Buffer":list(policy_data["Buffer"])
    #     }
        
    #     if os.path.exists(file_name):
    #         with open(file_name, 'r') as json_file:
    #             all_policies = json.load(json_file)

    #     else:
    #         all_policies = []  
        
    #     # Add the new policy to the list of all policies
    #     all_policies.append(policy_to_save)
        
    #     with open(file_name, 'w') as json_file:
    #         json.dump(all_policies, json_file, indent=4)
        
    #     print(f"Policy {policy_data['Set']} from task {coordinates} saved to {file_name}.")
    
    def save_policy_to_json(self, file_name, task_idx, policy_idx, task_dictionary):
        """
        Save or update a policy in a JSON file.
        
        If a policy with the same start coordinate as the new policy's end coordinate exists, update it.
        Otherwise, add the new policy as a standalone entry.

        Args:
        - file_name: Name of the JSON file to save to.
        - task_idx: The index of the task.
        - policy_idx: The index of the policy within the task.
        - task_dictionary: The task dictionary containing the task and policy data.
        """
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        policy_data = task_dictionary[task_key]["Sets_and_Buffers"][policy_key]
        coordinates = task_dictionary[task_key]["Coordinates"]
        
        new_policy_data = {
            "Coordinates": coordinates,
            "SetPairs": list(policy_data["Set"]),
            "Buffer": list(policy_data["Buffer"]),
            "AddCount": 0  
        }

        if os.path.exists(file_name):
            with open(file_name, 'r') as json_file:
                all_policies = json.load(json_file)
        else:
            all_policies = []

        existing_policy = None
        for policy in all_policies:
            print(f"Policy coordinates in json: {policy['Coordinates'][1]} and policy coordinates new: {coordinates[0]}")
            if list(policy["Coordinates"][1]) == list(coordinates[0]):  
                existing_policy = policy
                break
        
        print(f"Existing policy: {existing_policy}")
        if(existing_policy):
            print(f"Add count: {existing_policy['AddCount']}")
        if existing_policy and existing_policy["AddCount"] < 3:
            # Update the existing policy
            existing_policy["SetPairs"].extend(new_policy_data["SetPairs"])  
            existing_policy["Coordinates"][1] = coordinates[1] 
            existing_policy["AddCount"] += 1  
            existing_policy["Buffer"].extend(new_policy_data["Buffer"])
            
            print(f"Updated existing policy with new data. AddCount: {existing_policy['AddCount']}")
        else:
            # Add the new policy as a standalone entry
            all_policies.append(new_policy_data)
            print(f"Saved new policy with start coordinate {coordinates[0]} to {file_name}.")

        # Save the updated policies back to the file
        with open(file_name, 'w') as json_file:
            json.dump(all_policies, json_file, indent=4)

        if existing_policy and existing_policy["AddCount"] >= 3:
            print(f"Policy has reached the maximum additions (3). No further updates will be made to this policy.")
            

    def load_all_policies_from_json(self,file_name):
        """
        Load all policies from a JSON file.
        
        Args:
        - file_name: The name of the JSON file to load from.
        
        Returns:
        - all_policies: A list of dictionaries, each representing a saved policy.
        """
        with open(file_name, 'r') as json_file:
            all_policies = json.load(json_file)
        
        print(f"{len(all_policies)} policies loaded from {file_name}.")
        return all_policies

    def execute_loaded_policies(self, file_name):
        """
        Load and execute each policy from a JSON file.
        
        Args:
            - file_name: The name of the JSON file to load from.
        """
        all_policies = self.load_all_policies_from_json(file_name)
        all_final_points = [] #stores all trajectories for each task
        
        for policy in all_policies:
            coordinates = policy["Coordinates"]
            set_pairs = policy["SetPairs"]

            set_pairs = list(set_pairs)[1:]  # Skip the first element
            
            merged_coordinates = [coordinates[0]] + set_pairs + [coordinates[1]]
            
            print("Executing Task Policy")
            print("Coordinates:", coordinates)
            print("Set Pairs:", set_pairs)

            print(f"Trajectory: {merged_coordinates} ")
            
            final_points=[]
            for idx, coord in enumerate(merged_coordinates):
                     #visual_input = tools.denormalize_vector(somVisual.get_weights()[coord[0], coord[1]], gps_data)
                     visual_input =somVisual.get_weights()[coord[0], coord[1]]
                     #visual_input=np.round(visual_input,4)
                     
                     motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
                     
                     print(f"Point {idx}: {tools.denormalize_vector(visual_input, gps_data)}")
                     if motor_angles_coord is not None:
                         rotation_angles = tools.denormalize_vector(somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                         
                         self.robot.MoveArm(rotation_angles)
                         real_coords=self.robot.getRelativeCoords()
                         final_points.append(real_coords)
                         print(real_coords)
                     else:
                         print("Point not executable")
            
            all_final_points.append(final_points) 
            print(f"Real points{final_points}")
            
        # Write all final points to a JSON file
        json_file = "real_final_coords.json"
        with open(json_file, 'w') as file:
            json.dump(all_final_points, file)

        print(f"All task trajectories saved to {json_file}")
        
    def execute_policy_by_index(self, file_name, policy_index):
        """
        Execute a specific policy by its index from a JSON file.
        
        Args:
            - file_name: The name of the JSON file to load from.
            - policy_index: The index of the policy to execute.
        """
        all_policies = self.load_all_policies_from_json(file_name)
        
        if policy_index < 0 or policy_index >= len(all_policies):
            print("Error: Índice fuera de rango.")
            return

        policy = all_policies[policy_index]
        coordinates = policy["Coordinates"]
        set_pairs = policy["SetPairs"]
        
        set_pairs = list(set_pairs)[1:]  
        
        merged_coordinates = [coordinates[0]] + set_pairs + [coordinates[1]]
        
        print("Executing Task Policy")
        print("Coordinates:", coordinates)
        print("Set Pairs:", set_pairs)
        print(f"Trajectory: {merged_coordinates} ")
        
        for idx, coord in enumerate(merged_coordinates):
            visual_input = somVisual.get_weights()[coord[0], coord[1]]
            #visual_input = np.round(visual_input, 4)
            
            motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
            
            print(f"Point {idx}: {tools.denormalize_vector(visual_input, gps_data)}")
            if motor_angles_coord is not None:
                rotation_angles = tools.denormalize_vector(
                    somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data
                )
                
                self.robot.MoveArm(rotation_angles)
                real_coords = self.robot.getRelativeCoords()
                print(real_coords)
        
    def learnt_task_remove(self, task_idx, task_dictionary, json_file):
        
        # Remove the task from the task_dictionary
        task_key = f"Task_{task_idx}"
        
        coordinates = task_dictionary[task_key]["Coordinates"]
        goal_coordinate=coordinates[1]
        start_coordinate=coordinates[0]
        print(f"Goal coordinate: {goal_coordinate}")
        print(f"Start coordinate: {start_coordinate}")
        print(f"Removing task:({coordinates}) ")
        if task_key in task_dictionary:
            del task_dictionary[task_key]
            print(f"Task {task_idx} removed from the dictionary.")
        else:
            print(f"Task {task_idx} not found in the dictionary.")
            return
        
        # Load all existing coordinates from the JSON file to avoid duplicates
        existing_coordinates = set()
        # if os.path.exists(json_file):
        #     with open(json_file, 'r') as file:
        #         saved_policies = json.load(file)
        #         for policy in saved_policies:
        #             for coord in policy["SetPairs"]:
        #                 existing_coordinates.add(tuple(coord))

        # Also collect all coordinates from the task dictionary to avoid duplicates
        for task, values in task_dictionary.items():
            # existing_coordinates.add(tuple(values["Coordinates"][0]))
            existing_coordinates.add(tuple(values["Coordinates"][1]))
            for policy, policy_data in values["Sets_and_Buffers"].items():
                for coord in policy_data["Set"]:
                    existing_coordinates.add(tuple(coord))
                    
        existing_coordinates.add(start_coordinate)

        # Generate new unique coordinates for the new task
        new_task_coordinates = self.generate_unique_coordinate(existing_coordinates, goal_coordinate)
       
        # Add the new task to the dictionary
        self.add_new_task_to_dictionary(task_dictionary, new_task_coordinates, task_idx)
        
            
    def remove_task(self, task_idx, task_dictionary, json_file):
        """
        Remove a  task from the task_dictionary and replace it with a new task.
        The new task must have a unique pair of coordinates not present in the dictionary or JSON file.
        
        Args:
        - task_idx: The index of the learned task to be removed.
        - task_dictionary: The dictionary holding the tasks and their policies.
        - json_file: The file where the tasks are saved to ensure unique coordinates.
        """
        # Remove the task from the task_dictionary
        task_key = f"Task_{task_idx}"
        if task_key in task_dictionary:
            del task_dictionary[task_key]
            print(f"Task {task_idx} Updated from the dictionary.")
        else:
            print(f"Task {task_idx} not found in the dictionary.")
            return

        # Load all existing coordinates from the JSON file to avoid duplicates
        existing_coordinates = set()
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                saved_policies = json.load(file)
                for policy in saved_policies:
                    existing_coordinates.add(tuple(policy["Coordinates"][0]))  # Add both task coordinates
                    existing_coordinates.add(tuple(policy["Coordinates"][1]))
                    for coord in policy["SetPairs"]:
                        existing_coordinates.add(tuple(coord))

        # Also collect all coordinates from the task dictionary to avoid duplicates
        for task, values in task_dictionary.items():
            existing_coordinates.add(tuple(values["Coordinates"][0]))
            existing_coordinates.add(tuple(values["Coordinates"][1]))
            for policy, policy_data in values["Sets_and_Buffers"].items():
                for coord in policy_data["Set"]:
                    existing_coordinates.add(tuple(coord))

        # Generate new unique coordinates for the new task
        new_task_coordinates = self.generate_unique_coordinates(existing_coordinates)

        # Add the new task to the dictionary
        self.add_new_task_to_dictionary(task_dictionary, new_task_coordinates, task_idx)

    def generate_unique_coordinate(self,existing_coordinates, start_coordinate):
        """
        Generate a unique pair of coordinates not present in the existing_coordinates set.
        
        Args:
        - existing_coordinates: A set containing coordinates that must be avoided.
        
        Returns:
        - A new unique pair of coordinates.
        """
        som_height = somVisual.get_weights().shape[0]  
        som_width = somVisual.get_weights().shape[1]

        while True:
           
            coord = (random.randint(0, som_height - 1), random.randint(0, som_width - 1))
            visual_goal = somVisual.get_weights()[coord[0], coord[1]]
            motor_angles_goal_coord = hebbian_table.getConectionsFromSOM1(visual_goal)
            if  motor_angles_goal_coord is not None and not coord in existing_coordinates and coord!= start_coordinate:
                return [start_coordinate,coord]
            
    def generate_unique_coordinates(self,existing_coordinates):
        """
        Generate a unique pair of coordinates not present in the existing_coordinates set.
        
        Args:
        - existing_coordinates: A set containing coordinates that must be avoided.
        
        Returns:
        - A new unique pair of coordinates.
        """
        som_height = somVisual.get_weights().shape[0]  
        som_width = somVisual.get_weights().shape[1]

        while True:
            coord1 = (random.randint(0, som_height - 1), random.randint(0, som_width - 1))
            coord2 = (random.randint(0, som_height - 1), random.randint(0, som_width - 1))

            if coord1 != coord2 and not coord1 in existing_coordinates and not coord2 in existing_coordinates:
                return [coord1, coord2]

    # def add_new_task_to_dictionary(self,task_dictionary, coordinates, task_idx):
    #     """
    #     Add a new task with a set of policies to the task_dictionary.
        
    #     Args:
    #     - task_dictionary: The dictionary to which the new task will be added.
    #     - coordinates: The new pair of task coordinates.
    #     - task_idx: The task index of the new task.
    #     """
    #     task_key = f"Task_{task_idx}"

    #     # Initialize the new task with its coordinates
    #     numberOfPolicies = 4
    #     lengthOfPolicies = self.intrinsic_motivation.lengthOfBuffers-1
    #     lengthOfBuffers = self.intrinsic_motivation.lengthOfBuffers
    #     new_task_data = {
    #         "Coordinates": coordinates,
    #         "Sets_and_Buffers": {}
    #     }

    #     # Add 4 policies with random set_pairs and buffers
    #     for policy_idx in range(0, numberOfPolicies):
    #         set_pairs = set()

    #         # Add random coordinates not equal to the task coordinates
    #         while len(set_pairs) < lengthOfPolicies:
    #             random_coord = (random.randint(0, somVisual.get_weights().shape[0] - 1), random.randint(0, somVisual.get_weights().shape[1] - 1))
    #             if random_coord != coordinates[0] and random_coord != coordinates[1]:
    #                 set_pairs.add(random_coord)

    #         set_pairs = list(set_pairs)
    #         set_pairs.insert(0, coordinates[0])

    #         # Initialize buffer with increasing values
    #         buffer = []
    #         valor = 10
    #         while len(buffer) < lengthOfBuffers:
    #             buffer.append(valor)
    #             valor += 10

    #         policy_key = f"Policy_{policy_idx}"

    #         new_task_data["Sets_and_Buffers"][policy_key] = {
    #             "Set": set_pairs,
    #             "Buffer": buffer
    #         }

    #     # Add the new task to the dictionary
    #     task_dictionary[task_key] = new_task_data
    #     print(f"New Task {task_idx} added to the dictionary with coordinates: {coordinates}")

    def add_new_task_to_dictionary(self,task_dictionary, coordinates, task_idx):
        """
        Add a new task with a set of policies to the task_dictionary.
        
        Args:
        - task_dictionary: The dictionary to which the new task will be added.
        - coordinates: The new pair of task coordinates.
        - task_idx: The task index of the new task.
        """
        task_key = f"Task_{task_idx}"
        print(f"Adding new task to dictionary with start ({coordinates[0]}) and goal ({coordinates[1]})")
        # Initialize the new task with its coordinates
        numberOfPolicies = 4
        lengthOfPolicies = self.intrinsic_motivation.lengthOfBuffers-1
        lengthOfBuffers = self.intrinsic_motivation.lengthOfBuffers
        executable_map = self.intrinsic_motivation.generate_executable_map(somVisual.get_weights().shape[0], somVisual.get_weights().shape[1])
        start= coordinates[0]
        end= coordinates[1]
        all_coordinates = [(row, col) for row in range(somVisual.get_weights().shape[0]) for col in range(somVisual.get_weights().shape[1])]
        
        base_path = self.intrinsic_motivation.astar_with_length(executable_map, coordinates[0], coordinates[1], lengthOfPolicies+2)
        if base_path and len(base_path) == lengthOfPolicies + 2:  # A* incluye inicio y fin
            base_path = base_path[1:-1]  # Excluir inicio y fin
        elif base_path==None:#No path found so create a task with new coordinates
            while True:
                start = self.intrinsic_motivation.find_executable_neuron(all_coordinates)
                end = self.intrinsic_motivation.find_executable_neuron(all_coordinates)

                if start != end:
                    base_path = self.intrinsic_motivation.astar_with_length(executable_map, start, end, lengthOfPolicies+2)
                    print(type(base_path))
                    if base_path and len(base_path) == lengthOfPolicies + 2:  # A* incluye inicio y fin
                        base_path = base_path[1:-1]  # Excluir inicio y fin
                        break

            
        sets_and_buffers = {}
        for policy_index in range(numberOfPolicies):
            if policy_index == 0:
                # Primera política: exactamente el camino A*
                policy_path = [coordinates[0]] + base_path 
            else:
                # Variar ligeramente las demás políticas
                policy_path = [coordinates[0]]  # Añadir inicio
                viable_points = set(
                    (x, y)
                    for x in range(min(start[0],  end[0]), max(start[0], end[0]) + 1)
                    for y in range(min(start[1], end[1]), max(start[1], end[1]) + 1)
                    if executable_map[x][y] == 0 and (x, y) not in policy_path and (x,y) != end
                )

                # Expandir el área hasta encontrar suficientes puntos
                step = 1
                while len(viable_points) < lengthOfPolicies:
                    new_points = self.intrinsic_motivation.expand_viable_area(start, end, policy_path, somVisual.get_weights().shape[0], somVisual.get_weights().shape[1], executable_map, step)
                    viable_points.update(new_points)
                    step += 1
                    if step > (somVisual.get_weights().shape[0] + somVisual.get_weights().shape[1]):  # Prevenir ciclos infinitos
                        raise ValueError(
                            f"No se encontraron suficientes puntos viables para la tarea {task_idx}."
                        )

                # Seleccionar puntos únicos al azar
                policy_path += random.sample(viable_points, lengthOfPolicies)

            sets_and_buffers[f"Policy_{policy_index}"] = {
                "Set": tuple(policy_path),
                "Buffer": [10 * (i + 1) for i in range(lengthOfBuffers)]  # Buffer inicial
            }
    
        new_task_data = {
            "Coordinates": coordinates,
            "Sets_and_Buffers": sets_and_buffers
        }
        print(f"New tasks and policies: {new_task_data}")
        # # Add the new task to the dictionary
        task_dictionary[task_key] = new_task_data
        print(f"New Task {task_idx} added to the dictionary with coordinates: {coordinates}")

    def save_task_dictionary_to_txt(self, task_dictionary, file_name, iteration):
        """
        Save the task_dictionary to a text file.
        
        Args:
            - task_dictionary: The dictionary to be saved in the file.
            - file_name: The name of the text file.
            - iteration: The iteration the simulation is currently at.
        """

        dict_copy = self.json_serializable_copy(task_dictionary)

        try:
            with open(file_name, 'a') as file:
                file.write(f"\n\n### Iteration {iteration} ###\n")

                json.dump(dict_copy, file, indent=4)
                print(f"Task dictionary saved successfully to {file_name}.")
        except Exception as e:
            print(f"Error saving task dictionary: {e}")

    def json_serializable_copy(self, data):
        """
        Recursively convert sets to lists in the dictionary to make it JSON serializable.
        
        Args:
        - data: The input data.
        
        Returns:
        - A new dictionary with sets converted to lists.
        """
        if isinstance(data, dict):
            return {key: self.json_serializable_copy(value) for key, value in data.items()}
        elif isinstance(data, set):
            return list(data)  
        elif isinstance(data, list):
            return [self.json_serializable_copy(item) for item in data]
        else:
            return data  
    
    def clear_previous_file(self, file_name):
        """
        Delete the file if it exists
        
        Args:
        - file_name: The name of the text file (default is 'task_dictionary_debug.txt').
        """
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"{file_name} has been deleted to start a fresh simulation.")
        else:
            print(f"{file_name} does not exist, starting fresh.")
          

####train data
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
            
            
                        
#####load SOMS           
with open('somVisual.p', 'rb') as infile:
    somVisual = pickle.load(infile)

with open('somAngles.p', 'rb') as infile:
    somAngles = pickle.load(infile)

#####load hebbian table
    
# Crear una instancia de HebbianTable
hebbian_table = hebbian.HebbianTable()
# Inicializar la tabla Hebbiana con las SOMs y un factor de aprendizaje
hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

hebbian_table.loadFromFile("hebbian_table_new.txt")
    