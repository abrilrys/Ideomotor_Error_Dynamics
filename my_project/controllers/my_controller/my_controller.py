from controller import Robot, Keyboard, Motion # type: ignore
import random
import numpy as np
import pandas as pd
import time
import csv
import pickle
from minisom import MiniSom
import math
import json
import os
import HebbianTable as hebbian

class Nao (Robot):
    PHALANX_MAX = 8
    maxRShoulderPitchPosition=0
    minRShoulderPitchPosition=0
    maxRShoulderRollPosition=0
    minRShoulderRollPosition=0
    maxRElbowYawPosition=0
    minRElbowYawPosition=0
    maxRElbowRollPosition=0
    minRElbowRollPosition=0

    
    def setArmAngle(self, angleShoulderPitch, angleShoulderRoll, angleElbowYaw, angleElbowRoll):
        clampedAngleShoulderPitch = angleShoulderPitch
        if clampedAngleShoulderPitch > self.maxRShoulderPitchPosition:
            clampedAngleShoulderPitch = self.maxRShoulderPitchPosition
        elif clampedAngleShoulderPitch < self.minRShoulderPitchPosition:
            clampedAngleShoulderPitch = self.minRShoulderPitchPosition
            
        clampedAngleShoulderRoll = angleShoulderRoll
        if clampedAngleShoulderRoll > self.maxRShoulderRollPosition:
          clampedAngleShoulderRoll = self.maxRShoulderRollPosition
        elif clampedAngleShoulderRoll < self.minRShoulderRollPosition:
          clampedAngleShoulderRoll = self.minRShoulderRollPosition
      
        clampedAngleElbowYaw = angleElbowYaw
        if clampedAngleElbowYaw > self.maxRElbowYawPosition:
            clampedAngleElbowYaw = self.maxRElbowYawPosition
        elif clampedAngleElbowYaw < self.minRElbowYawPosition:
            clampedAngleElbowYaw = self.minRElbowYawPosition
            
        clampedAngleElbowRoll = angleElbowRoll
        if clampedAngleElbowRoll > self.maxRElbowRollPosition:
            clampedAngleElbowRoll = self.maxRElbowRollPosition
        elif clampedAngleElbowRoll < self.minRElbowRollPosition:
            clampedAngleElbowRoll = self.minRElbowRollPosition
  
  
        self.RShoulderPitch.setPosition(clampedAngleShoulderPitch)
        self.RShoulderRoll.setPosition(clampedAngleShoulderRoll)
        self.RElbowYaw.setPosition(clampedAngleElbowYaw)
        self.RElbowRoll.setPosition(clampedAngleElbowRoll)
      

    def printGps(self):
        p = self.gps.getValues()
        print('----------gps----------')
        print('position: [ x y z ] = [%f %f %f]' % (p[0], p[1], p[2]))

    def printCameraImage(self, camera):
        scaled = 2  # defines by which factor the image is subsampled
        width = camera.getWidth()
        height = camera.getHeight()

        # read rgb pixel values from the camera
        image = camera.getImage()

        print('----------camera image (gray levels)---------')
        print('original resolution: %d x %d, scaled to %d x %f'
              % (width, height, width / scaled, height / scaled))

        for y in range(0, height // scaled):
            line = ''
            for x in range(0, width // scaled):
                gray = camera.imageGetGray(image, width, x * scaled, y * scaled) * 9 / 255  # rescale between 0 and 9
                line = line + str(int(gray))
            print(line)
    
    def getRelativeCoords(self):
        # Leer la posición global del GPS de la mano
        gps_hand_coords = self.gps.getValues()
        
        # Leer la posición del cuerpo del NAO
        gps_body_coords = self.gps_body.getValues()
        
        # Calcular la posición relativa del GPS con respecto al cuerpo
        relative_coords = [gps_hand_coords[i] - gps_body_coords[i] for i in range(3)]
        #print("Coordenadas GPS relativas:", relative_coords)
        return relative_coords
        
    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())

        # camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # gps_hand
        self.gps = self.getDevice('hand_gps')
        self.gps.enable(1)
        #self.printGps()
        
        self.gps_body= self.getDevice('gps_body')
        self.gps_body.enable(1)

        # right arm motors
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.RShoulderRoll=self.getDevice("RShoulderRoll")
        self.RElbowYaw=self.getDevice("RElbowYaw")
        self.RElbowRoll=self.getDevice("RElbowRoll")
        
        self.maxRShoulderPitchPosition=self.RShoulderPitch.getMaxPosition();
        self.minRShoulderPitchPosition=self.RShoulderPitch.getMinPosition();
        self.maxRShoulderRollPosition=self.RShoulderRoll.getMaxPosition();
        self.minRShoulderRollPosition=self.RShoulderRoll.getMinPosition();
        self.maxRElbowYawPosition=self.RElbowYaw.getMaxPosition();
        self.minRElbowYawPosition=self.RElbowYaw.getMinPosition();
        self.maxRElbowRollPosition=self.RElbowRoll.getMaxPosition();
        self.minRElbowRollPosition=self.RElbowRoll.getMinPosition();
        #print("Shoulder Pitch max:", self.maxRShoulderPitchPosition, "min :", self.minRShoulderPitchPosition);
        #print("Shoulder Roll max:", self.maxRShoulderRollPosition, "min :", self.minRShoulderRollPosition);
        #print("Elbow Yaw Pitch max:", self.maxRElbowYawPosition, "min :", self.minRElbowYawPosition);
        #print("Elbow Roll max:", self.maxRElbowRollPosition, "min :", self.minRElbowRollPosition);
        
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        Robot.__init__(self)

        # initialize stuff
        self.findAndEnableDevices()
    
    #función en donde se puede testear las conexiones del som 1 al som2 y viceversa    
    def hebbianTest(self, choice):
        #choice =1 if you want to test from som 1 to som 2
        #choice =2 if you want to test from som 2 to som 1
        random.seed(10)
        
        while robot.step(self.timeStep) != -1:
         # Generate random angles within the specified range
            randomShoulderPitch =  round(random.uniform(self.minRShoulderPitchPosition,self.maxRShoulderPitchPosition),4)
            randomShoulderRoll = round(random.uniform(self.minRShoulderRollPosition, self.maxRShoulderRollPosition),4)
            randomElbowYaw = round(random.uniform(self.minRElbowYawPosition, self.maxRElbowYawPosition),4)
            randomElbowRoll = round(random.uniform(self.minRElbowRollPosition, self.maxRElbowRollPosition),4)
            
            # Set the random angles using the function
            self.setArmAngle(randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll)
            
            motor_entry = [randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll]
            # Get GPS data
            gps_entry = self.getRelativeCoords()
            #print(gps_entry)
            #normalize
            motor_entry= min_max_normalize_with_data(motor_entry, motor_data)
            gps_entry= min_max_normalize_with_data(gps_entry, gps_data)
            #print(gps_entry)
            
            time.sleep(1)
            if choice == 1 :
                print("BMU SOM 2: ", hebbian_table.getConectionsFromSOM1(gps_entry))
            
            else:
                print("BMU SOM 1: ", hebbian_table.getConectionsFromSOM2(motor_entry))
        
    def executeMovement(self, rotation_angles, target_coordinate):
        while robot.step(self.timeStep) != -1:
            # Set the random angles using the function
            self.setArmAngle(rotation_angles[0], rotation_angles[1], rotation_angles[2], rotation_angles[3])
            # Get GPS data
            time.sleep(0.002)
            gps_entry = self.getRelativeCoords()
            # Calculate predictive error between current position and target
            pred_error = np.linalg.norm(np.array(gps_entry) - np.array(target_coordinate))
            #print("Prediction error: ", pred_error)
            break
        return pred_error

    def MoveArm(self, rotation_angles):
        while robot.step(self.timeStep) != -1:
            time.sleep(1)
            self.setArmAngle(rotation_angles[0], rotation_angles[1], rotation_angles[2], rotation_angles[3])
            break
        
    def hebbianTrain(self):
        random.seed(10)
            
        i = 0  # Initialize iteration counter
        max_iterations = 5000
        print("Initializing hebbian table training for "+ str(max_iterations) + " iterations. \t")

         # Mean and standard deviation for the normal distribution
        mu_ShoulderPitch = (self.minRShoulderPitchPosition + self.maxRShoulderPitchPosition) / 2
        sigma_ShoulderPitch = (self.maxRShoulderPitchPosition - self.minRShoulderPitchPosition) / 6  # Example std dev
    
        mu_ShoulderRoll = (self.minRShoulderRollPosition + self.maxRShoulderRollPosition) / 2
        sigma_ShoulderRoll = (self.maxRShoulderRollPosition - self.minRShoulderRollPosition) / 6  # Example std dev
    
        mu_ElbowYaw = (self.minRElbowYawPosition + self.maxRElbowYawPosition) / 2
        sigma_ElbowYaw = (self.maxRElbowYawPosition - self.minRElbowYawPosition) / 6  # Example std dev
    
        mu_ElbowRoll = (self.minRElbowRollPosition + self.maxRElbowRollPosition) / 2
        sigma_ElbowRoll = (self.maxRElbowRollPosition - self.minRElbowRollPosition) / 6  # Example std dev
            

        #loop_delay = 0.5  # Adjust the delay as needed
        while robot.step(self.timeStep) != -1:
            # Generate random angles within the specified range
            randomShoulderPitch = round(random.gauss(mu_ShoulderPitch, sigma_ShoulderPitch), 4)
            randomShoulderRoll = round(random.gauss(mu_ShoulderRoll, sigma_ShoulderRoll), 4)
            randomElbowYaw = round(random.gauss(mu_ElbowYaw, sigma_ElbowYaw), 4)
            randomElbowRoll = round(random.gauss(mu_ElbowRoll, sigma_ElbowRoll), 4)
            
            # Ensure the angles are within the specified range
            randomShoulderPitch = max(min(randomShoulderPitch, self.maxRShoulderPitchPosition), self.minRShoulderPitchPosition)
            randomShoulderRoll = max(min(randomShoulderRoll, self.maxRShoulderRollPosition), self.minRShoulderRollPosition)
            randomElbowYaw = max(min(randomElbowYaw, self.maxRElbowYawPosition), self.minRElbowYawPosition)
            randomElbowRoll = max(min(randomElbowRoll, self.maxRElbowRollPosition), self.minRElbowRollPosition)
                    
            # Set the random angles using the function
            self.setArmAngle(randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll)
            
            motor_entry = [randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll]
            # Get GPS data
            gps_entry = self.getRelativeCoords()
            
            #normalize
            motor_entry= min_max_normalize_with_data(motor_entry, motor_data)
            gps_entry= min_max_normalize_with_data(gps_entry, gps_data)
            
            print("Iteration "+ str(i)+ "\t")
            print("Waiting for 0.002s \t")
            time.sleep(0.002)
            hebbian_table.learnUsingWinners(gps_entry, motor_entry)
            
            i += 1
            
            if i>=max_iterations:
                break
    
    def run(self):
        
        with open("motor_angles.csv", "w",newline='') as  motor_csvfile, \
             open("gps_hand.csv", "w", newline='') as gps_csvfile:
            motor_writer = csv.writer(motor_csvfile)
            gps_writer = csv.writer(gps_csvfile)
            #motor_writer.writerow(["Index", "ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll"])
            #gps_writer.writerow(["Index", "X", "Y", "Z"])
            self.LShoulderPitch.setPosition(2)
            random.seed(10)
            
            i = 0  # Initialize iteration counter
            max_iterations = 5000
            print("Obtaining training set samples for "+ str(max_iterations) + " iterations. \t")
            # Mean and standard deviation for the normal distribution
            mu_ShoulderPitch = (self.minRShoulderPitchPosition + self.maxRShoulderPitchPosition) / 2
            sigma_ShoulderPitch = (self.maxRShoulderPitchPosition - self.minRShoulderPitchPosition) / 6  # Example std dev
        
            mu_ShoulderRoll = (self.minRShoulderRollPosition + self.maxRShoulderRollPosition) / 2
            sigma_ShoulderRoll = (self.maxRShoulderRollPosition - self.minRShoulderRollPosition) / 6  # Example std dev
        
            mu_ElbowYaw = (self.minRElbowYawPosition + self.maxRElbowYawPosition) / 2
            sigma_ElbowYaw = (self.maxRElbowYawPosition - self.minRElbowYawPosition) / 6  # Example std dev
        
            mu_ElbowRoll = (self.minRElbowRollPosition + self.maxRElbowRollPosition) / 2
            sigma_ElbowRoll = (self.maxRElbowRollPosition - self.minRElbowRollPosition) / 6  # Example std dev
            
            
            #loop_delay = 0.5  # Adjust the delay as needed
            while robot.step(self.timeStep) != -1:
                try:
                    # Generate random angles within the specified range
                    randomShoulderPitch = round(random.gauss(mu_ShoulderPitch, sigma_ShoulderPitch), 4)
                    randomShoulderRoll = round(random.gauss(mu_ShoulderRoll, sigma_ShoulderRoll), 4)
                    randomElbowYaw = round(random.gauss(mu_ElbowYaw, sigma_ElbowYaw), 4)
                    randomElbowRoll = round(random.gauss(mu_ElbowRoll, sigma_ElbowRoll), 4)
                    
                    # Ensure the angles are within the specified range
                    randomShoulderPitch = max(min(randomShoulderPitch, self.maxRShoulderPitchPosition), self.minRShoulderPitchPosition)
                    randomShoulderRoll = max(min(randomShoulderRoll, self.maxRShoulderRollPosition), self.minRShoulderRollPosition)
                    randomElbowYaw = max(min(randomElbowYaw, self.maxRElbowYawPosition), self.minRElbowYawPosition)
                    randomElbowRoll = max(min(randomElbowRoll, self.maxRElbowRollPosition), self.minRElbowRollPosition)
                    
                    # Set the random angles using the function
                    self.setArmAngle(randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll)
                    
                    
                    # Get GPS data
                    #gps_data = self.gps.getValues()
                    gps_data = self.getRelativeCoords()
                    #print(self.gps.getSamplingPeriod())
                    print('----------gps----------')
                    print('position: [ x y z ] = [%f %f %f]' % (gps_data[0], gps_data[1], gps_data[2]))
                    
                    time.sleep(0.002)
                    print("Waiting for 0.002s \t")
                       
                    
                    motor_writer.writerow([i, randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll])
                    gps_writer.writerow([i,gps_data[0], gps_data[1], gps_data[2]])
                    
                    print(i)
                    i += 1
                    
                    if i>=max_iterations:
                        break
                except Exception as e:
                    print(f"Error at iteration {i}: {e}")
                    break



#######################################################################################
class IntrinsicMotivation:
    #self.lengthOfBuffers
    #self.task_dictionary
    #self.slopes
    #self.buffers
    
    def __init__(self):
        self.lengthOfBuffers=11
        self.task_dictionary=self.initialize_task_dictionary()
        self.slopes=self.get_slopes()
        #self.print_task_dict()
        #print(self.buffers)
        
    def initialize_task_dictionary(self):
        # Define the dimensions of the SOM
        som_height = somVisual.get_weights().shape[0]  # Number of rows
        som_width = somVisual.get_weights().shape[1]   # Number of columns
        
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
            visual_goal = denormalize_vector(somVisual.get_weights()[pair[1][0], pair[1][1]], gps_data)
            
            for policy, policy_data in values["Sets_and_Buffers"].items():
                set_pairs = policy_data["Set"]
                buffer = policy_data["Buffer"]
                
                # Calculate predictive error for each coordinate in set_pairs
                for idx, coord in enumerate(set_pairs):
                    visual_input = denormalize_vector(somVisual.get_weights()[coord[0], coord[1]], gps_data)
                    motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
                    
                    if motor_angles_coord is not None:
                        rotation_angles = denormalize_vector(somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                        
                        # Assume the final goal is the second coordinate of the pair
                        predictive_error = robot.executeMovement(rotation_angles, visual_goal)
                        
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
        visual_goal = denormalize_vector(somVisual.get_weights()[coordinates[1][0], coordinates[1][1]], gps_data)

        # Calculate predictive error for each coordinate in set_pairs
        for idx, coord in enumerate(set_pairs):
            visual_input = denormalize_vector(somVisual.get_weights()[coord[0], coord[1]], gps_data)
            motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
            
            if motor_angles_coord is not None:
                rotation_angles = denormalize_vector(somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                
                # Assume the final goal is the second coordinate of the pair
                predictive_error = robot.executeMovement(rotation_angles, visual_goal)
                
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
        som_height= somVisual.get_weights().shape[0] 
        som_width = somVisual.get_weights().shape[1]

        som_weights=somVisual.get_weights()
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
        som_height = somVisual.get_weights().shape[0]  # Number of rows
        som_width = somVisual.get_weights().shape[1]   # Number of columns
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
    
class Experiment:
    
    def __init__(self, eps, duration):
        self.intrinsic_motivation = IntrinsicMotivation()
        self.eps=eps
        self.duration=duration
        self.buffer_agent=[]
        self.current_goal_idx = -1
        self.prev_goal_idx = -1

    def get_task_policy_from_index(self, index):
        # Calculate the task and policy index
        task_idx = index // 4  
        policy_idx = index % 4  
        
        return task_idx, policy_idx
    
    def run_exp(self):

        self.clear_previous_file("task_dictionary.txt")
        self.clear_previous_file("learnt_policies.json")

        start_time = time.time()
        
        it=0
        buffer_time=[]
        while time.time() - start_time < self.duration:
            
            print(f"######################################################\n Iteration {it}\n")

            self.prev_goal_idx =self.current_goal_idx

            # get the current task using the intrinsic motivation strategy and e-greedy algorithm
            p = np.random.random() 
            if p < self.eps: 
                #select random task
                self.current_goal_idx=self.intrinsic_motivation.get_random_goal()
                
                task_idx, policy_idx=self.get_task_policy_from_index(self.current_goal_idx)
                #get previous dynamic of said taks and policy
                prev_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))
            else: 
                #select best rated task
                self.current_goal_idx=self.intrinsic_motivation.get_best_goal()
                task_idx, policy_idx=self.get_task_policy_from_index(self.current_goal_idx)
                #get previous dynamic of said taks and policy
                prev_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))


            print(f"Previous dynamic: {prev_din}")
            
            
           
            #modify the policy
            self.intrinsic_motivation.change_policy(policy_idx, task_idx,3 )
            
            
            #update the PE
            self.intrinsic_motivation.update_buffer(policy_idx, task_idx)
            
           
            #get current dynamic of current goal
            new_din=np.copy(self.intrinsic_motivation.get_buffer_from_task_policy(task_idx, policy_idx))
            print(f"New dynamic: {new_din}")
            
            #get PE of dynamic
            mse = np.mean((np.array(prev_din) - np.array(new_din)) ** 2)
            print(f"Mean Squared Error: {mse}")
            
            self.buffer_agent.append(mse)
            
            buffer_time.append(it)
            #print(f"buffer time: {buffer_time}")
            it=it+1
           
            
            b0, b1=self.intrinsic_motivation.estimate_coef(np.array(buffer_time), np.array(self.buffer_agent))
            print(f"Agent behaviour: b0= {b0}, b1= {b1}")
          
            #check if the task is learnt
            evaluation_buffer=self.intrinsic_motivation.evaluate_buffer(new_din)
            goal_task=self.intrinsic_motivation.get_goal_from_task(task_idx)
            min_distance_to_neighbors=self.intrinsic_motivation.get_min_distance_to_neighbors(tuple(goal_task))
            #print("min_distance_to_neighbors > ", min_distance_to_neighbors)
            if evaluation_buffer[0]<min_distance_to_neighbors and evaluation_buffer[1]<0 and evaluation_buffer[2]:
                print("task learnt")
                self.save_policy_to_json("learnt_policies.json",task_idx, policy_idx, self.intrinsic_motivation.task_dictionary)
                self.remove_learned_task(task_idx,self.intrinsic_motivation.task_dictionary,"learnt_policies.json" )

            #self.save_task_dictionary_to_txt(self.intrinsic_motivation.task_dictionary, "task_dictionary.txt", it)      
    
    def save_policy_to_json(self,file_name, task_idx, policy_idx, task_dictionary):
        """
        Save a specific policy's coordinates and set of coordinates to a JSON file.
        
        Parameters:
        - file_name: Name of the JSON file to save to.
        - task_idx: The index of the task.
        - policy_idx: The index of the policy within the task.
        - task_dictionary: The task dictionary containing the task and policy data.
        """
        task_key = f"Task_{task_idx}"
        policy_key = f"Policy_{policy_idx}"
        
        policy_data = task_dictionary[task_key]["Sets_and_Buffers"][policy_key]

        coordinates = task_dictionary[task_key]["Coordinates"]
        
        # Structure the data
        policy_to_save = {
            "Coordinates": coordinates,
            "SetPairs": list(policy_data["Set"]),  # Convert set to a list for JSON compatibility
        }
        
        if os.path.exists(file_name):
            with open(file_name, 'r') as json_file:
                all_policies = json.load(json_file)

        else:
            all_policies = []  
        
        # Add the new policy to the list of all policies
        all_policies.append(policy_to_save)
        
        with open(file_name, 'w') as json_file:
            json.dump(all_policies, json_file, indent=4)
        
        print(f"Policy {policy_data['Set']} from task {coordinates} saved to {file_name}.")

    def load_all_policies_from_json(self,file_name):
        """
        Load all policies from a JSON file.
        
        Parameters:
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
        
        Parameters:
            - file_name: The name of the JSON file to load from.
        """
        all_policies = self.load_all_policies_from_json(file_name)
        
        for policy in all_policies:
            coordinates = policy["Coordinates"]
            set_pairs = policy["SetPairs"]

            set_pairs = list(set_pairs)[1:]  # Skip the first element
            
            merged_coordinates = [coordinates[0]] + set_pairs + [coordinates[1]]
            
            print("Executing Task Policy")
            print("Coordinates:", coordinates)
            print("Set Pairs:", set_pairs)

            print(f"Trajectory: {merged_coordinates} ")
            for idx, coord in enumerate(merged_coordinates):
                     visual_input = denormalize_vector(somVisual.get_weights()[coord[0], coord[1]], gps_data)
                     motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
                     
                     print(f"Point {idx}: {visual_input}")
                     if motor_angles_coord is not None:
                         rotation_angles = denormalize_vector(somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]], motor_data)
                         
                         robot.MoveArm(rotation_angles)

    def remove_learned_task(self, task_idx, task_dictionary, json_file):
        """
        Remove a learned task from the task_dictionary and replace it with a new task.
        The new task must have a unique pair of coordinates not present in the dictionary or JSON file.
        
        Parameters:
        - task_idx: The index of the learned task to be removed.
        - task_dictionary: The dictionary holding the tasks and their policies.
        - json_file: The file where the tasks are saved to ensure unique coordinates.
        """
        # Remove the task from the task_dictionary
        task_key = f"Task_{task_idx}"
        if task_key in task_dictionary:
            del task_dictionary[task_key]
            print(f"Task {task_idx} removed from the dictionary.")
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

    def generate_unique_coordinates(self,existing_coordinates):
        """
        Generate a unique pair of coordinates not present in the existing_coordinates set.
        
        Parameters:
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

    def add_new_task_to_dictionary(self,task_dictionary, coordinates, task_idx):
        """
        Add a new task with a set of policies to the task_dictionary.
        
        Parameters:
        - task_dictionary: The dictionary to which the new task will be added.
        - coordinates: The new pair of task coordinates.
        - task_idx: The task index of the new task.
        """
        task_key = f"Task_{task_idx}"

        # Initialize the new task with its coordinates
        numberOfPolicies = 4
        lengthOfPolicies = 10
        lengthOfBuffers = self.intrinsic_motivation.lengthOfBuffers
        new_task_data = {
            "Coordinates": coordinates,
            "Sets_and_Buffers": {}
        }

        # Add 4 policies with random set_pairs and buffers
        for policy_idx in range(0, numberOfPolicies):
            set_pairs = set()

            # Add random coordinates not equal to the task coordinates
            while len(set_pairs) < lengthOfPolicies:
                random_coord = (random.randint(0, somVisual.get_weights().shape[0] - 1), random.randint(0, somVisual.get_weights().shape[1] - 1))
                if random_coord != coordinates[0] and random_coord != coordinates[1]:
                    set_pairs.add(random_coord)

            set_pairs = list(set_pairs)
            set_pairs.insert(0, coordinates[0])

            # Initialize buffer with increasing values
            buffer = []
            valor = 10
            while len(buffer) < lengthOfBuffers:
                buffer.append(valor)
                valor += 10

            policy_key = f"Policy_{policy_idx}"

            new_task_data["Sets_and_Buffers"][policy_key] = {
                "Set": set_pairs,
                "Buffer": buffer
            }

        # Add the new task to the dictionary
        task_dictionary[task_key] = new_task_data
        print(f"New Task {task_idx} added to the dictionary with coordinates: {coordinates}")

    def save_task_dictionary_to_txt(self, task_dictionary, file_name, iteration):
        """
        Save the task_dictionary to a text file.
        
        Parameters:
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
        
        Parameters:
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
        
        Parameters:
        - file_name: The name of the text file (default is 'task_dictionary_debug.txt').
        """
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f"{file_name} has been deleted to start a fresh simulation.")
        else:
            print(f"{file_name} does not exist, starting fresh.")

#denormalize a vector given dataset
def denormalize_vector(normalized_vector, data):
    # Calculate min and max for each feature in the dataset
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    
    # Denormalize the vector
    denormalized_vector = []
    for i in range(len(normalized_vector)):
        denormalized_value = normalized_vector[i] * (max_values[i] - min_values[i]) + min_values[i]
        denormalized_vector.append(denormalized_value)
    return denormalized_vector
    
    
#normalize a data set
def min_max_normalize(x):
    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x

#normalize one value given a data set
def min_max_normalize_with_data(vector, data):
    # Determine the dimensionality of the vectors
    vector_dim = len(vector)
    
    # Transpose the data to get lists of components
    component_values = [[] for _ in range(vector_dim)]
    for entry in data:
        for i in range(vector_dim):
            component_values[i].append(entry[i])
    
    # Calculate min and max for each component
    min_values = [min(components) for components in component_values]
    max_values = [max(components) for components in component_values]
    
    # Normalize each component of the vector
    normalized_vector = []
    for i in range(vector_dim):
        if max_values[i] != min_values[i]:
            normalized_component = (vector[i] - min_values[i]) / (max_values[i] - min_values[i])
            # Clip the normalized value to be within [0, 1]
            normalized_component = max(0, min(normalized_component, 1))
        else:
            normalized_component = 0.5  # Default to 0.5 if max and min are equal
        normalized_vector.append(normalized_component)
    
    # Return the normalized vector
    return normalized_vector
    

def generateAnglesSOM():
    
    global somAngles
    
    # Load data
    columns = ['Key','RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
    data = pd.read_csv('motor_angles.csv', names=columns, sep=',', engine='python', header=None)


    target = data['Key'].values

    # Remove first column 
    data = data[data.columns[1:]]
    
    # Data normalization
    data = min_max_normalize(data)
    #print(type(data))
   
    
    data = data.values
    #print(data)
    #print(type(data))
    # Initialization and training
    num_samples = len(data)
    # Calculate the number of neurons using the rule of thumb
    num_neurons = 5 * math.sqrt(num_samples)
    # Determine the grid size
    grid_size = int(math.ceil(math.sqrt(num_neurons)))

    somAngles = MiniSom(grid_size, grid_size, data.shape[1], sigma=3, learning_rate=.5, neighborhood_function='gaussian', random_seed=0, topology='rectangular')

    somAngles.pca_weights_init(data)
    somAngles.train(data, 1000, verbose=True)  # random training
    
    print("SOM Sensorial trained")
    
    # saving the som
    with open('somAngles.p', 'wb') as outfile:
        pickle.dump(somAngles, outfile)
    
def generateVisualSOM():
    
    global somVisual
    
    columns = ['Key','X', 'Y', 'Z']
    data = pd.read_csv('gps_hand.csv', names=columns, sep=',', engine='python', header=None)
    
    
    target = data['Key'].values
    
    # Remove first column 
    data = data[data.columns[1:]]
    # Data normalization
    data = min_max_normalize(data)
    data = data.values
    
    num_samples = len(data)
    # Calculate the number of neurons using the rule of thumb
    num_neurons = 5 * math.sqrt(num_samples)
    # Determine the grid size
    grid_size = int(math.ceil(math.sqrt(num_neurons)))
    
    somVisual = MiniSom(grid_size, grid_size, data.shape[1], sigma=3, learning_rate=.5, 
                  neighborhood_function='gaussian', random_seed=0, topology='rectangular')
    
    somVisual.pca_weights_init(data)
    somVisual.train(data, 1000, verbose=True)  # random training
    
    print("SOM Visual trained")
    
    # saving the som
    with open('somVisual.p', 'wb') as outfile:
        pickle.dump(somVisual, outfile)

somAngles = None
somVisual = None

# create the Robot instance and run main loop
robot = Nao()
#robot.run()

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
            
####
#print(motor_data)

    
#####train SOMS
#generateAnglesSOM()
#generateVisualSOM()

#####load SOMS           
with open('somVisual.p', 'rb') as infile:
    somVisual = pickle.load(infile)

with open('somAngles.p', 'rb') as infile:
    somAngles = pickle.load(infile)

#####train hebbian table
#hebbian_table = hebbian.HebbianTable()
#hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

#robot.hebbianTrain()
#hebbian_table.saveTable("hebbian_table_new.txt")



#####load hebbian table
    
# Crear una instancia de HebbianTable
hebbian_table = hebbian.HebbianTable()
# Inicializar la tabla Hebbiana con las SOMs y un factor de aprendizaje
hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

hebbian_table.loadFromFile("hebbian_table_new.txt")

#robot.hebbianTest(1)
            
exp= Experiment(0.1, 300)
#exp.run_exp()	
exp.execute_loaded_policies("learnt_policies.json")

