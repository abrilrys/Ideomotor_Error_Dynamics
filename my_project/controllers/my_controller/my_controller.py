from controller import Robot, Keyboard, Motion
import random
import numpy as np
import pandas as pd
import time
import csv
import pickle
import minisom
from minisom import MiniSom
import array
import math


class HebbianTable:
    def __init__(self):
        self.som1 = None
        self.som2 = None
        self.som_size1 = 0
        self.som_size2 = 0
        self.eta = 0.0
        self.neighbors_size = 0
        self.hasTam = 0
        self.axons = []
        self.status = []

    def init(self, s1, s2, learning_factor):
        self.eta = learning_factor
        self.som1 = s1
        self.som2 = s2
        #sn.get_weights().shape[0]*sn.get_weights().shape[1] = total number of neurons in the som
        self.som_size1 = s1.get_weights().shape[0]*s1.get_weights().shape[1]
        self.som_size2 = s2.get_weights().shape[0]*s2.get_weights().shape[1]
        self.hasTam = 50000
        self.crea_Hash(self.hasTam)

    def loadFromFile(self, filename):
        print(f"Loading hebbian table from {filename}")
        with open(filename, "r") as inputfile:
            lines = inputfile.readlines()
            self.axons = np.zeros(self.hasTam)
            self.status = np.zeros((self.hasTam, 2), dtype=int)
            for i, line in enumerate(lines):
                parts = line.split()
                self.status[i][0] = int(parts[0])
                self.status[i][1] = int(parts[1])
                self.axons[i] = float(parts[2])
        print("Hebbian table loaded.")
    
    def cantor_pairing(self, x, y):
        return ((x + y) * (x + y + 1))/2 + y
    
    def decantor_pairing(self,z):
        w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
        t = (w**2 + w) / 2
        y = int(z - t)
        x = int(w - y)
        return (x, y)
    
    def learnUsingWinners(self, input_som1, input_som2):
        distance1, distance2 = 0.0, 0.0
        
        #get the map weight of each som
        weights_som1 = self.som1.get_weights()
        weights_som2 = self.som2.get_weights()
        
        #get the coordinate of the winner neuron
        som1_winner = self.som1.winner(input_som1)
        som2_winner = self.som2.winner(input_som2)
        
        #get the weight of the winner neuron
        weight_winner_som1=weights_som1[som1_winner[0], som1_winner[1]]
        weight_winner_som2=weights_som2[som2_winner[0], som2_winner[1]]
       
        
        #get the euclidean distance from the input vector to the winner neuron
        distance1 = self.som1._euclidean_distance(input_som1,weight_winner_som1) #1-
        distance2 = self.som2._euclidean_distance(input_som2, weight_winner_som2) #1-
        
        #get the unique index of the neuron
        u_index_neuron_som1 = int(self.cantor_pairing(som1_winner[0], som1_winner[1]))
        u_index_neuron_som2 = int(self.cantor_pairing(som2_winner[0], som2_winner[1]))
       
        #set a unique index to save the weigth to the table
        position = (u_index_neuron_som1 * self.som_size2) + u_index_neuron_som2
                
        peso = self.eta * distance1 * distance2
        
        indice = self.busca_Hash(self.hasTam, position, 0)
        
        if indice == -1:
            self.insrew_Hash(self.hasTam, position, peso)
        else:
            self.insrew_Hash(self.hasTam, position, peso + self.axons[indice])

    def funcion(self, k, m, i):
        return ((k + i) % m)

    def crea_Hash(self, m):
        self.axons = np.zeros(m)
        self.status = np.zeros((m, 2), dtype=int)

    #retorna la posición 
    def busca_Hash(self, m, k, i):
        j = 0
        if i < m:
            j = self.funcion(k, m, i)
            if self.status[j][1] == 0:
                return -1
            elif self.status[j][1] == 1:
                return self.busca_Hash(m, k, i + 1)
            elif self.status[j][0] == k:
                return j
            else:
                return self.busca_Hash(m, k, i + 1)
        return -1

    #actualiza o ingresa el peso si aun no está en la tabla
    def insrew_Hash(self, m, k, peso):
        i = 0
        l = self.busca_Hash(m, k, 0)
        if l == -1:
            while i < m:
                j = self.funcion(k, m, i)
                if self.status[j][1] == 0 or self.status[j][1] == 1:
                    self.status[j][0] = k
                    self.status[j][1] = 2
                    self.axons[j] = peso
                    return
                else:
                    i += 1
            print("\nTabla hash llena!\n")
        else:
            self.axons[l] = peso

    #retorna el peso
    def axonsbypos(self, som1indice, som2indice):
        position = som1indice * self.som_size2 + som2indice
        indice = self.busca_Hash(self.hasTam, position, 0)
        if indice == -1:
            return 0.0
        else:
            return self.axons[indice]
            
    def getConectionsFromSOM1(self, som1_vector):
        
        
        #get winner neuron
        som1_winner = self.som1.winner(som1_vector)
        #print("BMU SOM 1: ", som1_winner)
        
        #get the coordinates from som2
        coordinates = []
        
        for row in range(self.som2.get_weights().shape[0]):
            for col in range(self.som2.get_weights().shape[1]):
                coordinates.append((row, col))
        
        #get the unique index of the neuron
        u_index_neuron_som1 = int(self.cantor_pairing(som1_winner[0], som1_winner[1]))
        
        u_indices = []
        for coord in coordinates:
            u_index = int(self.cantor_pairing(coord[0], coord[1]))
            u_indices.append(u_index)
        
        connected_neurons = []
        min_activation = float('inf')  # Initialize minimum activation to a large value
        min_activation_neuron = None
        
        for som2_unit in u_indices:
            # Calcular el índice correspondiente en la tabla Hebbiana
            position = u_index_neuron_som1 * self.som_size2 + som2_unit
            # Buscar en la tabla Hebbiana
            indice = self.busca_Hash(self.hasTam, position, 0)
            #print(indice)
            # Si hay una conexión, agregar la neurona som2 a la lista de conexiones
            if indice != -1 and self.axons[indice] < min_activation:
                connected_neurons.append(som2_unit)
                min_activation = self.axons[indice]
                min_activation_neuron = som2_unit
        
        if min_activation_neuron != None:
            return self.decantor_pairing(min_activation_neuron)
        else:
            return None
            
    def getConectionsFromSOM2(self, som2_vector):
        
        #get winner neuron
        som2_winner = self.som2.winner(som2_vector)
        print("BMU SOM 2: ", som2_winner)
        
        #get the coordinates from som1
        coordinates = []
        
        for row in range(self.som1.get_weights().shape[0]):
            for col in range(self.som1.get_weights().shape[1]):
                coordinates.append((row, col))
        
        #get the unique index of the neuron
        u_index_neuron_som2 = int(self.cantor_pairing(som2_winner[0], som2_winner[1]))
        
        u_indices = []
        for coord in coordinates:
            u_index = int(self.cantor_pairing(coord[0], coord[1]))
            u_indices.append(u_index)
        
        connected_neurons = []
        min_activation = float('inf')  # Initialize minimum activation to a large value
        min_activation_neuron = None
        
        for som1_unit in u_indices:
            # Calcular el índice correspondiente en la tabla Hebbiana
            position = som1_unit * self.som_size2 + u_index_neuron_som2
            # Buscar en la tabla Hebbiana
            indice = self.busca_Hash(self.hasTam, position, 0)
            #print(indice)
            # Si hay una conexión, agregar la neurona som2 a la lista de conexiones
            if indice != -1 and self.axons[indice] < min_activation:
                connected_neurons.append(som1_unit)
                min_activation = self.axons[indice]
                min_activation_neuron = som1_unit
        
        if min_activation_neuron != None:
            return self.decantor_pairing(min_activation_neuron)
        else:
            return None
            


    def saveTable(self, filename):
        print("Saving hebbian table to {filename}")
        with open(filename, "w") as myfile:
            for x1 in range(self.hasTam):
                myfile.write(f"{self.status[x1][0]} {self.status[x1][1]} {self.axons[x1]}\n")


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
               


somAngles = None
somVisual = None
somTasks = None

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
    
    
def generateTaskSOM():
    
    global somTasks
    
    data = pd.read_csv('tasks_train_dataset.csv', sep=',', engine='python', header=None)
    
    # Data normalization
    data = min_max_normalize(data)
    data = data.values
    
    num_samples = len(data)
    # Calculate the number of neurons using the rule of thumb
    num_neurons = 5 * math.sqrt(num_samples)
    # Determine the grid size
    grid_size = int(math.ceil(math.sqrt(num_neurons)))
       
    #Initialization and training
    
    somTasks = MiniSom(grid_size, grid_size, data.shape[1], sigma=1.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=0, topology='rectangular')
        
    somTasks.pca_weights_init(data)
    somTasks.train(data, 1000, verbose=True)  # random training
    
    print("SOM Tasks trained")
    
    # saving the som
    with open('somTasks.p', 'wb') as outfile:
        pickle.dump(somTasks, outfile)
        

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
#hebbian_table = HebbianTable()
#hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

#robot.hebbianTrain()
#hebbian_table.saveTable("hebbian_table_new.txt")



#####load hebbian table
    
# Crear una instancia de HebbianTable
hebbian_table = HebbianTable()
# Inicializar la tabla Hebbiana con las SOMs y un factor de aprendizaje
hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

hebbian_table.loadFromFile("hebbian_table_new.txt")

#robot.hebbianTest(1)

######################################################################################

# # Define the dimensions of the SOM
# som_height = somVisual.get_weights().shape[0]  # Number of rows
# som_width = somVisual.get_weights().shape[1]   # Number of columns

# # Generate all possible coordinates
# all_coordinates = [(row, col) for row in range(som_height) for col in range(som_width)]

# # Shuffle the list of coordinates
# random.shuffle(all_coordinates)

# # Select 10 unique pairs of coordinates without repetition
# selected_pairs = []
# selected_pairs_count = 0

# numberOfTasks = 10

# while selected_pairs_count < numberOfTasks:
#     # Select two random coordinates
#     coord1 = random.choice(all_coordinates)
#     coord2 = random.choice(all_coordinates)
    
#     # Ensure the pair is unique and not repeated
#     if coord1 != coord2 and (coord1, coord2) not in selected_pairs and (coord2, coord1) not in selected_pairs:
#         selected_pairs.append((coord1, coord2))
#         selected_pairs_count += 1

# # Create a dictionary to store the data
# data_dict = {}

# numberOfPolicies = 4
# lengthOfPolicies = 10
# lengthOfBuffers = 10

# # Populate the dictionary with selected pairs, associated sets, and buffers
# for pair in selected_pairs:
#     sets_and_buffers = {}
#     for i in range(numberOfPolicies):
#         set_pairs = set()
#         # Add random coordinates not equal to the pair coordinates
#         while len(set_pairs) < lengthOfPolicies:
#             random_coord = random.choice(all_coordinates)
#             if random_coord != pair[0] and random_coord != pair[1]:
#                 set_pairs.add(random_coord)
#         # Initialize buffer with zeros
#         buffer = []
#         valor = 0
#         while len(buffer) < lengthOfBuffers:
#             buffer.append(valor)
#             valor += 10
#         #print(buffer)
#         sets_and_buffers[tuple(set_pairs)] = buffer
#     data_dict[pair] = {"Sets_and_Buffers": sets_and_buffers}


# #for pair, values in data_dict.items():
#  #   print("Pair:", pair)
#   #  print("Associated Sets and Buffers:")
#    # for set_pairs, buffer in values["Sets_and_Buffers"].items():
#     #    print("Set:", set_pairs)
#      #   print("Associated Buffer:", buffer)
#       #  print()
#     #print()  
    

# for pair, values in data_dict.items():
#     visual_goal= denormalize_vector(somVisual.get_weights()[pair[1][0], pair[1][1]],gps_data)
#     for set_pairs, buffer in values["Sets_and_Buffers"].items():
#         # Calculate predictive error for each coordinate in set_pairs
#         for idx, coord in enumerate(set_pairs):
#             visual_input = denormalize_vector(somVisual.get_weights()[coord[0], coord[1]], gps_data)
#             motor_angles_coord = hebbian_table.getConectionsFromSOM1(visual_input)
#             #print(visual_input)
            
            
#             #print(motor_angles_coord)
#             if motor_angles_coord != None:
#                 rotation_angles=denormalize_vector(somAngles.get_weights()[motor_angles_coord[0], motor_angles_coord[1]],motor_data)
#                 #print(rotation_angles)
                
#                 # Assume the final goal is the second coordinate of the pair
#                 predictive_error = robot.executeMovement(rotation_angles, visual_goal)
#                 # Store predictive error in the buffer
#                 if idx < len(buffer):  # Ensure we don't go out of bounds
#                     buffer[idx] = predictive_error
    


    
# #Get the vector to train tasks som
# feature_vectors = []

# for pair, values in data_dict.items():
#     for set_pairs, buffer in values["Sets_and_Buffers"].items():
#         feature_vectors.append(buffer)

# # Convert to numpy array
# tasks_array = np.array(feature_vectors)

# # Save into CSV file
# np.savetxt('tasks_train_dataset.csv', tasks_array, delimiter=',', fmt='%.6f')

# ########TRAIN TASK SOM    
# generateTaskSOM()  

# #linear regression
# def estimate_coef(x, y):
#     n = np.size(x)
#     #mean of x and y vector
#     m_x = np.mean(x)
#     m_y = np.mean(y)
    
#     SS_xy = np.sum(y*x) - n*m_y*m_x
#     SS_xx = np.sum(x*x) - n*m_x*m_x
    
#     b_1 = SS_xy / SS_xx
#     b_0 = m_y - b_1*m_x
#     return (b_0, b_1)
    

# time_buffer = np.arange(tasks_array.shape[1])

# #get the slopes for each error buffer
# slopes = []

# for buffer in tasks_array:
#     lin_reg=estimate_coef(time_buffer, buffer)
#     slope = lin_reg[1]
#     slopes.append(slope)

# slopes = np.array(slopes)
# print("Pendientes de las regresiones lineales sobre los buffers:\n", slopes)

#######################################################################################
class IntrinsicMotivation:
    def __init__(self):
        self.lengthOfBuffers=11
        self.task_dictionary=self.initialize_task_dictionary()
        self.slopes=self.get_slopes()
        #print(self.slopes)
        
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
        for task_index, pair in enumerate(selected_pairs, start=1):
            sets_and_buffers = {}
            for policy_index in range(1, numberOfPolicies + 1):
                set_pairs = set()
                # Add random coordinates not equal to the pair coordinates
                while len(set_pairs) < lengthOfPolicies:
                    random_coord = random.choice(all_coordinates)
                    if random_coord != pair[0] and random_coord != pair[1]:
                        set_pairs.add(random_coord)
                        
                set_pairs = list(set_pairs)
                set_pairs.insert(0, pair[0])
                #print(set_pairs)
                # Initialize buffer with zeros
                buffer = []
                valor = 0
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
        
        # Convert the list of buffers to a NumPy array
        tasks_array = np.array(feature_vectors)
        
        # Save the NumPy array into a CSV file
        np.savetxt('tasks_train_dataset.csv', tasks_array, delimiter=',', fmt='%.6f')
        
        print("Buffers saved into 'tasks_train_dataset.csv'")
        return data_dict
        
    def print_task_dict(self):
        for task, content in self.task_dictionary.items():
            print(task, content)
       
    def get_best_goal(self):
        best_goal=1
        return best_goal
        
    def get_random_goal(self):
        random_goal=1
        return random_goal
        
    def change_policy(self):
        self.goal=1
        
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
    

intrin= IntrinsicMotivation()		
    
class Experiment:
    
    def __init__(self, eps, iterations):
        self.intrinsic_motivation = IntrinsicMotivation()
        self.eps=eps
        self.iterations=iterations
        
        self.current_goal_idx = -1
        self.prev_goal_idx = -1
    
    def run_exp(self):
        for _ in range(1, self.iterations):
            # get the current task using the intrinsic motivation strategy and e-greedy algorithm
            p = np.random.random() 
            if p < self.eps: 
                #select random task
                self.current_goal_idx = 1 
            else: 
                #select best rated task
                self.current_goal_idx = 1 