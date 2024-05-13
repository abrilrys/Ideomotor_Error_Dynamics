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
    
    def learnUsingWinners(self, logfile, input_som1, input_som2):
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
        
        logfile.write("{som1_winner} {som2_winner} ")
        
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
        print("BMU SOM 1: ", som1_winner)
        
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

        
    def findAndEnableDevices(self):
        # get the time step of the current world.
        self.timeStep = int(self.getBasicTimeStep())

        # camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # gps
        self.gps = self.getDevice('hand_gps')
        self.gps.enable(1)
        #self.printGps()


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
            gps_entry = self.gps.getValues()
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
        
        
    def hebbianTrain(self):
        random.seed(10)
            
        i = 0  # Initialize iteration counter
        max_iterations = 100
    
        #loop_delay = 0.5  # Adjust the delay as needed
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
            gps_entry = self.gps.getValues()
            
            #normalize
            motor_entry= min_max_normalize_with_data(motor_entry, motor_data)
            gps_entry= min_max_normalize_with_data(gps_entry, gps_data)
            
            #print(self.gps.getSamplingPeriod())
            #print('----------gps----------')
            #print('position: [ x y z ] = [%f %f %f]' % (gps_data[0], gps_data[1], gps_data[2]))
            time.sleep(1)
            #call hebbian table
            logfile_path = "hebbian.txt"
            with open(logfile_path, "w") as logfile:
                # Llamar a los métodos de la clase HebbianTable y pasar el archivo de registro como argumento
                hebbian_table.learnUsingWinners(logfile, gps_entry, motor_entry)
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
            max_iterations = 100
    
            #loop_delay = 0.5  # Adjust the delay as needed
            while robot.step(self.timeStep) != -1:
            
                # Generate random angles within the specified range
                randomShoulderPitch =  round(random.uniform(self.minRShoulderPitchPosition,self.maxRShoulderPitchPosition),4)
                randomShoulderRoll = round(random.uniform(self.minRShoulderRollPosition, self.maxRShoulderRollPosition),4)
                randomElbowYaw = round(random.uniform(self.minRElbowYawPosition, self.maxRElbowYawPosition),4)
                randomElbowRoll = round(random.uniform(self.minRElbowRollPosition, self.maxRElbowRollPosition),4)
                
                # Set the random angles using the function
                self.setArmAngle(randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll)
                   
                # Get GPS data
                gps_data = self.gps.getValues()
                #print(self.gps.getSamplingPeriod())
                #print('----------gps----------')
                #print('position: [ x y z ] = [%f %f %f]' % (gps_data[0], gps_data[1], gps_data[2]))
                time.sleep(1)
                motor_writer.writerow([i, randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll])
                gps_writer.writerow([i,gps_data[0], gps_data[1], gps_data[2]])
                
                i += 1
                
                if i>=max_iterations:
                    break
               


somAngles = None
somVisual = None
somCombined = None

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
    data = pd.read_csv('motor_angles.csv', names=columns, sep=',', engine='python')


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
    n_neurons = 7 #5*sqrt(N) where N is the number of samples in the dataset
    m_neurons = 7

    somAngles = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, neighborhood_function='gaussian', random_seed=0, topology='rectangular')

    somAngles.pca_weights_init(data)
    somAngles.train(data, 1000, verbose=True)  # random training
    
    # saving the som
    with open('somAngles.p', 'wb') as outfile:
        pickle.dump(somAngles, outfile)
    
    
    
def generateVisualSOM():
    
    global somVisual
    
    columns = ['Key','X', 'Y', 'Z']
    data = pd.read_csv('gps_hand.csv', names=columns, sep=',', engine='python')
    
    
    target = data['Key'].values
    
    # Remove first column 
    data = data[data.columns[1:]]
    # Data normalization
    data = min_max_normalize(data)
    #(data - np.mean(data, axis=0)) / np.std(data, axis=0)
    data = data.values
    
    
    # Initialization and training
    n_neurons = 7
    m_neurons = 7
    
    somVisual = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5, 
                  neighborhood_function='gaussian', random_seed=0, topology='rectangular')
    
    somVisual.pca_weights_init(data)
    somVisual.train(data, 1000, verbose=True)  # random training
    
    # saving the som
    with open('somVisual.p', 'wb') as outfile:
        pickle.dump(somVisual, outfile)
    
    
def generateSOIMA():
    
    global somCombined
    
    motor_data = []
    gps_data = []
    
    # Leer los angulos de los motores del CSV
    with open("motor_angles.csv", "r", newline='') as motor_csvfile:
        motor_reader = csv.reader(motor_csvfile)
       
        for row in motor_reader:
            motor_data.append([float(value) for value in row[1:]])  # Pasar la columna del indice
    
    # Leer los datos del GPS del CSV
    with open("gps_hand.csv", "r", newline='') as gps_csvfile:
        gps_reader = csv.reader(gps_csvfile)
       
        for row in gps_reader:
            gps_data.append([float(value) for value in row[1:]])  # Pasar la columna del indice
    
    #print(somVisual.winner(random.choice(gps_data)));
    
    #normalize
    motor_data= min_max_normalize(motor_data)
    gps_data= min_max_normalize(gps_data)
    
    # Obtener todas las coordinadas del som visual
    visual_coordinates = [somVisual.winner(entry) for entry in gps_data]
    # Obtener todas las coordinadas del som motor
    motor_coordinates = [somAngles.winner(entry) for entry in motor_data]
    
    
    # Crear un nuevo conjunto de datos que consiste en todas las combinaciones de las coordenadas
    combined_dataset = []
    
    for visual_coord, motor_coord in zip(visual_coordinates, motor_coordinates):
        combined_entry = visual_coord + motor_coord  # Concatenate the two coordinate vectors
        combined_dataset.append(combined_entry)
    
    combined_dataset= pd.DataFrame(combined_dataset)
    combined_dataset=combined_dataset.values
    
    #print(combined_dataset.size)
    #print(combined_dataset)
    
    #Initialization and training
    n_neurons_combined = 10
    m_neurons_combined = 10
    
    
    somCombined = MiniSom(n_neurons_combined, m_neurons_combined, combined_dataset.shape[1], sigma=1.5, learning_rate=.5, 
                      neighborhood_function='gaussian', random_seed=0, topology='rectangular')
        
    somCombined.pca_weights_init(combined_dataset)
    somCombined.train(combined_dataset, 1000, verbose=True)  # random training
    
    # saving the som
    with open('soima.p', 'wb') as outfile:
        pickle.dump(somCombined, outfile)

# create the Robot instance and run main loop
robot = Nao()
#robot.run()

####train data
motor_data = []
gps_data = []
    
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

#train
#generateAnglesSOM()
#generateVisualSOM()
#generateSOIMA()

#hebbian_table = HebbianTable()
#hebbian_table.init(somVisual, somAngles, learning_factor=0.1)

#robot.hebbianTrain()
#hebbian_table.saveTable("hebbian_table_new.txt")


#load            
with open('somVisual.p', 'rb') as infile:
    somVisual = pickle.load(infile)

with open('somAngles.p', 'rb') as infile:
    somAngles = pickle.load(infile)

with open('soima.p', 'rb') as infile:
    somCombined = pickle.load(infile)
    
# Crear una instancia de HebbianTable
hebbian_table = HebbianTable()
# Inicializar la tabla Hebbiana con las SOMs y un factor de aprendizaje
hebbian_table.init(somVisual, somAngles, learning_factor=0.1)



hebbian_table.loadFromFile("hebbian_table_new.txt")

robot.hebbianTest(1)

