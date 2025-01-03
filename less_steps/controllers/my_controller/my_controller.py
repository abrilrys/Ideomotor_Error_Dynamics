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
import tools
import IntrinsicMotivation as intrinsic
import Experiment as experimentation

import matplotlib.pyplot as plt


motor_tolerance=0.00001

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
        """
        Sets the angles of the right arm's joints (shoulder and elbow).

        The method clamps the input angles to ensure they are within the defined
        maximum and minimum position limits for each joint before applying them.

        Args:
            -angleShoulderPitch (float): The desired angle for the shoulder pitch joint.
            -angleShoulderRoll (float): The desired angle for the shoulder roll joint.
            -angleElbowYaw (float): The desired angle for the elbow yaw joint.
            -angleElbowRoll (float): The desired angle for the elbow roll joint.
        """        
        #print(f"Rotation entry: {angleShoulderPitch}, {angleShoulderRoll}, {angleElbowYaw}, {angleElbowRoll}")
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
        
        # motor_tolerance= 0.01
        # while True:
        # # Check if the arm is within the motor_tolerance range
        #     if (abs(self.RShoulderPitchPS.getValue()-self.RShoulderPitch.getTargetPosition()) > motor_tolerance) and (abs(self.RShoulderRollPS.getValue()-self.RShoulderRoll.getTargetPosition()) > motor_tolerance) and (abs(self.RElbowRollPS.getValue()-self.RElbowRoll.getTargetPosition()) > motor_tolerance) and (abs(self.RShoulderPitchPS.getValue()-self.RShoulderPitch.getTargetPosition()) > motor_tolerance):
        #         break
       
        
      

    def printGps(self):
        """
        Prints the current GPS coordinates of the right robot's hand.
        """        
        p = self.getRelativeCoords()
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
        """
        Calculates and returns the relative GPS coordinates of the robot's hand with respect to its body.

        Returns:
            -list: A list containing the relative coordinates [x, y, z].
        """        
        # Leer la posición global del GPS de la mano
        gps_hand_coords = self.gps.getValues()
        #print(f"Hand gps: {gps_hand_coords}")
        
        # Leer la posición del cuerpo del NAO
        gps_body_coords = self.gps_body.getValues()
        #print(f"Body gps: {gps_body_coords}")
        
        
        # Calcular la posición relativa del GPS con respecto al cuerpo
        relative_coords = [round(gps_hand_coords[i] - gps_body_coords[i], 4) for i in range(3)]
        #print("Coordenadas GPS relativas:", relative_coords)
        return relative_coords
        
    def findAndEnableDevices(self):
        """
        Initializes and enables various devices for the robot.
        """        
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
        
        self.RShoulderPitchPS= self.RShoulderPitch.getPositionSensor()
        self.RShoulderPitchPS.enable(1)
        self.RShoulderRollPS= self.RShoulderRoll.getPositionSensor()
        self.RShoulderRollPS.enable(1)
        self.RElbowYawPS= self.RElbowYaw.getPositionSensor()
        self.RElbowYawPS.enable(1)
        self.RElbowRollPS= self.RElbowRoll.getPositionSensor()
        self.RElbowRollPS.enable(1)
        
        self.maxRShoulderPitchPosition=self.RShoulderPitch.getMaxPosition();
        self.minRShoulderPitchPosition=self.RShoulderPitch.getMinPosition();
        self.maxRShoulderRollPosition=self.RShoulderRoll.getMaxPosition();
        self.minRShoulderRollPosition=self.RShoulderRoll.getMinPosition();
        self.maxRElbowYawPosition=self.RElbowYaw.getMaxPosition();
        self.minRElbowYawPosition=self.RElbowYaw.getMinPosition();
        self.maxRElbowRollPosition=self.RElbowRoll.getMaxPosition();
        self.minRElbowRollPosition=self.RElbowRoll.getMinPosition();
        print("Shoulder Pitch max:", self.maxRShoulderPitchPosition, "min :", self.minRShoulderPitchPosition);
        print("Shoulder Roll max:", self.maxRShoulderRollPosition, "min :", self.minRShoulderRollPosition);
        print("Elbow Yaw Pitch max:", self.maxRElbowYawPosition, "min :", self.minRElbowYawPosition);
        print("Elbow Roll max:", self.maxRElbowRollPosition, "min :", self.minRElbowRollPosition);
        
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        # keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def __init__(self):
        """
        Initializes the Nao robot instance and sets up its devices.
        """        
        Robot.__init__(self)
        # initialize stuff
        self.findAndEnableDevices()
    
    #función en donde se puede testear las conexiones del som 1 al som2 y viceversa    
    def hebbianTest(self, choice):
        """
        Tests the Hebbian learning connections between two self-organizing maps.

        Args:
            -choice (int): If 1, tests from SOM1 to SOM2; if 2, tests from SOM2 to SOM1.
        """        
        random.seed(10)
        max_iterations=5000
        for i in range (max_iterations):
         # Generate random angles within the specified range
            randomShoulderPitch =  round(random.uniform(self.minRShoulderPitchPosition,self.maxRShoulderPitchPosition),4)
            randomShoulderRoll = round(random.uniform(self.minRShoulderRollPosition, self.maxRShoulderRollPosition),4)
            randomElbowYaw = round(random.uniform(self.minRElbowYawPosition, self.maxRElbowYawPosition),4)
            randomElbowRoll = round(random.uniform(self.minRElbowRollPosition, self.maxRElbowRollPosition),4)
            
            motor_entry = [randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll]
            # Set the random angles using the function
            self.MoveArm(motor_entry)
            
            # Get GPS data
            gps_entry = self.getRelativeCoords()
            #print(gps_entry)
            #normalize
            motor_entry= tools.min_max_normalize_with_data(motor_entry, motor_data)
            gps_entry= tools.min_max_normalize_with_data(gps_entry, gps_data)
            #print(gps_entry)
            
            time.sleep(1)
            if choice == 1 :
                print("BMU SOM 2: ", hebbian_table.getConectionsFromSOM1(gps_entry))
            
            else:
                print("BMU SOM 1: ", hebbian_table.getConectionsFromSOM2(motor_entry))
        
    def GetPredError(self, rotation_angles, target_coordinate):
        """
        Executes a movement of the right robot's arm and calculates the predictive error.

        Args:
            -rotation_angles (list): A list containing the angles for the arm joints.
            -target_coordinate (list): The target coordinates to compare against.

        Returns:
            -float: The calculated predictive error between the current and target position.
        """        
        # Set the random angles using the function
        self.MoveArm(rotation_angles)
        # Get GPS data
        gps_entry = self.getRelativeCoords()
        time.sleep(0.002)
        # Calculate predictive error between current position and target
        pred_error = round(np.linalg.norm(np.array(target_coordinate)- np.array(gps_entry)),5)
        #print(f"$$$Target coordinate: {target_coordinate}, Gps entry: {gps_entry}, pred error: {pred_error}")
        #print("Prediction error: ", pred_error)
            
        return pred_error
    
    def getRealGpsGoal(self, rotation_angles):
        """
        Executes a movement of the right robot's arm and calculates the asociated gps relative coordinates.

        Args:
            -rotation_angles (list): A list containing the angles for the arm joints.

        Returns:
            The calculated relative gps coordinates.
        """        
        
        # Set the random angles using the function
        self.MoveArm(rotation_angles)
        #print(f"Goal rotation angles: {rotation_angles}")
        # Get GPS data
        
        gps_entry = self.getRelativeCoords()
        #print(f"Actual goal gps: {gps_entry}")
        time.sleep(0.002)
        return gps_entry

    def MoveArm(self, rotation_angles):
        """
        Moves the right robot's arm to the specified angles.

        Args:
            -rotation_angles (list): A list containing the angles for the arm joints.
        """        
        while robot.step(self.timeStep) != -1:
            self.setArmAngle(rotation_angles[0], rotation_angles[1], rotation_angles[2], rotation_angles[3])
            #time.sleep(1)
            
            if (abs(self.RShoulderPitchPS.getValue()-self.RShoulderPitch.getTargetPosition()) < motor_tolerance) and (abs(self.RShoulderRollPS.getValue()-self.RShoulderRoll.getTargetPosition()) < motor_tolerance) and (abs(self.RElbowRollPS.getValue()-self.RElbowRoll.getTargetPosition()) < motor_tolerance) and (abs(self.RElbowYawPS.getValue()-self.RElbowYaw.getTargetPosition()) < motor_tolerance):
                break
        
    def hebbianTrain(self):
        """
        Trains the Hebbian learning table using random angles based on a normal distribution for the arm joints.
        """        
        random.seed(10)
            
        max_iterations = 10000
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
        for i in range(max_iterations):
            # Generate random angles within the specified range
            randomShoulderPitch = random.gauss(mu_ShoulderPitch, sigma_ShoulderPitch)
            randomShoulderRoll = random.gauss(mu_ShoulderRoll, sigma_ShoulderRoll)
            randomElbowYaw = random.gauss(mu_ElbowYaw, sigma_ElbowYaw)
            randomElbowRoll = random.gauss(mu_ElbowRoll, sigma_ElbowRoll)
            
            # Ensure the angles are within the specified range
            randomShoulderPitch = max(min(randomShoulderPitch, self.maxRShoulderPitchPosition), self.minRShoulderPitchPosition)
            randomShoulderRoll =max(min(randomShoulderRoll, self.maxRShoulderRollPosition), self.minRShoulderRollPosition)
            randomElbowYaw = max(min(randomElbowYaw, self.maxRElbowYawPosition), self.minRElbowYawPosition)
            randomElbowRoll = max(min(randomElbowRoll, self.maxRElbowRollPosition), self.minRElbowRollPosition)

            motor_entry = [randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll]
            # Set the random angles using the function
            self.MoveArm(motor_entry)
            
            # Get GPS data
            
            gps_entry = self.getRelativeCoords()
            
            #normalize
            motor_entry= tools.min_max_normalize_with_data(motor_entry, motor_data)
            gps_entry= tools.min_max_normalize_with_data(gps_entry, gps_data)
            
            print("Iteration "+ str(i)+ "\t")
            print("Waiting for 0.002s \t")
            time.sleep(0.002)
            hebbian_table.learnUsingWinners(gps_entry, motor_entry)
    
    def testGPS(self):
        self.printGps()
        
    def run(self):
        """
        Runs the main loop to collect training samples for the robot's movements.
        
        The loop runs for a specified number of iterations or until an error occurs or the iterations are over.
        """        
        with open("motor_angles.csv", "w",newline='') as  motor_csvfile, \
             open("gps_hand.csv", "w", newline='') as gps_csvfile:
            motor_writer = csv.writer(motor_csvfile)
            gps_writer = csv.writer(gps_csvfile)
            #motor_writer.writerow(["Index", "ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll"])
            #gps_writer.writerow(["Index", "X", "Y", "Z"])
            self.LShoulderPitch.setPosition(2)
            random.seed(10)
            
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
            for i in range (max_iterations):
                try:
                    # Generate random angles within the specified range
                    randomShoulderPitch = random.gauss(mu_ShoulderPitch, sigma_ShoulderPitch)
                    randomShoulderRoll = random.gauss(mu_ShoulderRoll, sigma_ShoulderRoll)
                    randomElbowYaw = random.gauss(mu_ElbowYaw, sigma_ElbowYaw)
                    randomElbowRoll = random.gauss(mu_ElbowRoll, sigma_ElbowRoll)
        
                    
                    # Ensure the angles are within the specified range
                    randomShoulderPitch = max(min(randomShoulderPitch, self.maxRShoulderPitchPosition), self.minRShoulderPitchPosition)
                    randomShoulderRoll = max(min(randomShoulderRoll, self.maxRShoulderRollPosition), self.minRShoulderRollPosition)
                    randomElbowYaw = max(min(randomElbowYaw, self.maxRElbowYawPosition), self.minRElbowYawPosition)
                    randomElbowRoll = max(min(randomElbowRoll, self.maxRElbowRollPosition), self.minRElbowRollPosition)
                    
                    rotation_angles=[randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll]
                    # Set the random angles using the function
                    self.MoveArm(rotation_angles)
                    
                    # Get GPS data
                    gps_data = self.getRelativeCoords()
                    
                    print('----------gps----------')
                    print('position: [ x y z ] = [%f %f %f]' % (gps_data[0], gps_data[1], gps_data[2]))
                    
                    time.sleep(0.002)
                    print("Waiting for 0.002s \t")
                       
                    
                    motor_writer.writerow([i, randomShoulderPitch, randomShoulderRoll, randomElbowYaw, randomElbowRoll])
                    gps_writer.writerow([i,gps_data[0], gps_data[1], gps_data[2]])
                    
                    print(i)
                except Exception as e:
                    print(f"Error at iteration {i}: {e}")
                    break
                
     

def generateAnglesSOM():
    """
    Generates a Self-Organizing Map (SOM) for motor angles.
    """    
    global somAngles
    
    # Load data
    columns = ['Key','RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
    data = pd.read_csv('motor_angles.csv', names=columns, sep=',', engine='python', header=None)


    target = data['Key'].values

    # Remove first column 
    data = data[data.columns[1:]]
    
    normalized_data = tools.min_max_normalize(data)
    
    normalized_df = pd.DataFrame(normalized_data, columns=data.columns)
    normalized_df.to_csv('normalized_motor_angles.csv', index=False)
    print("Normalized data saved to 'normalized_motor_angles.csv'.")

    data = normalized_data.values
    #print(data)
    #print(type(data))
    # Initialization and training
    num_samples = len(data)
    # Calculate the number of neurons using the rule of thumb
    num_neurons = 5 * math.sqrt(num_samples)
    # Determine the grid size
    grid_size = int(math.ceil(math.sqrt(num_neurons)))
    grid_size=70
    somAngles = MiniSom(grid_size, grid_size, data.shape[1], sigma=3, learning_rate=.5, neighborhood_function='gaussian', random_seed=0, topology='rectangular')

    somAngles.random_weights_init(data)
    somAngles.train(data, 10000, verbose=True)  # random training
    
    print("SOM Motor trained")
    
    # saving the som
    with open('somAngles.p', 'wb') as outfile:
        pickle.dump(somAngles, outfile)
    
def generateVisualSOM():
    """
    Generates a Self-Organizing Map (SOM) for visual coordinates.
    """    
    global somVisual
    
    columns = ['Key','X', 'Y', 'Z']
    data = pd.read_csv('gps_hand.csv', names=columns, sep=',', engine='python', header=None)
    
    
    target = data['Key'].values
    
    # Remove first column 
    data = data[data.columns[1:]]
    # Data normalization
    data = tools.min_max_normalize(data)
    data = data.values
    
    num_samples = len(data)
    # Calculate the number of neurons using the rule of thumb
    num_neurons = 5 * math.sqrt(num_samples)
    # Determine the grid size
    grid_size = int(math.ceil(math.sqrt(num_neurons)))
    grid_size=70
    somVisual = MiniSom(grid_size, grid_size, data.shape[1], sigma=3, learning_rate=.5, 
                  neighborhood_function='gaussian', random_seed=0, topology='rectangular')
    
    somVisual.random_weights_init(data)
    somVisual.train(data, 10000, verbose=True)  # random training
    
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


####train SOMS
# generateAnglesSOM()
# generateVisualSOM()

#####load SOMS           
with open('somVisual.p', 'rb') as infile:
    somVisual = pickle.load(infile)

with open('somAngles.p', 'rb') as infile:
    somAngles = pickle.load(infile)




# print("Distorsión de la cuantización vectorial del SOM Visual después de la denormalización del bmu: " , tools.totalerrorindataSOM(somVisual, gps_data))
# print("Distorsión de la cuantización vectorial del SOM Motor después de la denormalización del bmu: " , tools.totalerrorindataSOM(somAngles, motor_data))

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

#hebbian_table.loadFromFile("hebbian_table_new.txt")

#robot.hebbianTest(1)




grid_size = (70, 70)  
# print("running")
# intrinsic_motivation = intrinsic.IntrinsicMotivation(somVisual, somAngles, hebbian_table, robot)
# task_dict=intrinsic_motivation.initialize_task_dictionary()    
# # print(task_dict)   
# visualize_individual_paths(task_dict, grid_size)        


exp= experimentation.Experiment(0.1, 5000, robot)
#exp.run_exp()
if os.path.exists("learnt_policies.json"):
    #exp.execute_loaded_policies("learnt_policies.json")
    exp.execute_policy_by_index("learnt_policies.json",2)
    all_policies = exp.load_all_policies_from_json("learnt_policies.json")
    # # Visualiza las políticas cargadas
    tools.visualize_loaded_policies(all_policies, grid_size)
else:
    print("No tasks were learnt in this run")



# while (1):
#     rotation_angles=[1,0,0,2]
#     robot.MoveArm(rotation_angles)
#     print(robot.RShoulderPitchPS.getValue()-robot.RShoulderPitch.getTargetPosition())
#     gps_entry = robot.getRelativeCoords()
#     print(f"GPS: {gps_entry}")

