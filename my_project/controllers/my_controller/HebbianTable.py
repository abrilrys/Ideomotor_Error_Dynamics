import math 
import numpy as np

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
