import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

#denormalize a vector given dataset
def denormalize_vector(normalized_vector, data):
    """
    Takes a normalized vector and a dataset to return a denormalized version of the vector. 
    Args:
        -normalized_vector: A list or array representing a vector with values normalized between 0 and 1.
        -data: A dataset used to retrieve the minimum and maximum values for each feature, which is necessary for the denormalization process.

    Returns:
        A denormalized vector, where each element corresponds to its original scale based on the dataset's feature ranges.
    """    
    # Calculate min and max for each feature in the dataset
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    
    # Denormalize the vector
    denormalized_vector = []
    for i in range(len(normalized_vector)):
        denormalized_value = normalized_vector[i] * (max_values[i] - min_values[i]) + min_values[i]
        denormalized_vector.append(denormalized_value)
        
    denormalized_vector=np.round(denormalized_vector,4)
    return denormalized_vector
    
def min_max_normalize(x):
    """
    Performs min-max normalization on a dataset or vector.

    Args:
        -x: A dataset to be normalized.

    Returns:
        A normalized version of x, where each value is scaled between 0 and 1.
    """    
    #Get the min value of the dataset
    min_val = np.min(x, axis=0)
    #Get the max value of the dataset
    max_val = np.max(x, axis=0)
    #min_max normalization
    normalized_x = (x - min_val) / (max_val - min_val)
    return normalized_x

#normalize one value given a data set
def min_max_normalize_with_data(vector, data):
    """
    Normalizes a single vector based on the min and max values of a dataset.

    Args:
        -vector: A vector that is to be normalized.
        -data: A dataset used to calculate the min and max values for each component of the vector.

    Returns:
        A normalized version of the input vector.
    """    
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


def visualize_loaded_policies(all_policies, grid_size):
    folder_path = "learnt_policies"
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    idx = 0
    for policy in all_policies:
        coordinates = policy["Coordinates"]
        set_pairs = policy["SetPairs"]  

        start, end = coordinates
        set_pairs.append(end)
        path_coords = np.array(set_pairs)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, grid_size[1] - 0.5)
        ax.set_ylim(-0.5, grid_size[0] - 0.5)  

        ax.set_xticks(np.arange(grid_size[1]))
        ax.set_yticks(np.arange(grid_size[0]))
        
        ax.set_xticks(np.arange(-0.5, grid_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size[0], 1), minor=True)
        
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(
            left=True, bottom=True, labelleft=True, labelbottom=True, 
            labelsize=5
        )
        plt.setp(ax.get_xticklabels(), rotation=90)

        ax.plot(
            path_coords[:, 0], path_coords[:, 1], 
            marker="o", color="blue", markersize=3,  
            label=f"Policy Path"
        )  

        for i, (x, y) in enumerate(path_coords):
            ax.text(
                x, y, str(i), color="red", ha="center", va="center", 
                fontweight="bold", fontsize=7  
            )

        ax.set_title(f"Policy: Start: {start}, End: {end}")
        ax.legend(loc="upper right")

        file_name = os.path.join(folder_path, f"policy_{idx}.png")
        plt.savefig(file_name)
        plt.close()

        print(f"Saved: {file_name}")

        idx += 1


def totalerrorindataSOM(som, train_data):
    total_error = 0
    train_data_normalized = min_max_normalize(train_data)
    for data, original in zip(train_data_normalized, train_data):
        
        weights = som.get_weights()
        
        # Encontrar el BMU
        bmu = som.winner(data)  
        bmu_position = weights[bmu[0], bmu[1]]
        
        # Desnormalizar el vector del BMU
        bmu_position_denormalized = denormalize_vector(bmu_position, train_data)
        
        # Calcular la distancia Euclidiana entre la posición original y la del BMU
        distance = np.linalg.norm(original - bmu_position_denormalized)
        total_error += distance

    average_error = total_error / len(train_data)
    return average_error
    
