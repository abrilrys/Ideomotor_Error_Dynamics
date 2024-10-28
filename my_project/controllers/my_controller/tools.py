import numpy as np

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
    
