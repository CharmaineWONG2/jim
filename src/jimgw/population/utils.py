import importlib
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jimgw.single_event.utils import Mc_eta_to_m1_m2
import glob 

def create_model(model_name):
    try:
        module = importlib.import_module('population_model')

        # Check if model_name is a string
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")

        model_class = getattr(module, model_name)
        return model_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import model '{model_name}': {str(e)}")


import glob
import numpy as np
import jax
import jax.numpy as jnp

def extract_data_from_npz_files(data_dir, column_names, num_samples=50, random_seed=42):
    """
    Extracts specified column data from the given .npz files.

    Parameters:
    - data_dir (str): The directory containing all the .npz files.
    - column_names (list of str): The names of the columns to extract from the DataFrame.
    - num_samples (int): Number of samples to extract from each file.
    - random_seed (int): Seed for random number generation.

    Returns:
    - jnp.array: Stacked array of extracted data for each column.
    """
    
    npz_files = glob.glob(f"{data_dir}/*.npz")
    key = jax.random.PRNGKey(random_seed)
    result_dict = {column: [] for column in column_names}

    for npz_file in npz_files:
        print(f"Loading file: {npz_file}")

        with np.load(npz_file, allow_pickle=True) as data:
            data_dict = data['arr_0'].item() 
            for column_name in column_names:
                if column_name not in data_dict:
                    raise ValueError(f"Column '{column_name}' not found in the data.")
                
                extracted_data = data_dict[column_name].reshape(-1,)

                if isinstance(extracted_data, np.ndarray):
                    extracted_data = jax.device_put(extracted_data) 

                key, subkey = jax.random.split(key)
                sample_indices = jax.random.choice(subkey, extracted_data.shape[0], shape=(num_samples,), replace=True)

                sampled_data = extracted_data[sample_indices]
                result_dict[column_name].append(sampled_data)
    
    # Stack the arrays for each column
    stacked_arrays = {column: jnp.stack(data) for column, data in result_dict.items()}
    
    return stacked_arrays
    return stacked_array

def compute_mass_ratio(masses: dict) -> float:
    """
    Compute the mass ratio given a dictionary of primary and secondary masses.

    masses: dict containing 'm_1' and 'm_2' 
    return: mass ratio q (m_2 / m_1)
    """
    m_1 = masses['m_1']
    m_2 = masses['m_2']
    mass_ratio = m_2 / m_1
    return mass_ratio