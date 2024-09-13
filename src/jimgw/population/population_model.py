from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float
from dataclasses import dataclass, field
import jax

class PopulationModelBase(ABC):
    def __init__(self, parameter_names: list[str], pop_params_names: list[str] = [] , param_mapping: dict = None):
        """
        parameter_names : list[str]
            A list of names for the individual event parameters of the population model.
        pop_params_names : list[str]
            A list of names for the population parameters of the population model.
        param_mapping : dict    
            A dictionary to map original parameter names to new names, keyed by model instance
        """
        self.parameter_names = parameter_names
        self.pop_params_names = pop_params_names
        self.param_mapping = param_mapping

    def evaluate(self, pop_params: dict, data: dict) -> float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError

class CombinePopulationModel(PopulationModelBase):
    
    base_population_models: list[PopulationModelBase] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Combine(population_models={self.base_population_models}, parameter_names={self.parameter_names})"
        )
    
    def __init__(self, population_models: list[PopulationModelBase]):
        super().__init__(parameter_names=[], pop_params_names=[])
        self.base_population_models = population_models
        
    def evaluate(self, pop_params: dict, data: dict) -> float:
        """
        Evaluate the combined population model.

        pop_params: Dictionary of population parameters
        data: Dictionary of data to evaluate the model on

        return: Combined evaluation result
        """
        combined_result = 1.0
        for model in self.base_population_models:
            result = model.evaluate(pop_params, data)
            combined_result *= result  
        return combined_result

class TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.param_mapping = param_mapping if param_mapping else {}
        
        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.param_mapping.get("x_min", "x_min"),
            "x_max": self.param_mapping.get("x_max", "x_max"),
            "alpha": self.param_mapping.get("alpha", "alpha")
        }

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x[self.param] >= x_min) & (x[self.param] <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x[self.param])  
        pdf = jnp.where(valid_indices, C / (x[self.param] ** alpha), pdf)
        return pdf

    def evaluate(self, pop_params: dict[str, float], data: dict) -> float:
        """
        Evaluate the model using the provided population parameters and data.
        
        pop_params: Dictionary of population parameters
        data: Dictionary of data to evaluate the model on
        
        return: Evaluation result
        """
        x_min = pop_params[self.mapped_params["x_min"]]
        x_max = pop_params[self.mapped_params["x_max"]]
        alpha = pop_params[self.mapped_params["alpha"]]
        
        return self.truncated_power_law(data, x_min, x_max, alpha)

# May be composed by the CombinePopulationModel
class PrimaryMassMassRatioTruncatedPowerLawModel(PopulationModelBase):
    def __init__(self):
        super().__init__()

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x)
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)
        return pdf

    def truncated_power_law_2d(self, x, x_min, x_max, alpha, beta):
        """
        x: dict of primary mass and mass ratio
        x_min: minimum value of primary mass
        x_max: maximum value of primary mass
        alpha: power law index for primary mass
        beta: power law index for mass ratio
        """
        power_m1 = self.truncated_power_law(x["m_1"], x_min, x_max, alpha)
        power_q = self.truncated_power_law((x["m_2"] / x["m_1"]), (x_min/x["m_1"]), 1, beta)  
        pdf = power_m1 * power_q
        return pdf

    def evaluate(self, pop_params: dict, data: dict) -> float:
        return self.truncated_power_law_2d(data, pop_params["m_min"], pop_params["m_max"], pop_params["alpha"], pop_params["beta"])
    


class DefaultSpinModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.param_mapping = param_mapping if param_mapping else {}

        self.mapped_params = {
            "alpha": self.param_mapping.get("alpha", "alpha"),
            "beta": self.param_mapping.get("beta", "beta"),
        }
        
    def beta_distribution(self, x, alpha, beta):
        return jax.scipy.stats.beta.pdf(x[self.param], alpha, beta)
    
    def evaluate(self, pop_params: dict, data: dict) -> float:
        """
        Evaluate the model using the provided population parameters and data.
        
        pop_params: Dictionary of population parameters
        data: Dictionary of data to evaluate the model on
        
        return: Evaluation result
        """
        alpha = pop_params[self.mapped_params["alpha"]]
        beta = pop_params[self.mapped_params["beta"]]
        
        return self.beta_distribution(data, alpha, beta)
    




        


    
    
    
 

