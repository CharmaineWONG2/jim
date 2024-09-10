from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float

class PopulationModelBase(ABC):
    @abstractmethod
    def __init__(self, *params):
        self.params = params  
        
    @abstractmethod
    def evaluate(self, pop_params: dict, data: dict) -> Float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError


class TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self):
        super().__init__()  

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x)  
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)
        return pdf

    def evaluate(self, pop_params: dict[str, Float], data: dict) -> Float:
        return self.truncated_power_law(data, pop_params["m_min"], pop_params["m_max"],pop_params["alpha"])
    
class PrimaryMassMassRatioTruncatedPowerLawModel(PopulationModelBase):
    def __init__(self):
        super().__init__()  
    
    def truncated_power_law_2d(self, x, x_min, x_max, alpha, beta):
        """
        x: dict of primary mass and mass ratio
        x_min: minimum value of primary mass
        x_max: maximum value of primary mass
        alpha: power law index for primary mass
        beta: power law index for mass ratio
        """
        power_m1 = truncated_power_law(x["m_1"], x_min, x_max, alpha)
        power_q = truncated_power_law(x["q"], x_min/x["m_1"], 1, beta)
        pdf = power_m1 * power_q
        return pdf

    def evaluate(self, pop_params: dict[str, Float], data: dict) -> Float:
        return self.truncated_power_law_2d(data, pop_params["m_min"], pop_params["m_max"],pop_params["alpha"],pop_params["beta"])
    
    
        

    
    
    
 

