from abc import ABC, abstractmethod
import jax.numpy as jnp
from jaxtyping import Float
from dataclasses import dataclass, field
import jax

class PopulationModelBase(ABC):
    def __init__(self, parameter_names: list[str], pop_params_names: list[str] = [] , pop_param_mapping: dict = None):
        """
        parameter_names : list[str]
            A list of names for the individual event parameters of the population model.
        pop_params_names : list[str]
            A list of names for the population parameters of the population model.
        pop_param_mapping : dict    
            A dictionary to map original parameter names to new names, keyed by model instance
        """
        self.parameter_names = parameter_names
        self.pop_params_names = pop_params_names
        self.pop_param_mapping = pop_param_mapping

    def evaluate(self, pop_params: dict, data: dict) -> float:
        """
        Evaluate the likelihood for a given set of parameters.
        """
        raise NotImplementedError

class CombinePopulationModel(PopulationModelBase):
    """
    To compose two independent population models together.
    """
    
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
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}
        
        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.pop_param_mapping.get("x_min", "x_min"),
            "x_max": self.pop_param_mapping.get("x_max", "x_max"),
            "alpha": self.pop_param_mapping.get("alpha", "alpha")
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
    
class BrokenPowerLawModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}
        
        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.pop_param_mapping.get("x_min", "x_min"),
            "x_max": self.pop_param_mapping.get("x_max", "x_max"),
            "alpha1": self.pop_param_mapping.get("alpha1", "alpha1"),
            "alpha2": self.pop_param_mapping.get("alpha2", "alpha2"),
            "b": self.pop_param_mapping.get("b", "b"),
            "delta": self.pop_param_mapping.get("delta", "delta")
        }
        
    def smoothing_function(x, x_min, delta):
        """
        Apply a smoothing function to the input array x based on the parameters x_min and delta.

        Parameters:
        - x (jnp.array): Input array.
        - x_min (float): Minimum value for the smoothing function.
        - delta (float): Smoothing parameter.

        Returns:
        - jnp.array: Smoothed array.
        """
        case1 = (x < x_min)
        case2 = (x >= x_min) & (x < x_min + delta)
        case3 = (x >= x_min + delta)

        result = jnp.where(case1, 0, jnp.where(case2, 1 / (jnp.exp(delta / (x - x_min) + delta / (x - x_min - delta)) + 1),1))
        return result
    
    def broken_power_law(self, x, x_min, x_max, alpha1, alpha2, b, delta):
        x_break = x_min + b * (x_max - x_min)
        case1 = (x > x_min) & (x < x_break)
        case2 = (x >= x_break) & (x < x_max)
        case3 = (x < x_min) | (x > x_max)
        
        result = jnp.where(case1, x **(-alpha1) * self.smoothing_function(x, x_min, delta), jnp.where(case2, x **(-alpha2) * self.smoothing_function(x, x_min, delta, 0)))
  

    def evaluate(self, pop_params: dict[str, float], data: dict) -> float:
        """
        Evaluate the model using the provided population parameters and data.
        
        pop_params: Dictionary of population parameters
        data: Dictionary of data to evaluate the model on
        
        return: Evaluation result
        """
        x_min = pop_params[self.mapped_params["x_min"]]
        x_max = pop_params[self.mapped_params["x_max"]]
        alpha1 = pop_params[self.mapped_params["alpha1"]]
        alpha2 = pop_params[self.mapped_params["alpha2"]]
        b = pop_params[self.mapped_params["b"]]
        delta = pop_params[self.mapped_params["delta"]]
        
        return self.broken_power_law(data, x_min, x_max, alpha1, alpha2, b, delta)
    
class PowerPeakModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}
        
        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.pop_param_mapping.get("x_min", "x_min"),
            "x_max": self.pop_param_mapping.get("x_max", "x_max"),
            "alpha": self.pop_param_mapping.get("alpha", "alpha"),
            "lamda_peak": self.pop_param_mapping.get("lamda_peak", "lamda_peak"),
            "mu": self.pop_param_mapping.get("mu", "mu"),
            "sigma": self.pop_param_mapping.get("sigma", "sigma"),
            "delta": self.pop_param_mapping.get("delta", "delta")
        }
    
    def smoothing_function(x, x_min, delta):
        """
        Apply a smoothing function to the input array x based on the parameters x_min and delta.

        Parameters:
        - x (jnp.array): Input array.
        - x_min (float): Minimum value for the smoothing function.
        - delta (float): Smoothing parameter.

        Returns:
        - jnp.array: Smoothed array.
        """
        case1 = (x < x_min)
        case2 = (x >= x_min) & (x < x_min + delta)
        case3 = (x >= x_min + delta)

        result = jnp.where(case1, 0, jnp.where(case2, 1 / (jnp.exp(delta / (x - x_min) + delta / (x - x_min - delta)) + 1),1))
        return result
    
    def normalized_gaussian(self, x, mu, sigma):
        return 1/(sigma * jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5 * ((x - mu) / sigma)**2)
    
    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x[self.param] >= x_min) & (x[self.param] <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x[self.param])  
        pdf = jnp.where(valid_indices, C / (x[self.param] ** alpha), pdf)
        return pdf
    
    def power_peak(self, x, x_min, x_max, alpha, lamda_peak, mu, sigma, delta):
        return((1-lamda_peak) * self.truncated_power_law(x, x_min, x_max, alpha) + lamda_peak * self.normalized_gaussian(x, mu, sigma) ) * self.smoothing_function(x, x_min, delta)

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
        lamda_peak = pop_params[self.mapped_params["lamda_peak"]]
        mu = pop_params[self.mapped_params["mu"]]
        sigma = pop_params[self.mapped_params["sigma"]]
        delta = pop_params[self.mapped_params["delta"]]
        
        return self.power_peak(data, x_min, x_max, alpha, lamda_peak, mu, sigma, delta)   
    
class M1_q_TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}
        
        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.pop_param_mapping.get("x_min", "x_min"),
            "x_max": self.pop_param_mapping.get("x_max", "x_max"),
            "alpha": self.pop_param_mapping.get("alpha", "alpha"),
            "beta": self.pop_param_mapping.get("beta", "beta")
        }
    
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
        x_min = pop_params[self.mapped_params["x_min"]]
        x_max = pop_params[self.mapped_params["x_max"]]
        alpha = pop_params[self.mapped_params["alpha"]]
        beta = pop_params[self.mapped_params["beta"]]
        
        return self.truncated_power_law_2d(data, x_min, x_max, alpha, beta)
    
class M1_BrokenPower_q_TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
    super().__init__(parameter_names=parameter_names)
    self.param = parameter_names[0]
    self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}
    
    # Apply the parameter mapping
    self.mapped_params = {
        "x_min": self.pop_param_mapping.get("x_min", "x_min"),
        "x_max": self.pop_param_mapping.get("x_max", "x_max"),
        "alpha1": self.pop_param_mapping.get("alpha1", "alpha1"),
        "alpha2": self.pop_param_mapping.get("alpha2", "alpha2"),
        "b": self.pop_param_mapping.get("b", "b"),
        "delta": self.pop_param_mapping.get("delta", "delta")
        "beta": self.pop_param_mapping.get("beta", "beta")
    }
    
    def smoothing_function(x, x_min, delta):
        """
        Apply a smoothing function to the input array x based on the parameters x_min and delta.

        Parameters:
        - x (jnp.array): Input array.
        - x_min (float): Minimum value for the smoothing function.
        - delta (float): Smoothing parameter.

        Returns:
        - jnp.array: Smoothed array.
        """
        case1 = (x < x_min)
        case2 = (x >= x_min) & (x < x_min + delta)
        case3 = (x >= x_min + delta)

        result = jnp.where(case1, 0, jnp.where(case2, 1 / (jnp.exp(delta / (x - x_min) + delta / (x - x_min - delta)) + 1),1))
        return result
    
    def broken_power_law(self, x, x_min, x_max, alpha1, alpha2, b, delta):
        x_break = x_min + b * (x_max - x_min)
        case1 = (x > x_min) & (x < x_break)
        case2 = (x >= x_break) & (x < x_max)
        case3 = (x < x_min) | (x > x_max)
        result = jnp.where(case1, x **(-alpha1) * self.smoothing_function(x, x_min, delta), jnp.where(case2, x **(-alpha2) * self.smoothing_function(x, x_min, delta, 0)))

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x)  
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)
        return pdf

    def broken_power_truncated_power(self, x, x_min, x_max, alpha1, alpha2, b, delta,beta):
        broken_power = self.broken_power_law(x["m_1"], x_min, x_max, alpha1, alpha2, b, delta)
        q_truncated_power = self.truncated_power_law((x["m_2"] / x["m_1"]), (x_min/x["m_1"]), 1, beta) * self.smoothing_function(x["m_2"], x_min, delta )
        return broken_power * q_truncated_power

    def evaluate(self, pop_params: dict, data: dict) -> float:
        x_min = pop_params[self.mapped_params["x_min"]]
        x_max = pop_params[self.mapped_params["x_max"]]
        alpha1 = pop_params[self.mapped_params["alpha1"]]
        alpha2 = pop_params[self.mapped_params["alpha2"]]
        b = pop_params[self.mapped_params["b"]]
        delta = pop_params[self.mapped_params["delta"]]
        beta = pop_params[self.mapped_params["beta"]]
   
        return self.broken_power_truncated_power(data, x_min, x_max, alpha1, alpha2, b, delta,beta )
        
class M1_PowerPeak_q_TruncatedPowerLawModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}

        # Apply the parameter mapping
        self.mapped_params = {
            "x_min": self.pop_param_mapping.get("x_min", "x_min"),
            "x_max": self.pop_param_mapping.get("x_max", "x_max"),
            "alpha": self.pop_param_mapping.get("alpha", "alpha"),
            "beta": self.pop_param_mapping.get("beta", "beta"),
            "lambda_peak": self.pop_param_mapping.get("lambda_peak", "lambda_peak"),
            "mu": self.pop_param_mapping.get("mu", "mu"),
            "sigma": self.pop_param_mapping.get("sigma", "sigma"),
            "delta": self.pop_param_mapping.get("delta", "delta")
        }

    def smoothing_function(self,x, x_min, delta):
        """
        Apply a smoothing function to the input array x based on the parameters x_min and delta.

        Parameters:
        - x (jnp.array): Input array.
        - x_min (float): Minimum value for the smoothing function.
        - delta (float): Smoothing parameter.

        Returns:
        - jnp.array: Smoothed array.
        """
        case1 = (x < x_min)
        case2 = (x >= x_min) & (x < x_min + delta)
        case3 = (x >= x_min + delta)

        result = jnp.where(case1, 0, jnp.where(case2, 1 / (jnp.exp(delta / (x - x_min) + delta / (x - x_min - delta)) + 1),1))
        return result

    def normalized_gaussian(self, x, mu, sigma):
        return 1/(sigma * jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5 * ((x - mu) / sigma)**2)

    def truncated_power_law(self, x, x_min, x_max, alpha):
        valid_indices = (x >= x_min) & (x <= x_max)
        C = (1 - alpha) / (x_max**(1 - alpha) - x_min**(1 - alpha))
        pdf = jnp.zeros_like(x)  
        pdf = jnp.where(valid_indices, C / (x ** alpha), pdf)
        return pdf

    def power_peak(self, x, x_min, x_max, alpha, lamda_peak, mu, sigma, delta):
        return((1-lamda_peak) * self.truncated_power_law(x, x_min, x_max, alpha) + lamda_peak * self.normalized_gaussian(x, mu, sigma) ) * self.smoothing_function(x, x_min, delta)

    def power_peak_truncated_power(self, x, x_min, x_max, alpha, lamda_peak, mu, sigma, delta, beta):
        m1_power_peak = self.power_peak(x["m_1"], x_min, x_max, alpha, lamda_peak, mu, sigma, delta)
        q_truncated_power = self.truncated_power_law((x["m_2"] / x["m_1"]), (x_min/x["m_1"]), 1, beta) * self.smoothing_function(x["m_2"], x_min, delta )
        return m1_power_peak * q_truncated_power

    def evaluate(self, pop_params: dict, data: dict) -> float:
        x_min = pop_params[self.mapped_params["x_min"]]
        x_max = pop_params[self.mapped_params["x_max"]]
        alpha = pop_params[self.mapped_params["alpha"]]
        beta = pop_params[self.mapped_params["beta"]]
        lambda_peak = pop_params[self.mapped_params["lambda_peak"]]
        mu = pop_params[self.mapped_params["mu"]]
        sigma = pop_params[self.mapped_params["sigma"]]
        delta = pop_params[self.mapped_params["delta"]]
        
        return self.power_peak_truncated_power(data, x_min, x_max, alpha, lambda_peak, mu, sigma, delta, beta)


class DefaultSpinModel(PopulationModelBase):
    def __init__(self, parameter_names: list[str], pop_param_mapping: dict[str, str] = None):
        super().__init__(parameter_names=parameter_names)
        self.param = parameter_names[0]
        self.pop_param_mapping = pop_param_mapping if pop_param_mapping else {}

        self.mapped_params = {
            "alpha": self.pop_param_mapping.get("alpha", "alpha"),
            "beta": self.pop_param_mapping.get("beta", "beta"),
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
    




        


    
    
    
 

