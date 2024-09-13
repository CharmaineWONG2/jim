import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from flowMC.strategy.optimization import optimization_Adam
from jimgw.population.population_likelihood import PopulationLikelihood
from jimgw.population.utils import create_model
import argparse
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.transforms import BoundToUnbound
from jimgw.population.transform import NullTransform
from jimgw.population.population_model import TruncatedPowerLawModel, DefaultSpinModel, CombinePopulationModel

jax.config.update("jax_enable_x64", True)

def parse_args():
    parser = argparse.ArgumentParser(description='Run population likelihood sampling.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the NPZ data files.')
    return parser.parse_args()

def main():
    args = parse_args()
    truncated_power_law_model = TruncatedPowerLawModel(parameter_names=['m_1'], param_mapping ={
        'x_min': 'm_min',
        'x_max': 'm_max',
        'alpha': 'alpha_m1'
    })
    default_spin_model = DefaultSpinModel(parameter_names=['s1_z'], param_mapping={
        'alpha': 'alpha_s1_z',
        'beta': 'beta_s1_z'
    }
    )
    
    model = CombinePopulationModel([truncated_power_law_model, default_spin_model])

    # model = truncated_power_law_model
    
    # pop_likelihood = PopulationLikelihood(args.data_dir, ["m_1", "s1_z"], 5000, model)
    pop_likelihood = PopulationLikelihood(args.data_dir, 5000, model)
    local_sampler_arg = {"step_size": 3e-3}

    Adam_optimizer = optimization_Adam(n_steps=5, learning_rate=0.01, noise_level=1)

    # Define the prior distributions
    m_min_prior = UniformPrior(10., 80., parameter_names=["m_min"])
    m_max_prior = UniformPrior(10., 80., parameter_names=["m_max"])
    alpha_m1_prior = UniformPrior(0., 10., parameter_names=["alpha_m1"])
    alpha_s1_z_prior = UniformPrior(0., 10., parameter_names=["alpha_s1_z"])
    beta_s1_z_prior = UniformPrior(0., 10., parameter_names=["beta_s1_z"])
    

    
    prior = CombinePrior([m_min_prior, m_max_prior, alpha_m1_prior, alpha_s1_z_prior, beta_s1_z_prior])
    # prior = CombinePrior([m_min_prior, m_max_prior, alpha_m1_prior])
    
    
    # Define sample transformations
    sample_transforms = [
        BoundToUnbound(name_mapping=[["m_min"], ["m_min_unbounded"]], original_lower_bound=10, original_upper_bound=80),
        BoundToUnbound(name_mapping=[["m_max"], ["m_max_unbounded"]], original_lower_bound=10, original_upper_bound=80),
        BoundToUnbound(name_mapping=[["alpha_m1"], ["alpha_m1_unbounded"]], original_lower_bound=0, original_upper_bound=10),
        BoundToUnbound(name_mapping=[["alpha_s1_z"], ["alpha_s1_z_unbounded"]], original_lower_bound=0, original_upper_bound=10),
        BoundToUnbound(name_mapping=[["beta_s1_z"], ["beta_s1_z_unbounded"]], original_lower_bound=0, original_upper_bound=10),

    ]
    
    name_mapping = (["m_min", "m_max", "alpha_m1", "alpha_s1_z","beta_s1_z"], ["m_min", "m_max", "alpha_m1", "alpha_s1_z","beta_s1_z"])   
    # name_mapping = (["m_min", "m_max", "alpha_m1"], ["m_min", "m_max", "alpha_m1"])
    likelihood_transforms = [NullTransform(name_mapping)]

    n_epochs = 5
    n_loop_training = 1
    learning_rate = 1e-4

    # Initialize the Jim object with the likelihood and prior
    jim = Jim(
        pop_likelihood,
        prior,
        sample_transforms,
        likelihood_transforms,
        n_loop_training=n_loop_training,
        n_loop_production=10,
        n_local_steps=10,
        n_global_steps=10,
        n_chains=10,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        n_max_examples=30,
        n_flow_samples=100,
        momentum=0.9,
        batch_size=100,
        use_global=True,
        train_thinning=1,
        output_thinning=1,
        local_sampler_arg=local_sampler_arg,
        strategies=[Adam_optimizer, "default"],
    )

    # Run the sampling
    jim.sample(jax.random.PRNGKey(42))
    
    # Get and print the samples
    samples = jim.get_samples()
    print(samples)
    
    # Print summary of the results
    jim.print_summary()

if __name__ == "__main__":
    main()