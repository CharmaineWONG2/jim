import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from flowMC.strategy.optimization import optimization_Adam
from jimgw.population.population_likelihood import PopulationLikelihood
from jimgw.population.utils import create_model, extract_data_from_npz_files
import argparse
from jimgw.prior import UniformPrior, CombinePrior
from jimgw.transforms import BoundToUnbound
from jimgw.population.transform import NullTransform

jax.config.update("jax_enable_x64", True)

def parse_args():
    parser = argparse.ArgumentParser(description='Run population likelihood sampling.')
    parser.add_argument('--pop_model', type=str, required=True, help='Population model to use.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the NPZ data files.')
    return parser.parse_args()

def main():
    args = parse_args()
    mass1_array = extract_data_from_npz_files(args.data_dir,"m_1", num_samples=5000, random_seed=42) 

    """
    need changes for the pop_likelihood
    """
    # def pop_likelihood(pop_params ,data):
    #         model = create_model(args.pop_model)
    #         likelihood = PopulationLikelihood(mass1_array, model, pop_params)
    #         log_likelihood = likelihood.evaluate(mass1_array, pop_params)
    #         return log_likelihood
        
    model = create_model(args.pop_model)
    pop_params = ["m_min",1,2]
    pop_likelihood = PopulationLikelihood(mass1_array, model, pop_params)
 
    mass_matrix = jnp.eye(11)
    mass_matrix = mass_matrix.at[1, 1].set(1e-3)
    mass_matrix = mass_matrix.at[5, 5].set(1e-3)
    local_sampler_arg = {"step_size": mass_matrix * 3e-3}

    Adam_optimizer = optimization_Adam(n_steps=5, learning_rate=0.01, noise_level=1)
    
    """"
    The following needs changing
    """
    m_min_prior = UniformPrior(10.,80.,parameter_names = ["m_min"])
    m_max_prior = UniformPrior(10.,80.,parameter_names = ["m_max"])
    alpha_prior = UniformPrior(0.,10.,parameter_names = ["alpha"])
    prior = CombinePrior([m_min_prior, m_max_prior, alpha_prior])
    sample_transforms = [BoundToUnbound(name_mapping = [["m_min"], ["m_min_unbounded"]], original_lower_bound=10, original_upper_bound=80),
                         BoundToUnbound(name_mapping = [["m_max"], ["m_max_unbounded"]], original_lower_bound=10, original_upper_bound=80),
                         BoundToUnbound(name_mapping = [["alpha"], ["alpha_unbounded"]], original_lower_bound=0, original_upper_bound  =10)]
    name_mapping = (
    ["m_min", "m_max", "alpha"], 
    ["m_min", "m_max", "alpha"]   
    )
    likelihood_transforms = [NullTransform(name_mapping)]

    n_epochs = 2
    n_loop_training = 1
    learning_rate = 1e-4


    jim = Jim(
        pop_likelihood,
        prior,
        sample_transforms,
        likelihood_transforms ,
        n_loop_training=n_loop_training,
        n_loop_production=1,
        n_local_steps=5,
        n_global_steps=5,
        n_chains=4,
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

    jim.sample(jax.random.PRNGKey(42))
    samples =jim.get_samples()
    jim.print_summary()

if __name__ == "__main__":
    main()
