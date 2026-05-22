#!/usr/bin/python
"""
XCS221 Homework 6: Bayesian Networks
"""

from util import (
    BayesianNetwork, BayesianNode, init_zero_conditional_probability_tables,
    normalize_counts, load_annotation_csv, plot_annotator_cpts, plot_label_cpt
)
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

############################################################
# Problem 2a: Converting Phylogenetic Tree to Bayesian Network

def initialize_phylogenetic_tree(mutation_rate: float, genome_length: int=1) -> BayesianNetwork:
    """
    Initialize the phylogenetic tree as a Bayesian network using the BayesianNode and
    Bayesian Network classes in util.py.
    """
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1].")
    pass
    # ### START CODE HERE ###
    mutation_cpt = np.eye(4) * (1 - mutation_rate) + (1 - np.eye(4)) * (mutation_rate / 3)

    thomas_bayus = BayesianNode(
        name='Thomas bayus',
        domain=['A', 'C', 'T', 'G'],
        conditional_prob_table=np.array([[0.25, 0.25, 0.25, 0.25]])
    )
    humblus_studentus = BayesianNode(
        name='Humblus studentus',
        domain=['A', 'C', 'T', 'G'],
        parents=[thomas_bayus],
        conditional_prob_table=mutation_cpt.copy()
    )
    aryamus_bayus = BayesianNode(
        name='Aryamus bayus',
        domain=['A', 'C', 'T', 'G'],
        parents=[thomas_bayus],
        conditional_prob_table=mutation_cpt.copy()
    )
    kenius_bayus = BayesianNode(
        name='Kenius bayus',
        domain=['A', 'C', 'T', 'G'],
        parents=[aryamus_bayus],
        conditional_prob_table=mutation_cpt.copy()
    )
    # ### END CODE HERE ###
    network = BayesianNetwork([aryamus_bayus, humblus_studentus, thomas_bayus, kenius_bayus], batch_size=genome_length)
    return network

############################################################
# Problem 2b: Sampling from Bayesian Networks

def forward_sampling(network: BayesianNetwork) -> Dict[str, str]:
    """
    Sample a single observation from the given Bayesian network.

    Use the topological ordering of variables in network.order and sample each
    variable according to its conditional probability distribution given the values
    of its parents (which have already been sampled).

    Args:
        network: A BayesianNetwork object containing nodes to sample from

    Returns:
        A dictionary mapping variable names to their sampled values
    """
    samples: Dict[str, List[str]] = {node.name: [] for node in network.order}

    for idx in range(network.batch_size):
        assignment: Dict[str, str] = {}
        pass
        # ### START CODE HERE ###
        for node in network.order:
            if not node.parents:
                probs = node.conditional_prob_table[idx]
            else:
                parent_indices = tuple(p.domain.index(assignment[p.name]) for p in node.parents)
                probs = node.conditional_prob_table[parent_indices]
            sampled_idx = np.random.choice(len(node.domain), p=probs)
            sampled_value = node.domain[sampled_idx]
            assignment[node.name] = sampled_value
            samples[node.name].append(sampled_value)
        # ### END CODE HERE ###

    return samples

############################################################
# Problem 2c: Computing Joint Probability

def compute_joint_probability(
    network: BayesianNetwork,
    assignment: Dict[str, List[str]],
    batch_indices: Optional[List[int]] = None
) -> float:
    """
    Compute the joint probability of a given assignment to all variables in the network.

    Args:
        network: A BayesianNetwork object
        assignment: Dictionary mapping variable names to their assigned values
        batch_indices: Optional list of batch indices to compute the joint probability for
            (default is [0, 1, ..., batch_size-1], so all the indices)

    Returns:
        The joint probability as a float
    """
    pass
    # ### START CODE HERE ###
    if batch_indices is None:
        batch_indices = list(range(network.batch_size))

    joint_prob = 1.0
    for list_pos, abs_idx in enumerate(batch_indices):
        assignment_i = {k: v[list_pos] for k, v in assignment.items()}
        for node in network.order:
            value = assignment_i[node.name]
            value_idx = node.domain.index(value)
            if not node.parents:
                prob = float(node.conditional_prob_table[abs_idx, value_idx])
            else:
                parent_values = {p.name: assignment_i[p.name] for p in node.parents}
                prob = float(node.get_probability(value, parent_values))
            joint_prob *= prob

    return joint_prob
    # ### END CODE HERE ###

# ############################################################
# # Problem 2d: Test forward sampling

def test_forward_sampling():
    np.random.seed(123)
    random.seed(123)
    pass
    # ### START CODE HERE ###
    network = initialize_phylogenetic_tree(mutation_rate=0.1, genome_length=10)
    sample = forward_sampling(network)
    joint_probability = compute_joint_probability(network, sample)
    # ### END CODE HERE ###
    print(sample)
    print(f"{joint_probability:.10%}")

# Uncomment to test forward sampling
# test_forward_sampling()

############################################################
# Problem 2e: Rejection Sampling

def rejection_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_samples: int
) -> Dict[Any, float]:
    """
    Use rejection sampling to estimate the likelihoods for each outcome in the 
    target variable conditioned on the given assignments (a dictionary mapping
    variable names to their assigned values), i.e. P(target_variable | conditioned_on_assignments).

    Args:
        network: A BayesianNetwork object
        target_variable: The name of the variable to estimate the likelihoods for
        conditioned_on_assignments: A dictionary mapping variable names to their assigned values
        num_samples: The number of samples to draw

    Returns:
        A dictionary mapping outcomes to their likelihoods, conditioned on the given assignments.
    """
    pass
    # ### START CODE HERE ###
    counts = defaultdict(int)
    total_accepted = 0

    for _ in range(num_samples):
        sample = forward_sampling(network)
        accept = all(
            sample[var][i] == val
            for var, vals in conditioned_on_assignments.items()
            for i, val in enumerate(vals)
        )
        if accept:
            target_val = tuple(sample[target_variable])
            counts[target_val] += 1
            total_accepted += 1

    if total_accepted == 0:
        return {}

    return {val: cnt / total_accepted for val, cnt in counts.items()}
    # ### END CODE HERE ###

############################################################
# Problem 2f: Gibbs Sampling

def gibbs_sampling(
    network: BayesianNetwork,
    target_variable: str,
    conditioned_on_assignments: Dict[str, List[str]],
    num_iterations: int,
    initial_state: Dict[str, List[str]] = None
) -> Dict[Any, float]:
    """
    Estimate P(target_variable | conditioned_on_assignments) via Gibbs sampling.

    - Initialization: random ancestral sample (forward_sampling)
    - Use compute_joint_probability to score single-variable proposals
      (simple and robust for small networks)
    - Record the Thomas bayus genome after each full sweep
    """
    # random initialization, then set evidence variables
    state = forward_sampling(network) if initial_state is None else initial_state
    for evidence_var, evidence_val in conditioned_on_assignments.items():
        state[evidence_var] = evidence_val

    # resample all non-evidence vars each sweep
    resample_nodes = [n for n in network.order if n.name not in conditioned_on_assignments]
    counts = defaultdict(int)

    for _ in range(num_iterations):
        pass
        # ### START CODE HERE ###
        for node in resample_nodes:
            for idx in range(network.batch_size):
                state_slice = {k: [v[idx]] for k, v in state.items()}
                probs = []
                for val in node.domain:
                    state_slice[node.name] = [val]
                    prob = compute_joint_probability(network, state_slice, batch_indices=[idx])
                    probs.append(prob)
                total = sum(probs)
                if total == 0:
                    continue
                norm_probs = [p / total for p in probs]
                sampled_idx = np.random.choice(len(node.domain), p=norm_probs)
                state[node.name][idx] = node.domain[sampled_idx]
        target_val = tuple(state[target_variable])
        counts[target_val] += 1
        # ### END CODE HERE ###
    total_samples = sum(counts.values())
    return {val: counts[val] / total_samples for val in counts.keys()}

def test_gibbs_vs_rejection(
    num_steps: int=10000,
    seed: int=0,
    mutation_rate: float=0.1,
    genome_length: int=4,
):
    def exact_inference():
        same = 1.0 - mutation_rate
        diff = mutation_rate / 3.0
        p_a = same * same + 3 * diff * diff
        p_not_a = diff * same + same * diff + 2 * diff * diff
        expected = (p_a ** 3) * p_not_a
        return expected
    np.random.seed(seed)
    random.seed(seed)
    network = initialize_phylogenetic_tree(mutation_rate=mutation_rate, genome_length=genome_length)
    kenius_bayus_genome = ['A'] * genome_length

    vals = gibbs_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_iterations=num_steps)
    gibbs_p_aaac = vals.get(('A', 'A', 'A', 'C'), 0.0)

    rejection_vals = rejection_sampling(
        network, 'Thomas bayus', {'Kenius bayus': kenius_bayus_genome}, num_samples=num_steps)
    rejection_p_aaac = rejection_vals.get(('A', 'A', 'A', 'C'), 0.0)

    print(f"{'Gibbs':>10} {gibbs_p_aaac:.4f}")
    print(f"{'Rejection':>10} {rejection_p_aaac:.4f}")
    print(f"{'Exact':>10} {exact_inference():.4f}")

# Uncomment to test Gibbs vs. rejection sampling
# test_gibbs_vs_rejection(num_steps=100, mutation_rate=0.1, genome_length=4)
# test_gibbs_vs_rejection(num_steps=10000, mutation_rate=0.1, genome_length=4)

############################################################
# Problem 3d: Bayesian network for annotators

def bayesian_network_for_annotators(num_annotators: int, dataset_size: int=1) -> BayesianNetwork:
    """
    Return the Bayesian network for the annotators.
    """
    pass
    # ### START CODE HERE ###
    labels = BayesianNode(
        name='Y',
        domain=['good', 'bad'],
        conditional_prob_table=np.array([[0.5, 0.5]])
    )
    annotator_cpt = np.array([[0.8, 0.2], [0.2, 0.8]])
    annotators = [
        BayesianNode(
            name=f'A_{i}',
            domain=['good', 'bad'],
            parents=[labels],
            conditional_prob_table=annotator_cpt.copy()
        )
        for i in range(num_annotators)
    ]
    return BayesianNetwork([labels] + annotators, batch_size=dataset_size)
    # ### END CODE HERE ###

############################################################
# Problem 3e: Maximum likelihood estimation

def accumulate_assignment(
    counts: Dict[str, np.ndarray],
    network: BayesianNetwork,
    assignment: Dict[str, str],
    weight: float = 1.0,
    batch_indices: Optional[List[int]] = None,
) -> None:
    """
    Add weighted counts for a fully or partially observed assignment.
    """
    batch_size = len(list(assignment.values())[0])
    for i in range(batch_size) if batch_indices is None else range(len(batch_indices)):
        idx = batch_indices[i] if batch_indices is not None else i
        assignment_i = {k: v[i] for k, v in assignment.items()}
        for node in network.nodes:
            pass
            # ### START CODE HERE ###
            value = assignment_i[node.name]
            value_idx = node.domain.index(value)
            if not node.parents:
                counts[node.name][idx, value_idx] += weight
            else:
                parent_indices = node.parent_assignment_indices(assignment_i)
                counts[node.name][parent_indices + (value_idx,)] += weight
            # ### END CODE HERE ###

def mle_estimation(network: BayesianNetwork, data: List[Dict[str, List[str]]], lambda_param: float = 1.0) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE.
    """
    pass
    # ### START CODE HERE ###
    counts = init_zero_conditional_probability_tables(network)
    for node in network.nodes:
        counts[node.name] += lambda_param
    for observation in data:
        accumulate_assignment(counts, network, observation)
    normalize_counts(network, counts)
    return network
    # ### END CODE HERE ###

############################################################
# Problem 3f: Maximum likelihood estimation for annotators

def mle_estimation_for_annotators(data: List[Dict[str, List[str]]]) -> BayesianNetwork:
    """
    Return the Bayesian network with the parameters estimated by MLE for the annotators.
    """
    pass
    # ### START CODE HERE ###
    num_annotators = sum(1 for k in data[0].keys() if k.startswith('A_'))
    dataset_size = len(list(data[0].values())[0])
    network = bayesian_network_for_annotators(num_annotators=num_annotators, dataset_size=dataset_size)
    return mle_estimation(network, data)
    # ### END CODE HERE ###

def test_mle_estimation_for_annotators():
    data = load_annotation_csv('data/annotations.csv', include_labels=True)
    trained = mle_estimation_for_annotators(data)
    plot_annotator_cpts(trained, "plots/annotators.png")

# test_mle_estimation_for_annotators()

############################################################
# Problem 4a: Expectation step

def e_step(
    network: BayesianNetwork, data: List[Dict[str, List[str]]]
) -> Tuple[List[Dict[str, List[str]]], List[float], List[List[int]]]:
    """
    Create the dataset of fully-observed weighted observations given some hidden variables, for the EM algorithm.
    """
    pass
    # ### START CODE HERE ###
    all_completions = []
    all_weights = []
    all_indices = []

    for observation in data:
        batch_size = len(list(observation.values())[0])
        observed_vars = set(observation.keys())
        hidden_nodes = [node for node in network.order if node.name not in observed_vars]
        hidden_var_names = [n.name for n in hidden_nodes]
        hidden_domains = [n.domain for n in hidden_nodes]

        for batch_pos in range(batch_size):
            obs_slice = {k: [v[batch_pos]] for k, v in observation.items()}

            if not hidden_var_names:
                all_completions.append(obs_slice)
                all_weights.append(1.0)
                all_indices.append([batch_pos])
            else:
                completions_raw = []
                probs = []
                for values in product(*hidden_domains):
                    completion = dict(obs_slice)
                    for var, val in zip(hidden_var_names, values):
                        completion[var] = [val]
                    prob = compute_joint_probability(network, completion, batch_indices=[batch_pos])
                    completions_raw.append(completion)
                    probs.append(prob)

                total = sum(probs)
                for completion, prob in zip(completions_raw, probs):
                    weight = prob / total if total > 0 else 1.0 / len(probs)
                    all_completions.append(completion)
                    all_weights.append(weight)
                    all_indices.append([batch_pos])

    return all_completions, all_weights, all_indices
    # ### END CODE HERE ###

############################################################
# Problem 4b: Maximization step

def m_step(
    network: BayesianNetwork,
    all_completions: List[Dict[str, str]],
    all_weights: List[float],
    all_indices: List[int],
) -> BayesianNetwork:
    """
    Update the CPTs of the Bayesian network using expected counts.
    """
    pass
    # ### START CODE HERE ###
    counts = init_zero_conditional_probability_tables(network)
    for completion, weight, indices in zip(all_completions, all_weights, all_indices):
        accumulate_assignment(counts, network, completion, weight=weight, batch_indices=indices)
    normalize_counts(network, counts)
    return network
    # ### END CODE HERE ###

############################################################
# Problem 4c: Expectation maximization

def em_learn(network: BayesianNetwork, data: List[Dict[str, str]], num_iterations: int) -> BayesianNetwork:
    """
    Run the EM algorithm for a given number of iterations.
    """
    pass
    # ### START CODE HERE ###
    for _ in range(num_iterations):
        all_completions, all_weights, all_indices = e_step(network, data)
        network = m_step(network, all_completions, all_weights, all_indices)
    return network
    # ### END CODE HERE ###

def test_em_learn():
    network = bayesian_network_for_annotators(num_annotators=3, dataset_size=100)
    data = load_annotation_csv('data/annotations.csv', include_labels=False)
    trained = em_learn(network, data, num_iterations=100)
    plot_annotator_cpts(trained, "plots/annotators_em.png")
    plot_label_cpt(trained, "plots/labels_em.png")

# Uncomment to test EM learning
# test_em_learn()
