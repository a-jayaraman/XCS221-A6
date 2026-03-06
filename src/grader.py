#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect, collections
from graderUtil import graded, CourseTestRunner, GradedTestCase

import util
import numpy as np

# Import student submission
import submission

#############################################
# HELPER FUNCTIONS
#############################################
TOLERANCE = 1e-4  # For measuring whether two floats are equal
NUM_SAMPLES = 1000

def is_collection(x):
    return isinstance(x, list) or isinstance(x, tuple)


# Return whether two answers are equal.
def is_equal(true_answer, pred_answer, tolerance=TOLERANCE):
    # Handle floats specially
    if isinstance(true_answer, float) or isinstance(pred_answer, float):
        return abs(true_answer - pred_answer) < tolerance
    # Recurse on collections to deal with floats inside them
    if is_collection(true_answer) and is_collection(pred_answer) and len(true_answer) == len(pred_answer):
        for a, b in zip(true_answer, pred_answer):
            if not is_equal(a, b):
                return False
        return True
    if isinstance(true_answer, dict) and isinstance(pred_answer, dict):
        if len(true_answer) != len(pred_answer):
            return False
        for k, v in list(true_answer.items()):
            if not is_equal(pred_answer.get(k), v):
                return False
        return True

    # Numpy array comparison
    if type(true_answer).__name__ == 'ndarray':
        if isinstance(true_answer, np.ndarray) and isinstance(pred_answer, np.ndarray):
            if true_answer.shape != pred_answer.shape:
                return False
            for a, b in zip(true_answer, pred_answer):
                if not is_equal(a, b):
                    return False
            return True

    # Do normal comparison
    return true_answer == pred_answer

#########
# TESTS #
#########

class Test_2a(GradedTestCase):
  @graded()
  def test_0(self):
    """2a-0-basic: Test phylogenetic tree initialization with mutation rate"""
    mutation_rate = 0.1
    network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=1)

    # Check that network is a BayesianNetwork
    self.assertTrue(isinstance(network, util.BayesianNetwork), "network should be of instance BayesianNetwork")
    self.assertTrue(is_equal(1, network.batch_size), "network batch size expected to be of size 1")

    # Check that we have the right number of nodes
    self.assertTrue(is_equal(4, len(network.nodes)), "network.nodes incorrect node count")

    # Check that all nodes have the correct domain (nucleotides)
    expected_domain = ['A', 'C', 'T', 'G']
    for node in network.nodes:
        self.assertTrue(is_equal(expected_domain, node.domain), f"expected domain {expected_domain}, got {node.domain}")

    # Get nodes by name for easier checking
    nodes_by_name = {node.name: node for node in network.nodes}

    # Check that all expected nodes exist
    expected_names = ['Aryamus bayus', 'Humblus studentus', 'Thomas bayus', 'Kenius bayus']
    for name in expected_names:
        self.assertTrue(name in nodes_by_name, f"expected node '{name}' not found in network")

    # Check parent-child relationships
    # Thomas bayus should be the root (no parents)
    thomas = nodes_by_name['Thomas bayus']
    self.assertTrue(is_equal(0, len(thomas.parents)), "Thomas bayus should have no parents (it is the root)")

    # Ensure Thomas bayus prior replicated across batch dimension
    self.assertTrue(is_equal((network.batch_size, 4), thomas.conditional_prob_table.shape),
        f"Thomas bayus prior should have shape ({network.batch_size}, 4), got {thomas.conditional_prob_table.shape}")

    # Humblus studentus should have Thomas bayus as parent
    humblus = nodes_by_name['Humblus studentus']
    self.assertTrue(is_equal(1, len(humblus.parents)), "Humblus studentus should have exactly one parent")
    self.assertTrue(is_equal('Thomas bayus', humblus.parents[0].name),
        f"Humblus studentus should have Thomas bayus as parent, got '{humblus.parents[0].name}'")

    # Aryamus bayus should have Thomas bayus as parent
    aryamus = nodes_by_name['Aryamus bayus']
    self.assertTrue(is_equal(1, len(aryamus.parents)), "Aryamus bayus should have exactly one parent")
    self.assertTrue(is_equal('Thomas bayus', aryamus.parents[0].name),
        f"Aryamus bayus should have Thomas bayus as parent, got '{aryamus.parents[0].name}'")

    # Kenius bayus should have Aryamus bayus as parent
    kenius = nodes_by_name['Kenius bayus']
    self.assertTrue(is_equal(1, len(kenius.parents)), "Kenius bayus should have exactly one parent")
    self.assertTrue(is_equal('Aryamus bayus', kenius.parents[0].name),
        f"Kenius bayus should have Aryamus bayus as parent, got '{kenius.parents[0].name}'")

    # Check that Thomas bayus has uniform prior
    self.assertTrue(np.allclose(thomas.conditional_prob_table, 0.25),
        "Thomas bayus should have a uniform prior (0.25 for each nucleotide)")

    # Check CPT structure for non-root nodes
    # Expected CPT: diagonal = (1 - mutation_rate), off-diagonal = mutation_rate / 3
    expected_cpt = np.eye(4) * (1 - mutation_rate) + (1 - np.eye(4)) * (mutation_rate / 3)

    for node_name in ['Humblus studentus', 'Aryamus bayus', 'Kenius bayus']:
        node = nodes_by_name[node_name]
        self.assertTrue(is_equal((4, 4), node.conditional_prob_table.shape),
            f"{node_name} conditional_prob_table should have shape (4, 4), got {node.conditional_prob_table.shape}")
        self.assertTrue(
            np.allclose(node.conditional_prob_table, expected_cpt),
            f"{node_name} conditional_prob_table does not match expected CPT for mutation_rate={mutation_rate}")

  @graded()
  def test_1(self):
    """2a-1-basic: Test mutation model CPT with various mutation rates"""
    for mutation_rate in [0.0, 0.05, 0.2, 0.5]:
      for genome_length in [1, 10, 50]:
        network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)
        self.assertTrue(is_equal(4, len(network.nodes)),
            f"expected 4 nodes, got {len(network.nodes)}")
        self.assertTrue(is_equal(genome_length, network.batch_size),
            f"expected batch_size={genome_length}, got {network.batch_size}")
        nodes_by_name = {node.name: node for node in network.nodes}

        thomas = nodes_by_name['Thomas bayus']
        self.assertTrue(is_equal(0, len(thomas.parents)),
            "Thomas bayus should have no parents (it is the root)")
        self.assertTrue(is_equal((genome_length, 4), thomas.conditional_prob_table.shape),
            f"Thomas bayus prior should have shape ({genome_length}, 4), got {thomas.conditional_prob_table.shape}")

        # Expected CPT for mutation model
        expected_cpt = np.eye(4) * (1 - mutation_rate) + (1 - np.eye(4)) * (mutation_rate / 3)

        # Check that each row sums to 1
        for i in range(4):
            row_sum = np.sum(expected_cpt[i, :])
            self.assertTrue(np.isclose(row_sum, 1.0),
                f"CPT row {i} should sum to 1.0, got {row_sum}")

        # Check CPT for each non-root node
        for node_name in ['Humblus studentus', 'Aryamus bayus', 'Kenius bayus']:
            node = nodes_by_name[node_name]
            self.assertTrue(is_equal((4, 4), node.conditional_prob_table.shape),
                f"{node_name} conditional_prob_table should have shape (4, 4), got {node.conditional_prob_table.shape}")

            # Verify diagonal elements (no mutation)
            for i in range(4):
                actual = node.conditional_prob_table[i, i]
                expected = 1 - mutation_rate
                self.assertTrue(np.isclose(actual, expected),
                    f"{node_name} diagonal element [{i},{i}] should be {expected} (1 - mutation_rate), got {actual}")

            # Verify off-diagonal elements (mutation)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        actual = node.conditional_prob_table[i, j]
                        expected = mutation_rate / 3
                        self.assertTrue(np.isclose(actual, expected),
                            f"{node_name} off-diagonal element [{i},{j}] should be {expected} (mutation_rate/3), got {actual}")

class Test_2b(GradedTestCase):
   @graded()
   def test_0(self):
      """2b-0-basic: Test that forward_sampling produces valid samples"""
      mutation_rate = 0.1
      network = submission.initialize_phylogenetic_tree(mutation_rate)

      # Sample once
      sample = submission.forward_sampling(network)

      # Check that sample is a dictionary
      self.assertTrue(isinstance(sample, dict),
          f"forward_sampling should return a dict, got {type(sample)}")

      # Check that all nodes are in the sample
      expected_names = ['Aryamus bayus', 'Humblus studentus', 'Thomas bayus', 'Kenius bayus']
      for name in expected_names:
          self.assertTrue(name in sample,
              f"expected node '{name}' missing from forward_sampling result")

      # Check that all sampled values are valid nucleotides
      valid_nucleotides = {'A', 'C', 'T', 'G'}
      for name, value in sample.items():
          self.assertTrue(value[0] in valid_nucleotides,
              f"sampled value '{value[0]}' for '{name}' is not a valid nucleotide (expected one of {valid_nucleotides})")

class Test_2c(GradedTestCase):
   @graded()
   def test_0(self):
      """2c-0-basic: Test compute_joint_probability"""
      mutation_rate = 0.1
      network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=1)

      # Test with a specific assignment
      assignment = {
          'Aryamus bayus': ['A'],
          'Humblus studentus': ['A'],
          'Thomas bayus': ['A'],
          'Kenius bayus': ['A']
      }

      prob = submission.compute_joint_probability(network, assignment)

      # Check that probability is a float
      self.assertTrue(isinstance(prob, float),
          f"compute_joint_probability should return a float, got {type(prob)}")

      # Check that probability is between 0 and 1
      self.assertTrue(0 <= prob <= 1,
          f"probability should be in [0, 1], got {prob}")

      # Check that probability is positive (since all transitions are possible)
      self.assertTrue(prob > 0,
          "probability for a valid all-A assignment should be positive")

      # Manually compute expected probability
      # P(A_bayus=A) = 0.25
      # P(H_studentus=A | A_bayus=A) = 1 - mutation_rate = 0.9
      # P(T_bayus=A | A_bayus=A) = 1 - mutation_rate = 0.9
      # P(K_bayus=A | T_bayus=A) = 1 - mutation_rate = 0.9
      expected_prob = 0.25 * (1 - mutation_rate) ** 3
      self.assertTrue(np.isclose(prob, expected_prob),
          f"expected joint probability {expected_prob}, got {prob}")


class Test_2e(GradedTestCase):
   @graded()
   def test_0(self):
      """2e-0-basic: Rejection sampling structural and deterministic checks"""
      valid_nucleotides = {'A', 'C', 'T', 'G'}

      # --- Structural checks ---
      mutation_rate = 0.1
      genome_length = 1
      network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)
      random.seed(42)
      np.random.seed(42)
      result = submission.rejection_sampling(
          network, 'Thomas bayus', {'Kenius bayus': ['A']}, num_samples=NUM_SAMPLES)

      self.assertTrue(isinstance(result, dict),
          f"rejection_sampling should return a dict, got {type(result)}")
      self.assertTrue(len(result) > 0,
          "rejection_sampling returned an empty dict; no samples were accepted")
      for key, value in result.items():
          key_tuple = tuple(key) if isinstance(key, (str, list)) else tuple(key)
          self.assertTrue(len(key_tuple) == genome_length,
              f"sampled genome {key_tuple} has length {len(key_tuple)}, expected {genome_length}")
          for nuc in key_tuple:
              self.assertTrue(nuc in valid_nucleotides,
                  f"sampled genome {key_tuple} contains invalid nucleotide '{nuc}'")
          self.assertTrue(float(value) >= 0,
              f"rejection_sampling returned negative count/weight {value} for genome {key_tuple}")

      # --- Deterministic check: mutation_rate=0 means no mutations occur ---
      # Thomas -> Aryamus -> Kenius with no changes, so Kenius = Thomas always.
      # Therefore P(Thomas=A | Kenius=A) = 1.0; only 'A' should ever be accepted.
      det_network = submission.initialize_phylogenetic_tree(0.0, genome_length=1)
      random.seed(0)
      np.random.seed(0)
      det_result = submission.rejection_sampling(
          det_network, 'Thomas bayus', {'Kenius bayus': ['A']}, num_samples=500)

      self.assertTrue(isinstance(det_result, dict),
          "rejection_sampling with mutation_rate=0 should return a dict")
      total = sum(float(v) for v in det_result.values())
      self.assertTrue(total > 0,
          "rejection_sampling with mutation_rate=0 accepted no samples; all samples should match evidence")
      for key, value in det_result.items():
          key_tuple = tuple(key) if isinstance(key, (str, list)) else tuple(key)
          self.assertTrue(key_tuple == ('A',),
              f"with mutation_rate=0 and Kenius=['A'], Thomas must always be 'A', got {key_tuple}")

   # BEGIN_HIDE
  # END_HIDE

class Test_2f(GradedTestCase):
   @graded()
   def test_0(self):
      """2f-0-basic: Gibbs sampling structural checks"""
      valid_nucleotides = {'A', 'C', 'T', 'G'}

      def parse_raw(raw_result):
          """Normalise the various return formats into a list of (tuple, weight) pairs."""
          if isinstance(raw_result, dict) and 'Thomas bayus' in raw_result \
                  and isinstance(raw_result['Thomas bayus'], dict):
              raw_local = raw_result['Thomas bayus']
          else:
              raw_local = raw_result
          if isinstance(raw_local, list):
              return [(tuple(e) if not isinstance(e, tuple) else e, 1.0) for e in raw_local]
          if isinstance(raw_local, dict):
              return [(tuple(k) if not isinstance(k, tuple) else k, float(v))
                      for k, v in raw_local.items()]
          self.fail(f"gibbs_sampling must return a dict or list, got {type(raw_local)}")
          return []

      # --- Structural checks ---
      mutation_rate = 0.1
      genome_length = 2
      network = submission.initialize_phylogenetic_tree(mutation_rate, genome_length=genome_length)
      random.seed(42)
      np.random.seed(42)
      result = submission.gibbs_sampling(
          network, 'Thomas bayus', {'Kenius bayus': ['A'] * genome_length},
          num_iterations=NUM_SAMPLES)

      items = parse_raw(result)
      self.assertTrue(len(items) > 0,
          "gibbs_sampling returned no samples")
      for key_tuple, value in items:
          self.assertTrue(len(key_tuple) == genome_length,
              f"sampled genome {key_tuple} has length {len(key_tuple)}, expected {genome_length}")
          for nuc in key_tuple:
              self.assertTrue(nuc in valid_nucleotides,
                  f"sampled genome {key_tuple} contains invalid nucleotide '{nuc}'")
          self.assertTrue(value >= 0,
              f"gibbs_sampling returned negative count/weight {value} for genome {key_tuple}")

      # --- Normalisation check: weights should sum to a positive value ---
      # (Note: mutation_rate=0 is intentionally avoided for Gibbs since deterministic
      # CPTs cause 0/0 in the conditional resampling step when the chain gets into an
      # inconsistent state, making the sampler undefined in that regime.)
      total_weight = sum(float(v) for _, v in items)
      self.assertTrue(total_weight > 0,
          f"total weight of all gibbs_sampling samples is {total_weight}; expected > 0")

   # BEGIN_HIDE
    # END_HIDE


class Test_3d(GradedTestCase):
  @graded()
  def test_0(self):
    """3d-0-basic: Annotator network structure"""
    num_annotators = 4
    dataset_size = 25
    network = submission.bayesian_network_for_annotators(num_annotators, dataset_size)

    self.assertTrue(isinstance(network, util.BayesianNetwork),
        "bayesian_network_for_annotators should return a BayesianNetwork instance")
    self.assertTrue(is_equal(dataset_size, network.batch_size),
        f"network.batch_size should be {dataset_size}, got {network.batch_size}")

    nodes_by_name = {node.name: node for node in network.nodes}
    self.assertTrue('Y' in nodes_by_name,
        "network is missing a node named 'Y' (the labels node)")

    labels_node = nodes_by_name['Y']
    self.assertTrue(is_equal(['good', 'bad'], labels_node.domain),
        f"Y node domain should be ['good', 'bad'], got {labels_node.domain}")
    self.assertTrue(is_equal(0, len(labels_node.parents)),
        f"Y node should have no parents (it is the root), got {len(labels_node.parents)} parent(s)")
    self.assertTrue(is_equal((dataset_size, 2), labels_node.conditional_prob_table.shape),
        f"Y node CPT shape should be ({dataset_size}, 2), got {labels_node.conditional_prob_table.shape}")
    self.assertTrue(np.allclose(labels_node.conditional_prob_table, 0.5),
        "Y node CPT should be initialized to a uniform prior (0.5 each)")

    for i in range(num_annotators):
        annotator_name = f'A_{i}'
        self.assertTrue(annotator_name in nodes_by_name,
            f"network is missing annotator node '{annotator_name}'")
        annotator_node = nodes_by_name[annotator_name]
        self.assertTrue(is_equal(['good', 'bad'], annotator_node.domain),
            f"annotator node '{annotator_name}' domain should be ['good', 'bad'], got {annotator_node.domain}")
        self.assertTrue(is_equal(1, len(annotator_node.parents)),
            f"annotator node '{annotator_name}' should have exactly 1 parent (Y), got {len(annotator_node.parents)}")
        self.assertTrue(is_equal('Y', annotator_node.parents[0].name),
            f"annotator node '{annotator_name}' parent should be 'Y', got '{annotator_node.parents[0].name}'")
        self.assertTrue(is_equal((2, 2), annotator_node.conditional_prob_table.shape),
            f"annotator node '{annotator_name}' CPT shape should be (2, 2), got {annotator_node.conditional_prob_table.shape}")

    expected_node_count = num_annotators + 1  # annotators + labels
    self.assertTrue(is_equal(expected_node_count, len(network.nodes)),
        f"network should have {expected_node_count} nodes ({num_annotators} annotators + 1 labels), got {len(network.nodes)}")

  @graded()
  def test_1(self):
      """3d-1-basic: Annotator CPT diagonal dominance"""
      np.random.seed(0)
      network1 = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)

      for idx in range(3):
          node1 = network1.get_node_by_name(f'A_{idx}')
          self.assertTrue(node1.conditional_prob_table.shape == (2, 2),
              f"A_{idx} CPT should have shape (2, 2), got {node1.conditional_prob_table.shape}")
          self.assertTrue(node1.conditional_prob_table[0, 0] > node1.conditional_prob_table[0, 1],
              f"A_{idx} CPT[0,0] should be > CPT[0,1]: annotators should agree with themselves "
              f"(P(good|Y=good) > P(bad|Y=good)), got {node1.conditional_prob_table[0, 0]:.4f} vs {node1.conditional_prob_table[0, 1]:.4f}")
          self.assertTrue(node1.conditional_prob_table[0, 0] > node1.conditional_prob_table[1, 0],
              f"A_{idx} CPT[0,0] should be > CPT[1,0]: P(good|Y=good) > P(good|Y=bad), "
              f"got {node1.conditional_prob_table[0, 0]:.4f} vs {node1.conditional_prob_table[1, 0]:.4f}")
          self.assertTrue(node1.conditional_prob_table[1, 1] > node1.conditional_prob_table[1, 0],
              f"A_{idx} CPT[1,1] should be > CPT[1,0]: P(bad|Y=bad) > P(good|Y=bad), "
              f"got {node1.conditional_prob_table[1, 1]:.4f} vs {node1.conditional_prob_table[1, 0]:.4f}")
          self.assertTrue(node1.conditional_prob_table[1, 1] > node1.conditional_prob_table[0, 1],
              f"A_{idx} CPT[1,1] should be > CPT[0,1]: P(bad|Y=bad) > P(bad|Y=good), "
              f"got {node1.conditional_prob_table[1, 1]:.4f} vs {node1.conditional_prob_table[0, 1]:.4f}")
  
  # BEGIN_HIDE
    # END_HIDE

class Test_3e(GradedTestCase):
  @graded()
  def test_0(self):
      """3e-0-basic: accumulate_assignment counting"""
      network = submission.bayesian_network_for_annotators(num_annotators=2, dataset_size=1)
      counts = util.init_zero_conditional_probability_tables(network)
      assignment = {
          'Y': ['good'],
          'A_0': ['good'],
          'A_1': ['bad'],
      }
      submission.accumulate_assignment(counts, network, assignment)

      labels_counts = counts['Y'][0]
      self.assertTrue(labels_counts[0] > labels_counts[1],
          f"Y counts[good] should be > counts[bad] since Y='good' was assigned; "
          f"got counts[good]={labels_counts[0]}, counts[bad]={labels_counts[1]}")

      annot0_counts = counts['A_0'][0]
      self.assertTrue(annot0_counts[0] > annot0_counts[1],
          f"A_0 counts[good|Y=good] should be > counts[bad|Y=good] since A_0='good' and Y='good'; "
          f"got counts[good]={annot0_counts[0]}, counts[bad]={annot0_counts[1]}")

      annot1_counts = counts['A_1'][0]
      self.assertTrue(annot1_counts[1] > annot1_counts[0],
          f"A_1 counts[bad|Y=good] should be > counts[good|Y=good] since A_1='bad' and Y='good'; "
          f"got counts[good]={annot1_counts[0]}, counts[bad]={annot1_counts[1]}")
  
  @graded()
  def test_1(self):
      """3e-1-basic: mle_estimation parameter learning"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
      data = [
          {'Y': ['good'], 'A_0': ['good']},
          {'Y': ['good'], 'A_0': ['good']},
          {'Y': ['bad'], 'A_0': ['bad']},
          {'Y': ['bad'], 'A_0': ['bad']},
      ]
      trained = submission.mle_estimation(network, data, lambda_param=0.0)

      labels_node = trained.get_node_by_name('Y')
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 0], 0.5),
          f"Y CPT[0,0] (P(good)) should be 0.5 with equal good/bad data, got {labels_node.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 1], 0.5),
          f"Y CPT[0,1] (P(bad)) should be 0.5 with equal good/bad data, got {labels_node.conditional_prob_table[0, 1]:.6f}")

      annot_node = trained.get_node_by_name('A_0')
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 0], 1.0),
          f"A_0 CPT[0,0] (P(good|Y=good)) should be 1.0 (annotator always agreed), got {annot_node.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 1], 0.0),
          f"A_0 CPT[0,1] (P(bad|Y=good)) should be 0.0 (annotator never disagreed on good), got {annot_node.conditional_prob_table[0, 1]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 0], 0.0),
          f"A_0 CPT[1,0] (P(good|Y=bad)) should be 0.0 (annotator never disagreed on bad), got {annot_node.conditional_prob_table[1, 0]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 1], 1.0),
          f"A_0 CPT[1,1] (P(bad|Y=bad)) should be 1.0 (annotator always agreed), got {annot_node.conditional_prob_table[1, 1]:.6f}")

  @graded()
  def test_2(self):
    """3e-2-basic: mle_estimation parameter learning"""
    network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
    data = [
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['good'], 'A_0': ['good']},
        {'Y': ['bad'], 'A_0': ['bad']},
        {'Y': ['bad'], 'A_0': ['bad']},
    ]
    trained = submission.mle_estimation(network, data, lambda_param=1.0)

    labels_node = trained.get_node_by_name('Y')
    self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 0], 0.5),
        f"Y CPT[0,0] (P(good)) should be 0.5 with equal good/bad data and lambda_param=1.0, "
        f"got {labels_node.conditional_prob_table[0, 0]:.6f}")
    self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 1], 0.5),
        f"Y CPT[0,1] (P(bad)) should be 0.5 with equal good/bad data and lambda_param=1.0, "
        f"got {labels_node.conditional_prob_table[0, 1]:.6f}")

    annot_node = trained.get_node_by_name('A_0')
    self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 0], 3 / 4),
        f"A_0 CPT[0,0] (P(good|Y=good)) should be 3/4 with Laplace smoothing (lambda=1), "
        f"got {annot_node.conditional_prob_table[0, 0]:.6f}")
    self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 1], 1 / 4),
        f"A_0 CPT[0,1] (P(bad|Y=good)) should be 1/4 with Laplace smoothing (lambda=1), "
        f"got {annot_node.conditional_prob_table[0, 1]:.6f}")
    self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 0], 1 / 4),
        f"A_0 CPT[1,0] (P(good|Y=bad)) should be 1/4 with Laplace smoothing (lambda=1), "
        f"got {annot_node.conditional_prob_table[1, 0]:.6f}")
    self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 1], 3 / 4),
        f"A_0 CPT[1,1] (P(bad|Y=bad)) should be 3/4 with Laplace smoothing (lambda=1), "
        f"got {annot_node.conditional_prob_table[1, 1]:.6f}")
  

class Test_3f(GradedTestCase):
  @graded()
  def test_0(self):
      """3f-0-basic: MLE for annotators"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)
      data = [
          {'Y': ['good', 'bad'], 'A_0': ['good', 'bad']},
      ]
      trained = submission.mle_estimation(network, data, lambda_param=0.0)
      self.assertTrue(np.isclose(trained.get_node_by_name('Y').conditional_prob_table[0, 0], 1.0),
          f"Y CPT[0,0] (P(good)) should be 1.0 (all observed labels were 'good'), "
          f"got {trained.get_node_by_name('Y').conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(trained.get_node_by_name('Y').conditional_prob_table[0, 1], 0.0),
          f"Y CPT[0,1] (P(bad)) should be 0.0 (no 'bad' labels observed), "
          f"got {trained.get_node_by_name('Y').conditional_prob_table[0, 1]:.6f}")
  
  # BEGIN_HIDE
  # END_HIDE

class Test_4a(GradedTestCase):
  @graded()
  def test_0(self):
      """4a-0-basic: Expectation step expected counts"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
      assignment = {'Y': ['good'], 'A_0': ['bad']}
      expected_counts = util.init_zero_conditional_probability_tables(network)
      submission.accumulate_assignment(expected_counts, network, assignment, weight=1.0)

      completions, weights, indices = submission.e_step(network, [assignment])
      self.assertTrue(is_equal(1, len(completions)),
          f"e_step with a fully-observed assignment should return 1 completion, got {len(completions)}")
      self.assertTrue(is_equal(1, len(weights)),
          f"e_step with a fully-observed assignment should return 1 weight, got {len(weights)}")
      self.assertTrue(is_equal(1, len(indices)),
          f"e_step with a fully-observed assignment should return 1 index list, got {len(indices)}")
      self.assertTrue(is_equal(assignment, completions[0]),
          f"e_step completion for a fully-observed assignment should equal the original assignment; "
          f"got {completions[0]}, expected {assignment}")
      self.assertTrue(is_equal(1.0, weights[0]),
          f"e_step weight for a fully-observed assignment should be 1.0, got {weights[0]}")
      self.assertTrue(is_equal([0], indices[0]),
          f"e_step index for the first data point should be [0], got {indices[0]}")
  
  @graded()
  def test_1(self):
      """4a-1-basic: Expectation step with hidden variables"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
      assignment = {'A_0': ['good']}

      completions, weights, indices = submission.e_step(network, [assignment])
      self.assertTrue(is_equal(2, len(completions)))
      self.assertTrue(is_equal(2, len(weights)))
      self.assertTrue(is_equal(2, len(indices)))
      self.assertTrue(np.isclose(sum(weights), 1.0),
          f"e_step weights should sum to 1.0, got {sum(weights):.6f}")
    
  @graded()
  def test_2(self):
      """4a-2-basic: Expectation step multiple hidden vars"""
      network = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)
      labels_node = network.get_node_by_name('Y')
      labels_node.conditional_prob_table[:] = np.array([[0.6, 0.4]])
      for i in range(3):
          annot_node = network.get_node_by_name(f'A_{i}')
          annot_node.conditional_prob_table[:] = np.array([[0.9, 0.1], [0.2, 0.8]])

      assignment = {
          'A_0': ['good'],
          'A_1': ['good'],
      }

      completions, weights, indices = submission.e_step(network, [assignment])
      self.assertTrue(is_equal(4, len(completions)),
          f"e_step with 2 hidden variables (Y, A_2) over a binary domain should return 4 completions "
          f"(2^2 = 4), got {len(completions)}")
      self.assertTrue(is_equal(4, len(weights)),
          f"e_step should return one weight per completion (expected 4), got {len(weights)}")
      self.assertTrue(is_equal(4, len(indices)),
          f"e_step should return one index list per completion (expected 4), got {len(indices)}")
      self.assertTrue(np.isclose(sum(weights), 1.0),
          f"e_step weights should sum to 1.0 (they are a posterior distribution), got {sum(weights):.6f}")

      expected = {
          ('good', 'good'): 0.8713147410358565,
          ('good', 'bad'): 0.09681274900398405,
          ('bad', 'good'): 0.006374501992031873,
          ('bad', 'bad'): 0.02549800796812749,
      }

      actual = {}
      for comp, wt in zip(completions, weights):
          self.assertTrue(is_equal(comp['A_0'], ['good']),
              f"A_0 was observed as 'good' and should not change in any completion, got {comp['A_0']}")
          self.assertTrue(is_equal(comp['A_1'], ['good']),
              f"A_1 was observed as 'good' and should not change in any completion, got {comp['A_1']}")
          key = (comp['Y'][0], comp['A_2'][0])
          actual[key] = wt

      for key, target in expected.items():
          self.assertTrue(np.isclose(actual.get(key, 0.0), target, atol=1e-3),
              f"e_step posterior weight for (Y={key[0]}, A_2={key[1]}) is {actual.get(key, 0.0):.6f}, "
              f"expected ~{target:.6f}")
  
  # BEGIN_HIDE
  # END_HIDE

class Test_4b(GradedTestCase):
  @graded()
  def test_0(self):
      """4b-0-basic: Maximization step parameter update"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=1)
      completions = [
          {'Y': ['good'], 'A_0': ['good']},
          {'Y': ['bad'], 'A_0': ['bad']},
      ]
      weights = [0.8, 0.2]
      indices = [[0], [0]]
      trained = submission.m_step(network, completions, weights, indices)

      labels_node = trained.get_node_by_name('Y')
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 0], 0.8),
          f"Y CPT[0,0] (P(good)) should be 0.8 (weight 0.8 on 'good'), "
          f"got {labels_node.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 1], 0.2),
          f"Y CPT[0,1] (P(bad)) should be 0.2 (weight 0.2 on 'bad'), "
          f"got {labels_node.conditional_prob_table[0, 1]:.6f}")

      annot_node = trained.get_node_by_name('A_0')
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 0], 1.0),
          f"A_0 CPT[0,0] (P(good|Y=good)) should be 1.0 (only 'good' observed when Y='good'), "
          f"got {annot_node.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[0, 1], 0.0),
          f"A_0 CPT[0,1] (P(bad|Y=good)) should be 0.0 (no 'bad' observed when Y='good'), "
          f"got {annot_node.conditional_prob_table[0, 1]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 0], 0.0),
          f"A_0 CPT[1,0] (P(good|Y=bad)) should be 0.0 (no 'good' observed when Y='bad'), "
          f"got {annot_node.conditional_prob_table[1, 0]:.6f}")
      self.assertTrue(np.isclose(annot_node.conditional_prob_table[1, 1], 1.0),
          f"A_0 CPT[1,1] (P(bad|Y=bad)) should be 1.0 (only 'bad' observed when Y='bad'), "
          f"got {annot_node.conditional_prob_table[1, 1]:.6f}")

  
  @graded()
  def test_1(self):
      """4b-1-basic: Maximization step multiple completions"""
      network = submission.bayesian_network_for_annotators(num_annotators=3, dataset_size=1)
      completions = [
          {'Y': ['good'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['good']},
          {'Y': ['good'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['bad']},
          {'Y': ['bad'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['good']},
          {'Y': ['bad'], 'A_0': ['good'], 'A_1': ['good'], 'A_2': ['bad']},
      ]
      weights = [0.8713147410358565, 0.09681274900398405, 0.006374501992031873, 0.02549800796812749]
      indices = [[0], [0], [0], [0]]

      trained = submission.m_step(network, completions, weights, indices)

      labels_node = trained.get_node_by_name('Y')
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 0], 0.9681274900398406),
          f"Y CPT[0,0] (P(good)) should be ~0.9681 given the weighted completions, "
          f"got {labels_node.conditional_prob_table[0, 0]:.10f}")
      self.assertTrue(np.isclose(labels_node.conditional_prob_table[0, 1], 0.03187250996015936),
          f"Y CPT[0,1] (P(bad)) should be ~0.0319 given the weighted completions, "
          f"got {labels_node.conditional_prob_table[0, 1]:.10f}")

      annot2 = trained.get_node_by_name('A_2')
      self.assertTrue(np.isclose(annot2.conditional_prob_table[0, 0], 0.9),
          f"A_2 CPT[0,0] (P(good|Y=good)) should be ~0.9 given the weighted completions, "
          f"got {annot2.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(np.isclose(annot2.conditional_prob_table[0, 1], 0.1),
          f"A_2 CPT[0,1] (P(bad|Y=good)) should be ~0.1 given the weighted completions, "
          f"got {annot2.conditional_prob_table[0, 1]:.6f}")
      self.assertTrue(np.isclose(annot2.conditional_prob_table[1, 0], 0.2),
          f"A_2 CPT[1,0] (P(good|Y=bad)) should be ~0.2 given the weighted completions, "
          f"got {annot2.conditional_prob_table[1, 0]:.6f}")
      self.assertTrue(np.isclose(annot2.conditional_prob_table[1, 1], 0.8),
          f"A_2 CPT[1,1] (P(bad|Y=bad)) should be ~0.8 given the weighted completions, "
          f"got {annot2.conditional_prob_table[1, 1]:.6f}")
  
  # BEGIN_HIDE
  # END_HIDE

class Test_4c(GradedTestCase):
  @graded()
  def test_0(self):
      """4c-0-basic: EM convergence on hidden labels"""
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)

      labels_node = network.get_node_by_name('Y')
      labels_node.conditional_prob_table[:] = np.array([[0.5, 0.5], [0.5, 0.5]])
      annot_node = network.get_node_by_name('A_0')
      annot_node.conditional_prob_table[:] = np.array([[0.8, 0.2], [0.3, 0.7]])

      data = [
          {'A_0': ['good', 'bad']},
          {'A_0': ['good', 'bad']},
          {'A_0': ['good', 'bad']},
          {'A_0': ['bad', 'good']},
      ]

      trained = submission.em_learn(network, data, num_iterations=10)

      labels_node = trained.get_node_by_name('Y')
      self.assertTrue(labels_node.conditional_prob_table[0, 0] > 0.6,
          f"Y CPT[0,0] (P(good)) should be > 0.6 after EM (data has more 'good' observations), "
          f"got {labels_node.conditional_prob_table[0, 0]:.6f}")
      self.assertTrue(labels_node.conditional_prob_table[0, 1] < 0.4,
          f"Y CPT[0,1] (P(bad)) should be < 0.4 after EM, "
          f"got {labels_node.conditional_prob_table[0, 1]:.6f}")

      annot_node = trained.get_node_by_name('A_0')
      self.assertTrue(annot_node.conditional_prob_table[0, 0] > annot_node.conditional_prob_table[0, 1],
          f"A_0 CPT: P(good|Y=good) should be > P(bad|Y=good) after EM (annotator should agree when Y=good), "
          f"got {annot_node.conditional_prob_table[0, 0]:.4f} vs {annot_node.conditional_prob_table[0, 1]:.4f}")
      self.assertTrue(annot_node.conditional_prob_table[1, 1] > annot_node.conditional_prob_table[1, 0],
          f"A_0 CPT: P(bad|Y=bad) should be > P(good|Y=bad) after EM (annotator should agree when Y=bad), "
          f"got {annot_node.conditional_prob_table[1, 1]:.4f} vs {annot_node.conditional_prob_table[1, 0]:.4f}")
  
  @graded()
  def test_1(self):
      """4c-1-basic: EM updates from random init"""
      np.random.seed(1)
      network = submission.bayesian_network_for_annotators(num_annotators=1, dataset_size=2)

      labels_initial = network.get_node_by_name('Y').conditional_prob_table.copy()
      annot_initial = network.get_node_by_name('A_0').conditional_prob_table.copy()

      data = [
          {'A_0': ['good', 'bad']},
          {'A_0': ['good', 'bad']},
          {'A_0': ['good', 'bad']},
          {'A_0': ['bad', 'good']},
      ]

      trained = submission.em_learn(network, data, num_iterations=5)
      labels_final = trained.get_node_by_name('Y').conditional_prob_table
      annot_final = trained.get_node_by_name('A_0').conditional_prob_table

      self.assertTrue(not np.allclose(labels_final, labels_initial),
          "Y CPT should change after 5 EM iterations but remained identical to initialization; "
          "check that your e_step and m_step are both updating parameters")
      self.assertTrue(not np.allclose(annot_final, annot_initial),
          "A_0 CPT should change after 5 EM iterations but remained identical to initialization; "
          "check that your e_step and m_step are both updating parameters")


def getTestCaseForTestID(test_id):
  question, part, _ = test_id.split('-')
  g = globals().copy()
  for name, obj in g.items():
    if inspect.isclass(obj) and name == ('Test_'+question):
      return obj('test_'+part)

if __name__ == '__main__':
  # Parse for a specific test
  parser = argparse.ArgumentParser()
  parser.add_argument('test_case', nargs='?', default='all')
  test_id = parser.parse_args().test_case

  assignment = unittest.TestSuite()
  if test_id != 'all':
    assignment.addTest(getTestCaseForTestID(test_id))
  else:
    assignment.addTests(unittest.defaultTestLoader.discover('.', pattern='grader.py'))
  CourseTestRunner().run(assignment)