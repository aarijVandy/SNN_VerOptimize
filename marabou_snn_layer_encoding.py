"""
This script demonstrates how to encode the spiking behavior of a SINGLE LAYER of a
Spiking Neural Network (SNN) as a verification query for the Marabou solver.

This approach completely AVOIDS the 2^n combinatorial explosion seen in the
STL_encoding.py script by using Marabou's native ability to handle piece-wise
linear constraints (specifically, disjunctions that model ReLUs and indicator
functions).

The core problem we are encoding is:
"For a single layer of an SNN, if all input neurons spike at time 0, does the
membrane potential of a specific output neuron exceed its threshold at time t=2?"

This involves three key steps:
1.  Representing the input neuron spike times as input variables to Marabou.
2.  Modeling the non-linear indicator function I(s_i <= t-1) using
    auxiliary variables and disjunctive constraints ("OR" clauses).
3.  Defining the output neuron's potential as a linear combination of the
    indicator variables and asserting a property about it.



                                        DELETE LATER. THIS IS ONLY A STUB.
"""
import numpy as np
from maraboupy import Marabou, MarabouCore

# --- SNN and Verification Parameters ---
# Let's consider a single layer mapping from 10 input neurons to 5 output neurons.
input_neurons = 10
output_neurons = 5
weights = np.random.rand(output_neurons, input_neurons) * 2 - 1  # Random weights in [-1, 1]

# The property we want to verify
verification_timestep = 2
potential_threshold = 2.5
target_output_neuron = 3 # We'll check the potential of this neuron

# --- Create the Marabou Query ---
# The Marabou network will represent the unrolled computation of the SNN layer
# at our specific timestep.
network = Marabou.create_network()

# 1. Define Input Variables: The spike times of the input neurons
# The state 'x' for Marabou will be [s_0, s_1, ..., s_9, ind_0, ind_1, ..., ind_9]
# where s_i is spike time and ind_i is the associated indicator variable.
num_vars = input_neurons * 2
network.setNumberOfVariables(num_vars)

input_spike_time_vars = list(range(input_neurons))
indicator_vars = list(range(input_neurons, input_neurons * 2))

# Set bounds on spike times [0, infinity]. Marabou doesn't have infinity,
# so we can use a large number or leave the upper bound open.
for var in input_spike_time_vars: # here just increase by 1 for each layer
    network.setLowerBound(var, 0.0)

# 2. Model the Indicator Functions: I(s_i <= verification_timestep - 1)
# This is the most critical part. For each input neuron `i`, we create an
# auxiliary variable `indicator_i` which will be forced to be 1 if
# `s_i <= 1` and 0 if `s_i > 1`.
# We use a disjunction (an OR of constraints) for this.

for i in range(input_neurons):
    s_i = input_spike_time_vars[i]
    indicator_i = indicator_vars[i]

    # Force indicator_i to be binary (0 or 1)
    network.setLowerBound(indicator_i, 0)
    network.setUpperBound(indicator_i, 1)
    MarabouCore.addRelu(network, indicator_i, indicator_i) # Trick to enforce integer value for binary var. check for true

    # Disjunction: EITHER (s_i <= 1 AND indicator_i = 1) OR (s_i >= 2 AND indicator_i = 0)
    # Note: We use s_i >= 2 instead of s_i > 1 because Marabou works with closed intervals.


    ###                     THIS IS WRONG. UNNECESSARY EXTRA CONSTRAINT


    # Case 1: s_i <= 1  (i.e., -s_i >= -1) and indicator_i = 1
    case1_eqs = MarabouCore.EquationArray(2)
    case1_eqs[0] = MarabouCore.Equation(MarabouCore.Equation.LE);
    case1_eqs[0].addAddend(1.0, s_i); case1_eqs[0].setScalar(1.0) # s_i <= 1
    case1_eqs[1] = MarabouCore.Equation(MarabouCore.Equation.EQ);
    case1_eqs[1].addAddend(1.0, indicator_i); case1_eqs[1].setScalar(1.0) # indicator_i = 1

    # Case 2: s_i >= 2 and indicator_i = 0
    case2_eqs = MarabouCore.EquationArray(2)
    case2_eqs[0] = MarabouCore.Equation(MarabouCore.Equation.GE);
    case2_eqs[0].addAddend(1.0, s_i); case2_eqs[0].setScalar(2.0) # s_i >= 2
    case2_eqs[1] = MarabouCore.Equation(MarabouCore.Equation.EQ);
    case2_eqs[1].addAddend(1.0, indicator_i); case2_eqs[1].setScalar(0.0) # indicator_i = 0
    ###                     THIS IS WRONG

    ###                     OWN IMPLEMENTATION

    # Case 1: s_i <= 1  (i.e., -s_i >= -1) and indicator_i = 1
    network.setUpperBound(s_i, 1)









    #  case1 OR case 2
    network.addDisjunction([case1_eqs, case2_eqs]) 

# 3. Define Output Potential and Assert Property
# The potential of the target neuron is: `potential = sum(weights * indicators)`
# This is a purely linear equation that Marabou handles perfectly.
potential_equation = MarabouCore.Equation(MarabouCore.Equation.EQ)
for i in range(input_neurons):
    # we are adding ` -weights * indicator_i ` to the equation
    potential_equation.addAddend(-weights[target_output_neuron, i], indicator_vars[i])

# The variable `out_potential_var` will hold the calculated potential.
# We create a new variable for it by adding it to the equation with a coefficient of 1.
out_potential_var = network.getNewVariable()
potential_equation.addAddend(1.0, out_potential_var)
network.addEquation(potential_equation)

# --- Set the Verification Goal ---
# Let's check the property: IF all input neurons spike at time 0...
print("Setting input property: all input spike times are 0.")
for var in input_spike_time_vars:
    network.setUpperBound(var, 0.0) # And lower bound is already 0

# ...THEN does the potential of the target neuron exceed the threshold?
# We ask Marabou to find a counterexample where potential < threshold.
# If it can't, the property holds.
print(f"Query: Is it possible for potential of neuron {target_output_neuron} to be < {potential_threshold}?")
network.setUpperBound(out_potential_var, potential_threshold - 1e-6) # potential < threshold

# --- Solve the Query ---
print("\nSolving with Marabou...")
options = Marabou.createOptions(verbose=False)
vals, stats = network.solve(options=options)

# --- Report Results ---
if vals:
    print("\nResult: SAT. Marabou found a counterexample.")
    potential_val = vals[out_potential_var]
    print(f"  - The property is FALSE.")
    print(f"  - Achieved potential: {potential_val:.4f}, which is less than the threshold {potential_threshold}")
else:
    print("\nResult: UNSAT. Marabou could not find a counterexample.")
    print(f"  - The property is TRUE.")
    print(f"  - If all inputs spike at t=0, the potential of neuron {target_output_neuron} is always >= {potential_threshold} at t=2.") 