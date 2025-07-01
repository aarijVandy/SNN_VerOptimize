from itertools import product
from tqdm.auto import tqdm
from z3 import *
from typing import List, Union
from typing import cast as typecast
from .dictionary_mnist import *
from .debug import info

def get_layer_neurons_iter(layer: int):
    return product(range(layer_shapes[layer][0]), range(layer_shapes[layer][1]))

def gen_spike_times() -> TSpikeTime:
    """_summary_
    Generate spike time names in z3 Int terms. 
    layer 0 is input layer and last layer is output layer.
    for example for layer 3 , neuron 5, the spike time is dSpkTime_5_3

    Returns:
        TSpikeTime: Dictionary including z3 Int terms.
    """
    spike_times = typecast(TSpikeTime, {})
    for layer, _ in enumerate(n_layer_neurons):
        for layer_neuron in get_layer_neurons_iter(layer):
            spike_times[layer_neuron, layer] = Int(f"dSpkTime_{layer_neuron}_{layer}")
    return spike_times


def gen_weights(weights_list: TWeightList) -> TWeight:
    """Convert a *concrete* ``weights_list`` into a dictionary that is easy to
    index during encoding.

    Unlike spike-times, the weights are plain Python ``float``\ s—they are
    *constants* in the verification query.  Each entry is addressed by the
    triple ``(inNeuron, outNeuronIndex, inLayer)`` to match the mathematical
    notation in the paper.


    Returns
    -------
    TWeight
        ``dict`` mapping ``(inNeuronTuple, outNeuronIdx, inLayer)`` → ``float``.

    Example
    -------
    >>> from utils.encoding_mnist import gen_weights
    >>> w = gen_weights([layer0_weights, layer1_weights])
    >>> w[((0, 0), 10, 0)]  # weight from pixel (0,0) to output neuron 10
    0.0354
    """
    weights = typecast(TWeight, {})
    print(num_steps, weights_list[0].shape, weights_list[1].shape)
    for in_layer in range(len(n_layer_neurons) - 1):
        layer_weight = weights_list[in_layer]
        out_layer = in_layer + 1
        for out_neuron in range(n_layer_neurons[out_layer]):
            in_neurons = get_layer_neurons_iter(in_layer)
            for in_neuron in in_neurons:
                weights[in_neuron, out_neuron, in_layer] = float(
                    layer_weight[out_neuron, *in_neuron]
                )
    info("Weights are generated.")
    return weights


def gen_node_eqns(weights: TWeight, spike_times: TSpikeTime) -> List[Union[BoolRef, bool]]:
    node_eqn: List[Union[BoolRef, bool]] = []
    for layer, _ in enumerate(n_layer_neurons):
        for neuron in tqdm(get_layer_neurons_iter(layer)):
            # out layer cannot spike in first "layer" steps.
            node_eqn.extend(
                [
                    spike_times[neuron, layer] >= layer,
                    spike_times[neuron, layer] <= num_steps - 1,
                ]
            )

    for i, layer_neurons in enumerate(n_layer_neurons[1:], start=1):
        in_layer = i - 1
        for out_neuron_pos in tqdm(
            range(layer_neurons), desc="Generating node equations. Nodes"
        ):
            out_neuron = (
                out_neuron_pos,
                0,
            )  # We only use position 0 in dimension 1 for layer output.
            flag: List[Union[BoolRef, bool]] = [False]
            # Does not include last step: [0,num_steps-1]
            for timestep in tqdm(
                range(i, num_steps - 1), desc="Timestep", leave=False
            ):
                time_cumulated_potential = []
                for in_neuron in get_layer_neurons_iter(in_layer):
                    time_cumulated_potential.append(
                        If(
                            # can keep a running sum here with maps to see if all have it. saves time
                            spike_times[in_neuron, in_layer] <= (timestep - 1),
                            weights[in_neuron, out_neuron_pos, in_layer],

                            0,
                        )
                    )
                over_threshold = Sum(time_cumulated_potential) >= threshold
                spike_condition = And(Not(Or(flag)), over_threshold)
                flag.append(over_threshold)
                term = typecast(
                    BoolRef,
                    spike_condition == (spike_times[out_neuron, i] == timestep),
                )
                node_eqn.append(term)
            # Force spike in last timestep.
            term = typecast(
                BoolRef,
                Not(Or(flag)) == (spike_times[out_neuron, i] == num_steps - 1),
            )
            node_eqn.append(term)
    info("Node equations are generated.")
    return node_eqn


def gen_input_property_z3(spike_times: TSpikeTime, img: TImage, delta: int) -> BoolRef:
    """Generates the Z3 input property constraint (L1 norm)."""
    input_layer = 0
    delta_pos = 0
    delta_neg = 0

    def relu(x):
        return If(x > 0, x, 0)

    for in_neuron in get_layer_neurons_iter(input_layer):
        delta_pos += relu(spike_times[in_neuron, input_layer] - int(img[in_neuron]))
        delta_neg += relu(int(img[in_neuron]) - spike_times[in_neuron, input_layer])
    return (delta_pos + delta_neg) <= delta


def gen_output_property_z3(spike_times: TSpikeTime, orig_pred: int) -> BoolRef:
    """Generates the Z3 output property for adversarial robustness."""
    op = []
    last_layer = len(n_layer_neurons) - 1
    orig_neuron = (orig_pred, 0)
    for out_neuron in get_layer_neurons_iter(last_layer):
        if out_neuron != orig_neuron:
            op.append(
                spike_times[out_neuron, last_layer]
                <= spike_times[orig_neuron, last_layer]
            )
    return Or(op)