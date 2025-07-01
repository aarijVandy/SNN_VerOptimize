from copy import deepcopy
from multiprocessing import Pool
from random import sample as random_sample
from random import seed
import time, logging, typing
import numpy as np
from collections.abc import Generator
from z3 import *
from STL_encoding import *

from utils.dictionary_mnist import *
from utils.encoding_mnist import *
from utils.config import CFG
from utils.debug import info
from utils.mnist_net import forward, backward, test_weights, prepare_weights
from mnist import MNIST
from functools import partial


### TODO: can make all the check samples into a separate class / file and call from there with choice of solver
class Model():
        def __init__(self, weights_list):
            self.input_neurons = get_layer_neurons_iter(0)
            self.output_neurons = get_layer_neurons_iter(len(n_layer_neurons) - 1)
            self.weights = gen_weights(weights_list)

def load_mnist() -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    # Parameter setting
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading MNIST dataset
    mndata = MNIST("data/mnist/raw/")

    Images, Labels = mndata.load_training()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images_test.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    # images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test

def get_samples(num_samples, images, weights_list):

        samples_no_list: list[int] = []
        sampled_imgs: list[TImage] = []
        orig_preds: list[int] = []
        for sample_no in random_sample([*range(len(images))], k= num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img: TImage = images[sample_no]
            sampled_imgs.append(img)  # type: ignore
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")

        return zip(samples_no_list, sampled_imgs, orig_preds)


def run_test(cfg: CFG):
    log_name = f"{cfg.log_name}_{num_steps}_{'_'.join(str(l) for l in n_layer_neurons)}_delta{cfg.deltas}.log"
    logging.basicConfig(filename="log/" + log_name, level=logging.INFO)
    info(cfg)

    seed(cfg.seed)
    np.random.seed(cfg.seed)
    weights_list = prepare_weights(subtype="mnist", load_data_func=load_mnist)
    images, labels, *_ = load_mnist()

    info("Data is loaded")

    
    if cfg.z3:
        S = Solver()
        spike_times = gen_spike_times()
        weights = gen_weights(weights_list)

        # Load equations.
        eqn_path = (
            f"eqn/eqn_{num_steps}_{'_'.join([str(i) for i in n_layer_neurons])}.txt"
        )
        if not load_expr or not os.path.isfile(eqn_path):
            node_eqns = gen_node_eqns(weights, spike_times)
            S.add(node_eqns)
            if save_expr:
                try:
                    with open(eqn_path, "w") as f:
                        f.write(S.sexpr())
                        info("Node equations are saved.")
                except:
                    pdb.set_trace(header="Failed to save node eqns.")
        else:
            S.from_file(eqn_path)
        info("Solver is loaded.")

        def check_sample_z3(sample: tuple[int, TImage, int], delta = 1):
            sample_no, img, orig_pred = sample
            orig_neuron = (orig_pred, 0)
            tx = time.time()

            # Input property terms
            prop: list[BoolRef] = []
            input_layer = 0
            delta_pos = IntVal(0)
            delta_neg = IntVal(0)

            def relu(x: Any):
                return If(x > 0, x, 0)

                # find L1 distance between spike times and the image val
            for in_neuron in get_layer_neurons_iter(input_layer):
                # Try to avoid using abs, as it makes z3 extremely slow.
                delta_pos += relu(
                    spike_times[in_neuron, input_layer] - int(img[in_neuron])
                )
                delta_neg += relu(
                    int(img[in_neuron]) - spike_times[in_neuron, input_layer]
                )
            prop.append((delta_pos + delta_neg) <= delta)
            info(f"Inputs Property Done in {time.time() - tx} sec")

            # Output property
            tx = time.time()
            op = []
            last_layer = len(n_layer_neurons) - 1
            for out_neuron in get_layer_neurons_iter(last_layer):
                if out_neuron != orig_neuron:
                    # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                    # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                    # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                    op.append(
                        spike_times[out_neuron, last_layer]
                        <= spike_times[orig_neuron, last_layer]
                    )
            op = Or(op)
            info(f"Output Property Done in {time.time() - tx} sec")

            tx = time.time()
            S_instance = deepcopy(S)
            info(f"Network Encoding read in {time.time() - tx} sec")
            S_instance.add(op)  # type: ignore
            S_instance.add(prop)  # type: ignore
            info(f"Total model ready in {time.time() - tx}")

            info("Query processing starts")
            # set_param(verbose=2)
            # set_param("parallel.enable", True)
            tx = time.time()
            result = S_instance.check()  # type: ignore
            info(f"Checking done in time {time.time() - tx}")
            if result == sat:
                info(f"Not robust for sample {sample_no} and delta={delta}")
            elif result == unsat:
                info(f"Robust for sample {sample_no} and delta={delta}")
            else:
                info(
                    f"Unknown at sample {sample_no} for reason {S_instance.reason_unknown()}"
                )
            info("")
            return result

    
        # For each delta
        for delta in cfg.deltas:
            
            samples = get_samples(cfg.num_samples, images, weights_list)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_z3, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_z3(sample, delta)

        info("")
    elif cfg.marabou: 
        
        spike_times_map = {}
        model = Model(weights_list)
        S = Convert_to_Marabou(model=model)
        weights = model.weights

        # Load equations onto solver
        network = S.encode_marabou(spike_times_map=spike_times_map)
        info("Contraint equations generated")


        # get samples and collect them
        samples = get_samples(cfg.num_samples, images, weights_list)

        # helper
        def _relu(net: MarabouCore.InputQuery, x: int) -> int:
            """Return a fresh var y with y = ReLU(x)."""
            y = net.getNewVariable()
            net.addRelu(x, y)
            return y
            
            ## check_sampe_marabou was here


        def check_sample_marabou(sample: tuple[int,TImage, int], conv, delta = 1):
            """
            TODO: need to explain what conv is
            returns sat for counter example found. unsat for proven robustness. unk for undecidable
            """
            sample, img, orig_pred = sample
            orig_neuron = (orig_pred, 0)
            

            net = MarabouCore.InputQuery()    # reminder: net is of instance marabou solver, network (locally conv ) 
                                              # encodes the constraints in marabou terms
            # encode STL specs

            for layer in range(len(n_layer_neurons)):
                for neuron in get_layer_neurons_iter(layer):
                    v = net.getNewVariable()
                    spike_times_map[(neuron, layer)] = v
            conv.encode_marabou(spike_times_map).transferTo(net)
        

            tx = time.time()
            # L1 norm check
            deltas = []
            input_layer = 0
            for neuron in get_layer_neurons_iter(layer):
                s_i = spike_times_map[(in_neuron, input_layer)]
                eps= int(img[neuron])
                
                # d_pos = ReLU(s_i − const), d_neg = ReLU(const − s_i)
                x_pos  = net.getNewVariable()
                x_neg  = net.getNewVariable()
                net.addEquation(Convert_to_Marabou._eq(x_pos, 0.0))   # initialise (will be overridden by ReLU)
                net.addEquation(Convert_to_Marabou._eq(x_neg, 0.0))

                d_pos = _relu(net, net.addEqualityFromScratch(s_i, eps, 1.0, -1.0))
                d_neg = _relu(net, net.addEqualityFromScratch(eps, s_i, 1.0, -1.0))

                # Fix the outputs to the ReLU variables
                net.addEquation(Convert_to_Marabou._eq(x_pos, d_pos))
                net.addEquation(Convert_to_Marabou._eq(x_neg, d_neg))
                deltas.extend([d_pos, d_neg])

            lhs = net.getNewVariable()
            sum_eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
            sum_eq.addAddend(-1.0, lhs)
            # Sum all pos & neg deltas from the original input spike times to encode the L1-norm constraint on the input perturbation
            # ensuring the total change does not exceed delta.
            for d in deltas:
                sum_eq.addAddend(1.0, d)
            sum_eq.setScalar(0.0)
            net.addEquation(sum_eq)
            net.addEquation(Convert_to_Marabou._le(lhs, float(delta)))
            net.setLowerBound(lhs, 0.0)
            info(f"Input property encoded in time {time.time() - tx} sec")

            tx = time.time()
            # output property for neruons firing < t_orig_neuron
            final_layer = len(n_layer_neurons) - 1
            orig_spike_time = spike_times_map[(orig_neuron, final_layer)]

            disjuncts = []
            for out_neuron in get_layer_neurons_iter(last_layer):
                if out_neuron == orig_neuron:
                    continue
                s_k = spike_times_map[(out_neuron, spike_times_map)]
                # Case k: s_k ≤ s_orig − 1e-6
                disjuncts.append([Convert_to_Marabou._le(s_k, -1e-6)])  # will be shifted below

                # Linearise  s_k − s_orig ≤ -1e-6
                ineq = Convert_to_Marabou._le(s_k, -1e-6)           
                ineq.addAddend(-1.0, orig_spike_time)         # difference s_k  − orig_spike  ≤ -1e-6
                disjuncts[-1] = [ineq]
            
            net.addDisjuct(disjuncts)
            info(f"Output property encoded in time {time.time() - tx} sec")

            tx = time.time()
            vals, stats = net.solve()
            info(f"Marabou solve-time  {time.time() - tx:6.2f}  sec")

            if vals is not None:
                info(f"NOT robust for sample {sample_no} (counter-example found)")
            elif stats["result"] == "unsat":
                info(f"Robust for sample {sample_no}")
            else:
                why = stats.get("result", "unknown")
                info(f"Unknown for sample {sample_no}  (reason: {why})")
       

        for delta in cfg.deltas:
            
            samples = get_samples(cfg.num_samples, images, weights_list)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(partial(check_sample_marabou, conv = conv , delta = delta),  samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_marabou(sample, delta)
         
    else:
        # Recursively find available adversarial attacks.
        def search_perts(
            img: TImage, delta: int, loc: int = 0, pert: TImage | None = None
        ) -> Generator[TImage, None, None]:
            # Initial case
            if pert is None: 
                pert = np.zeros_like(img, dtype=img.dtype)

            # Last case
            if delta == 0:
                yield img + pert
            # Search must be terminated at the end of image.
            elif loc < n_layer_neurons[0]:
                loc_2d = (loc // layer_shapes[0][1], loc % layer_shapes[0][1])
                orig_time = int(img[loc_2d])
                # Clamp delta at current location
                available_deltas = range(
                    -min(orig_time, delta), min((num_steps - 1) - orig_time, delta) + 1
                )
                for delta_at_neuron in available_deltas:
                    new_pert = pert.copy()
                    new_pert[loc_2d] += delta_at_neuron
                    yield from search_perts(
                        img, delta - abs(delta_at_neuron), loc + 1, new_pert
                    )

        samples_no_list: list[int] = []
        sampled_imgs: list[TImage] = []
        sampled_labels: list[int] = []
        orig_preds: list[int] = []
        for sample_no in random_sample([*range(len(images))], k=cfg.num_samples):
            info(f"sample {sample_no} is drawn.")
            samples_no_list.append(sample_no)
            img: TImage = images[sample_no]
            label = labels[sample_no]
            sampled_imgs.append(img)
            sampled_labels.append(label)
            orig_preds.append(forward(weights_list, img))
        info(f"Sampling is completed with {num_procs} samples.")


        def check_sample_non_smt(
            sample: tuple[int, TImage, int, int],
            adv_train: bool = False,
            weights_list: TWeightList = weights_list,
            delta = 1
        ):
            sample_no, img, label, orig_pred = sample

            info("Query processing starts")
            tx = time.time()
            sat_flag: bool = False
            adv_spk_times: list[list[np.ndarray[Any, np.dtype[np.float64]]]] = []
            n_counterexamples = 0
            for pertd_img in search_perts(img, delta):
                pert_pred = forward(weights_list, pertd_img, spk_times := [])
                adv_spk_times.append(spk_times)
                last_layer_spk_times = spk_times[-1]
                not_orig_mask = [
                    x for x in range(n_layer_neurons[-1]) if x != pert_pred
                ]
                # It is equal to Not(spike_times[out_neuron, last_layer] >= spike_times[orig_neuron, last_layer]),
                # we are checking p and Not(q) and q = And(q1, q2, ..., qn)
                # so Not(q) is Or(Not(q1), Not(q2), ..., Not(qn))
                if np.any(
                    last_layer_spk_times[not_orig_mask]
                    <= last_layer_spk_times[orig_pred]
                ):
                    sat_flag = True
                    if not adv_train:
                        break
                    n_counterexamples += 1
            info(f"Checking done in time {time.time() - tx}")
            if sat_flag:
                if adv_train:
                    info(
                        f"Not robust for sample {sample_no} and delta={delta} with {n_counterexamples} counterexamples."
                    )
                    info(f"Start adversarial training.")
                    updated_weights_list = weights_list
                    for spk_times in adv_spk_times:
                        updated_weights_list = backward(
                            updated_weights_list, spk_times, img, label
                        )
                    test_weights(updated_weights_list, load_mnist)
                    new_orig_pred = forward(updated_weights_list, img)
                    new_sample = (*sample[:3], new_orig_pred)
                    info(
                        f"Completed adversarial training. Checking robustness again."
                    )
                    check_sample_non_smt(
                        new_sample,
                        adv_train=False,
                        weights_list=updated_weights_list,
                    )
                else:
                    info(f"Not robust for sample {sample_no} and delta={delta}")
            elif sat_flag == False:
                info(f"Robust for sample {sample_no} and delta={delta}.")
            info("")
            return sat_flag

        # For each delta
        for delta in cfg.deltas:
            samples = zip(samples_no_list, sampled_imgs, sampled_labels, orig_preds)
            if mp:
                with Pool(num_procs) as pool:
                    pool.map(check_sample_non_smt, samples)
                    pool.close()
                    pool.join()
            else:
                for sample in samples:
                    check_sample_non_smt(sample, delta = delta)

        info("")


# %%
