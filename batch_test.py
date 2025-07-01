from typing import Literal
from utils import *
from adv_rob_iris_module import run_test as run_test_iris
from adv_rob_mnist_module import run_test as run_test_mnist
from adv_rob_fmnist_module import run_test as run_test_fmnist
from numpy import arange
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from time import strftime, localtime

required_arguments:list[str] = "test_type prefix".split()

TestType = Literal["iris", "mnist", "fmnist"]
def run_test(cfg:CFG, test_type:TestType="mnist"):
    if test_type == "iris":
        return run_test_iris(cfg)
    elif test_type == "mnist":
        return run_test_mnist(cfg)
    elif test_type == "fmnist":
        return run_test_fmnist(cfg)
    else:
        raise NotImplementedError(f"Test type must be in {TestType}.")


def parse():
    parser = ArgumentParser()
    parser.add_argument("-p", "--prefix", dest="prefix", type=str)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--delta-max", dest="delta_max", type=int, default=1)
    parser.add_argument("--repeat", dest="repeat", type=int, default=1)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=14)
    parser.add_argument("--test-type", dest="test_type", type=str)

    solver = parser.add_mutually_exclusive_group()
    solver.add_argument("--z3",      dest="solver", action="store_const",
                        const="z3",      help="use the Z3 solver (default)")
    solver.add_argument("--marabou", dest="solver", action="store_const",
                        const="marabou", help="use the Marabou solver")
    parser.set_defaults(solver="z3")   # fallback
    
    return parser.parse_args()

def prepare_log_name(parser:Namespace) -> str:
    words = [strftime('%m%d%H%M', localtime())]   
    if getattr(parser, "repeat") > 1: words.append(f"rep_{parser.repeat}")
    if parser.test_type is not None:
        words.append(parser.test_type)
    if parser.prefix is not None:
        words.append(parser.prefix)
    # if hasattr(parser, "test_type"): words.append(parser.test_type)
    # if hasattr(parser, "prefix"): words.append(parser.prefix)

    words.append(parser.solver)
    return '_'.join(words)

if __name__ == "__main__":
    import multiprocessing as mp

    parser = parse()

    if getattr(parser, "repeat") < 1:
        raise ValueError("repeat must be greater than 0.")
    

    # change this here to move to marabou instead
    if all(hasattr(parser, s) for s in required_arguments):
        for iteration in range(parser.repeat):
            mp.set_start_method("fork")
            run_test(CFG(log_name=prepare_log_name(parser),
                        seed=parser.seed,
                        num_samples=parser.num_samples,
                        deltas=(parser.delta_max,),
                        z3=parser.solver == "z3",
                        marabou = parser.solver == "marabou",
                        solver= parser.solver),
                    test_type=parser.test_type)
    else:
        raise ValueError("Not appropriate arguments.")
