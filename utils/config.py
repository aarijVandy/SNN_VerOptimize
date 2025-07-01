from dataclasses import dataclass

@dataclass
class CFG:
    log_name:str = ""
    seed:int = 42
    #Delta-perturbation params
    num_samples:int = 14
    deltas:tuple[int,...] = (1,)
    z3:bool = False
    marabou: bool = False
    solver:str="z3"

    @property
    def use_z3(self):     return self.solver=="z3"

class SolverIF:
    def add(self, *cnstr): ...
    def check(self): ...
    def is_sat(self): ...

def make_solver(kind:str):
    if kind=="z3":
        from z3 import Solver, sat, unsat
        class Z3Wrap(SolverIF, Solver):      # plain inheritance
            def is_sat(self,res): return res==sat
        return Z3Wrap(), sat, unsat
    elif kind=="marabou":
        from maraboupy import Marabou, MarabouCore
        class MWrap(SolverIF):
            def __init__(self):
                self.q = MarabouCore.InputQuery()
            def add(self, eq): ...          # translate eq -> MarabouCore.Equation
            def check(self):
                return Marabou.solve(self.q)
            def is_sat(self,res): return res[0]
        return MWrap(), True, False