import numpy as np
import matplotlib.pyplot as plt

class QuadraticProgram:
    def __init__(self):
        self.objective = [] # P, q, r
        self.ineq = [] # tuples (ci,di)
        self.eq = [] # tuples (ai, bi)

    def add_objective(self):
        pass

    def add_equality_constraint(self, expression):
        raise NotImplementedError

    def add_inequality_constraint(self, expression):
        raise NotImplementedError

    