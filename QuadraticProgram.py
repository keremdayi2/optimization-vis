import numpy as np
import matplotlib.pyplot as plt

class QuadraticProgam:
    def __init__(self):
        self.objective = []
        self.ineq = []
        self.eq = []

    def add_objective(self):
        pass

    def add_equality_constraint(self):
        raise NotImplementedError

    def add_inequality_constraint(self, constraint):
        raise NotImplementedError

    