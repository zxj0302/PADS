import networkx as nx
import numpy as np
import math
from .rwc_jit import rwc

class MitigationComparison:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class MitigationMeasurement:
    def __init__(self, G, opinion_list):
        self.G = G
        self.op = np.array(opinion_list)

    def opinion_variance(self):
        return np.var(self.op)

    def opinion_mean(self):
        return np.mean(self.op)

    def opinion_gap(self):
        return np.mean(self.op[self.op > 0]) - np.mean(self.op[self.op < 0])

    def opinion_controversy(self):
        return np.sum(self.op ** 2)

    def RWC(self):
        return rwc(self.G)

class MitigationStrategy:
    def __init__(self, name, description):
        self.name = name
        self.description = description
