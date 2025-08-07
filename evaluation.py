import numpy as np

class Speed_Reducer_Design():
    '''
    RE3-7-5 from https://arxiv.org/pdf/2009.12867
    Originally from https://link.springer.com/article/10.1007/s00158-002-0247-6

    Used as an example of a real world moo problem
    '''
    def __init__(self):
        self.problem_name = 'RE35'
        self.n_objectives = 3
        self.n_variables = 7
        self.n_constraints = 0
        self.n_original_constraints = 11

        self.design_space = [
            {'type': 'continuous', 'range': [2.6, 3.6]},
            {'type': 'continuous', 'range': [0.7, 0.8]},
            {'type': 'discrete', 'range': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
            {'type': 'continuous', 'range': [7.3, 8.3]},
            {'type': 'continuous', 'range': [7.3, 8.3]},
            {'type': 'continuous', 'range': [2.9, 3.9]},
            {'type': 'continuous', 'range': [5.0, 5.5]},
        ]
        # self.lbound = np.zeros(self.n_variables)
        # self.ubound = np.zeros(self.n_variables)
        # self.lbound[0] = 2.6
        # self.lbound[1] = 0.7
        # self.lbound[2] = 17
        # self.lbound[3] = 7.3
        # self.lbound[4] = 7.3
        # self.lbound[5] = 2.9
        # self.lbound[6] = 5.0    
        # self.ubound[0] = 3.6
        # self.ubound[1] = 0.8
        # self.ubound[2] = 28
        # self.ubound[3] = 8.3
        # self.ubound[4] = 8.3
        # self.ubound[5] = 3.9
        # self.ubound[6] = 5.5
        
    def evaluate(self, x):
        f = np.zeros(self.n_objectives)
        g = np.zeros(self.n_original_constraints)

        x1 = x[0]
        x2 = x[1]
        x3 = np.round(x[2])
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]

        # First original objective function (weight)
        f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    
        # Second original objective function (stress)
        tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
        f[1] =  np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions 	
        g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g[4] = -(x2 * x3) + 40.0
        g[5] = -(x1 / x2) + 12.0
        g[6] = -5.0 + (x1 / x2)
        g[7] = -1.9 + x4 - 1.5 * x6
        g[8] = -1.9 + x5 - 1.1 * x7
        g[9] =  -f[1] + 1300.0
        tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0	
        g = np.where(g < 0, -g, 0)                
        f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]
    
        return f