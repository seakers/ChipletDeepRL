import numpy as np
from copy import deepcopy


class Chiplet_Configuration_Design():

    def __init__(self, component_list, structure_panel):
        self.problem_name = 'Chiplet_Configuration'
        self.n_objectives = 5
        self.n_constraints = 3

        self.base_components = component_list
        self.num_components = len(component_list)
        self.base_panel = structure_panel
        self.max_shelves = 6

        self.design_space = [
            {'type': 'discrete', 'range': ['triangle', 'rectangle', 'hexagon']}, # shape of structural base
            {'type': 'continuous', 'range': [0.01, 2]}, # structure x dim (m)
            {'type': 'continuous', 'range': [0.01, 2]}, # structure y dim (m)
            {'type': 'continuous', 'range': [0.01, 2]}, # structure z dim (m)
            {'type': 'discrete', 'range': [i for i in range(self.max_shelves + 1)]}, # number of shelves
        ]

        for _ in range(self.num_components):
            self.design_space.append({'type': 'discrete', 'range': [i for i in range(8 + self.max_shelves)]}) # Panel choice. 14 options max, constraint will be violated for choosing a nonexistant panel
            self.design_space.append({'type': 'continuous', 'range': [-1, 1]}) # Component x loc
            self.design_space.append({'type': 'continuous', 'range': [-1, 1]}) # Component y loc
            self.design_space.append({'type': 'discrete', 'range': [i for i in range(6)]}) # choose which face to attach to panel
            self.design_space.append({'type': 'continuous', 'range': [0, 2*np.pi]}) # rotation of component about normal of panel

        self.n_variables = len(self.design_space)


    def triangle_prism_shell(self, structure_panels, x_dim, y_dim, z_dim):
        # create the 5 panels needed for a triangular prism shell
        # bottom panel
        structure_panels[0] = deepcopy(self.base_panel)
        structure_panels[0].dimensions = [x_dim, y_dim, structure_panels[0].thickness]
        structure_panels[0].location = [0, 0, -z_dim/2]

        # top panel
        structure_panels[1] = deepcopy(self.base_panel)
        structure_panels[1].dimensions = [x_dim, y_dim, structure_panels[1].thickness]
        structure_panels[1].location = [0, 0, z_dim/2]

        # side panels
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [x_dim, z_dim, structure_panels[2].thickness]
        structure_panels[2].location = [0, y_dim/2, 0]
        structure_panels[2].orientation = np.matrix([[0,0,1],[1,0,0],[0,1,0]])

        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [y_dim, z_dim, structure_panels[3].thickness]
        structure_panels[3].location = [-x_dim/2, -y_dim/4, 0]
        structure_panels[3].orientation = np.matrix([[np.sqrt(3)/2, 1/2, 0],[-1/2, np.sqrt(3)/2, 0],[0,0,1]])

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [y_dim, z_dim, structure_panels[4].thickness]
        structure_panels[4].location = [x_dim/2, -y_dim/4, 0]
        structure_panels[4].orientation = np.matrix([[-np.sqrt(3)/2, 1/2, 0],[-1/2, -np.sqrt(3)/2, 0],[0,0,1]])

        return structure_panels
    

    def rectangular_prism_shell(self, structure_panels, x_dim, y_dim, z_dim):
        # create the 6 panels needed for a rectangular prism shell
        # bottom panel
        structure_panels[0] = deepcopy(self.base_panel)
        structure_panels[0].dimensions = [x_dim, y_dim, structure_panels[0].thickness]
        structure_panels[0].location = [0, 0, -z_dim/2]

        # top panel
        structure_panels[1] = deepcopy(self.base_panel)
        structure_panels[1].dimensions = [x_dim, y_dim, structure_panels[1].thickness]
        structure_panels[1].location = [0, 0, z_dim/2]

        # side panels
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [x_dim, z_dim, structure_panels[2].thickness]
        structure_panels[2].location = [0, y_dim/2, 0]
        structure_panels[2].orientation = np.matrix([[0,0,1],[1,0,0],[0,1,0]])

        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [y_dim, z_dim, structure_panels[3].thickness]
        structure_panels[3].location = [-x_dim/2, 0, 0]
        structure_panels[3].orientation = np.matrix([[0,1,0],[0,0,1],[1,0,0]])

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [x_dim, z_dim, structure_panels[4].thickness]
        structure_panels[4].location = [0, -y_dim/2, 0]
        structure_panels[4].orientation = np.matrix([[0,0,-1],[1,0,0],[0,-1,0]])

        structure_panels[5] = deepcopy(self.base_panel)
        structure_panels[5].dimensions = [y_dim, z_dim, structure_panels[5].thickness]
        structure_panels[5].location = [x_dim/2, 0, 0]
        structure_panels[5].orientation = np.matrix([[0,-1,0],[0,0,1],[-1,0,0]])

        return structure_panels

    
    def hexagonal_prism_shell(self, structure_panels, x_dim, y_dim, z_dim):
        # create the 8 panels needed for a hexagonal prism shell
        # bottom panel
        structure_panels[0] = deepcopy(self.base_panel)
        structure_panels[0].dimensions = [x_dim, y_dim, structure_panels[0].thickness]
        structure_panels[0].location = [0, 0, -z_dim/2]

        # top panel
        structure_panels[1] = deepcopy(self.base_panel)
        structure_panels[1].dimensions = [x_dim, y_dim, structure_panels[1].thickness]
        structure_panels[1].location = [0, 0, z_dim/2]

        # side panels
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [x_dim, z_dim, structure_panels[2].thickness]
        structure_panels[2].location = [0, y_dim/2, 0]
        structure_panels[2].orientation = np.matrix([[0,0,1],[1,0,0],[0,1,0]])

        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [y_dim, z_dim, structure_panels[3].thickness]
        structure_panels[3].location = [-x_dim/2, y_dim/4, 0]
        structure_panels[3].orientation = np.matrix([[np.sqrt(3)/2, 1/2, 0],[-1/2, np.sqrt(3)/2, 0],[0,0,1]])

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [y_dim, z_dim, structure_panels[4].thickness]
        structure_panels[4].location = [-x_dim/2, -y_dim/4, 0]
        structure_panels[4].orientation = np.matrix([[np.sqrt(3)/2, -1/2, 0],[1/2, np.sqrt(3)/2, 0],[0,0,1]])

        structure_panels[5] = deepcopy(self.base_panel)
        structure_panels[5].dimensions = [x_dim, z_dim, structure_panels[5].thickness]
        structure_panels[5].location = [0, -y_dim/2, 0]
        structure_panels[5].orientation = np.matrix([[0,0,-1],[1,0,0],[0,-1,0]])

        structure_panels[6] = deepcopy(self.base_panel)
        structure_panels[6].dimensions = [y_dim, z_dim, structure_panels[6].thickness]
        structure_panels[6].location = [x_dim/2, -y_dim/4, 0]
        structure_panels[6].orientation = np.matrix([[-np.sqrt(3)/2, -1/2, 0],[-1/2, np.sqrt(3)/2, 0],[0,0,1]])

        structure_panels[7] = deepcopy(self.base_panel)
        structure_panels[7].dimensions = [y_dim, z_dim, structure_panels[7].thickness]
        structure_panels[7].location = [x_dim/2, y_dim/4, 0]
        structure_panels[7].orientation = np.matrix([[-np.sqrt(3)/2, 1/2, 0],[1/2, -np.sqrt(3)/2, 0],[0,0,1]])

        return structure_panels
    

    def evaluate(self, x):

        structure_panels = [None] * (8 + self.max_shelves)

        if x[0] == 'triangle':
            structure_panels = self.triangle_prism_shell(structure_panels, x[1], x[2], x[3])
        elif x[0] == 'rectangle':
            structure_panels = self.rectangular_prism_shell(structure_panels, x[1], x[2], x[3])
        elif x[0] == 'hexagon':
            structure_panels = self.hexagonal_prism_shell(structure_panels, x[1], x[2], x[3])
        else:
            raise ValueError(f"INVALID STRUCTURE SHAPE {x[0]}")
    


# class Speed_Reducer_Design():
#     '''
#     RE3-7-5 from https://arxiv.org/pdf/2009.12867
#     Originally from https://link.springer.com/article/10.1007/s00158-002-0247-6

#     Used as an example of a real world moo problem
#     '''
#     def __init__(self):
#         self.problem_name = 'RE35'
#         self.n_objectives = 3
#         self.n_variables = 7
#         self.n_constraints = 0
#         self.n_original_constraints = 11

#         self.design_space = [
#             {'type': 'continuous', 'range': [2.6, 3.6]},
#             {'type': 'continuous', 'range': [0.7, 0.8]},
#             {'type': 'discrete', 'range': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
#             {'type': 'continuous', 'range': [7.3, 8.3]},
#             {'type': 'continuous', 'range': [7.3, 8.3]},
#             {'type': 'continuous', 'range': [2.9, 3.9]},
#             {'type': 'continuous', 'range': [5.0, 5.5]},
#         ]
#         # self.lbound = np.zeros(self.n_variables)
#         # self.ubound = np.zeros(self.n_variables)
#         # self.lbound[0] = 2.6
#         # self.lbound[1] = 0.7
#         # self.lbound[2] = 17
#         # self.lbound[3] = 7.3
#         # self.lbound[4] = 7.3
#         # self.lbound[5] = 2.9
#         # self.lbound[6] = 5.0    
#         # self.ubound[0] = 3.6
#         # self.ubound[1] = 0.8
#         # self.ubound[2] = 28
#         # self.ubound[3] = 8.3
#         # self.ubound[4] = 8.3
#         # self.ubound[5] = 3.9
#         # self.ubound[6] = 5.5
        
#     def evaluate(self, x):
#         f = np.zeros(self.n_objectives)
#         g = np.zeros(self.n_original_constraints)

#         x1 = x[0]
#         x2 = x[1]
#         x3 = np.round(x[2])
#         x4 = x[3]
#         x5 = x[4]
#         x6 = x[5]
#         x7 = x[6]

#         # First original objective function (weight)
#         f[0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    
#         # Second original objective function (stress)
#         tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
#         f[1] =  np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

#         # Constraint functions 	
#         g[0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
#         g[1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
#         g[2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
#         g[3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
#         g[4] = -(x2 * x3) + 40.0
#         g[5] = -(x1 / x2) + 12.0
#         g[6] = -5.0 + (x1 / x2)
#         g[7] = -1.9 + x4 - 1.5 * x6
#         g[8] = -1.9 + x5 - 1.1 * x7
#         g[9] =  -f[1] + 1300.0
#         tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
#         g[10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0	
#         g = np.where(g < 0, -g, 0)                
#         f[2] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]
    
#         return f