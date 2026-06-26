import numpy as np
from copy import deepcopy

from utils.config_utils import faceAndRot2DCM, align_cuboid_with_plane
import itertools
import itertools
import itertools
from collections import deque


class Chiplet_Configuration_Design():

    def __init__(self, component_list, structure_panel):
        
        self.problem_name = 'Chiplet_Configuration'
        self.num_objectives = 5
        self.num_constraints = 3

        self.component_list = component_list
        self.num_components = len(component_list)
        self.base_panel = structure_panel
        self.max_shelves = 3

        self.design_space = [
            {'type': 'discrete', 'range': [0, 1, 2]}, # shape of structural base, trianle, rectangle, hexagon
            {'type': 'continuous', 'range': [0.01, 2]}, # structure x dim (m)
            {'type': 'continuous', 'range': [0.01, 2]}, # structure y dim (m)
            {'type': 'continuous', 'range': [0.01, 2]}, # structure z dim (m)
            {'type': 'discrete', 'range': [i for i in range(self.max_shelves + 1)]}, # number of shelves
        ]

        for i in range(self.num_components):
            self.design_space.append({'type': 'discrete', 'range': [i for i in range(8 + 2*self.max_shelves)]}) # Panel choice. Max 8 for shell, and 10 for shelves (one for each side of each shelf). 18 options max, constraint will be violated for choosing a nonexistant panel
            self.design_space.append({'type': 'continuous', 'range': [-1, 1]}) # Component x loc
            self.design_space.append({'type': 'continuous', 'range': [-1, 1]}) # Component y loc
            self.design_space.append({'type': 'discrete', 'range': [i for i in range(6)]}) # choose which face to attach to panel
            self.design_space.append({'type': 'continuous', 'range': [0, 2*np.pi]}) # rotation of component about normal of panel
            if i == 0:
                self.unique_des_space = deepcopy(self.design_space)

        self.n_variables = len(self.design_space)


    def center_mass_cost(self, locations, masses, struct_locs, struct_masses):
        
        # Calculate the distance of the center of mass from the circumcenter of the structure

        all_locs = locations + struct_locs
        all_masses = masses + struct_masses

        all_locs = np.array(all_locs)
        all_masses = np.transpose(np.array([all_masses, all_masses, all_masses]))

        center_mass = np.sum(all_locs * all_masses, 0) / np.sum(all_masses, 0)

        dist = np.sqrt(np.sum(np.square(center_mass)))
        return dist
    

    def inertia_cost(self, dimensions, locations, masses):
        
        # Calculate the moment of inertia (simplified)
        inertia = np.zeros((3, 3))
        for i in range(len(dimensions)):
            Icm = 1 / 12 * masses[i] * np.array([
                [(dimensions[i][1] ** 2 + dimensions[i][2] ** 2), 0, 0],
                [0, (dimensions[i][0] ** 2 + dimensions[i][2] ** 2), 0],
                [0, 0, (dimensions[i][0] ** 2 + dimensions[i][1] ** 2)]
            ])
            skew_sym = np.array([
                [0, -locations[i][2], locations[i][1]],
                [locations[i][2], 0, -locations[i][0]],
                [-locations[i][1], locations[i][0], 0]
            ])
            Ir = masses[i] * np.matmul(skew_sym, skew_sym)
            inertia += Icm - Ir

        off_axis_inertia = abs(inertia[0, 1]) + abs(inertia[0, 2]) + abs(inertia[1, 2])
        on_axis_inertia = inertia[0, 0] + inertia[1, 1] + inertia[2, 2]
        return off_axis_inertia, on_axis_inertia


    def overlap_cost(self, dimensions, locations):
        
        overlap = 0
        element_corners = []
        for i in range(len(dimensions)):
            min_corner = [locations[i][0] - dimensions[i][0] / 2,
                        locations[i][1] - dimensions[i][1] / 2,
                        locations[i][2] - dimensions[i][2] / 2]
            max_corner = [locations[i][0] + dimensions[i][0] / 2,
                        locations[i][1] + dimensions[i][1] / 2,
                        locations[i][2] + dimensions[i][2] / 2]
            el_corners = [min_corner, max_corner]
            element_corners.append(el_corners)
        el_comb_list = itertools.combinations(element_corners, 2)
        for comb in el_comb_list:
            (corners1, corners2) = comb
            x_overlap = min([corners1[1][0], corners2[1][0]]) - max([corners1[0][0], corners2[0][0]])
            y_overlap = min([corners1[1][1], corners2[1][1]]) - max([corners1[0][1], corners2[0][1]])
            z_overlap = min([corners1[1][2], corners2[1][2]]) - max([corners1[0][2], corners2[0][2]])
            if x_overlap >= 0 and y_overlap >= 0 and z_overlap >= 0:
                overlap += x_overlap * y_overlap * z_overlap
        return overlap


    def overlap_cost_arbitrary(self, dimensions, locations, orientations, struct_dims, struct_locs, struct_orients):
        
        num_comps = len(dimensions)
        num_panels = len(struct_dims)
        comp_pair_list = itertools.combinations(range(num_comps), 2)
        overlap_ind = [False] * num_comps
        for pair in comp_pair_list:
            (ind1, ind2) = pair
            overlap_bool = self.separating_axis_test(
                dimensions[ind1], locations[ind1], orientations[ind1],
                dimensions[ind2], locations[ind2], orientations[ind2]
            )
            if overlap_bool:
                # print(f"Overlap detected between components {ind1} and {ind2}")
                overlap_ind[max(ind1, ind2)] = True # max to penalize the second component placed
        struct_comp_pair_list = itertools.product(range(num_comps), range(num_panels))
        for struct_pair in struct_comp_pair_list:
            comp_ind, struct_ind = struct_pair
            overlap_bool = self.separating_axis_test(
                dimensions[comp_ind], locations[comp_ind], orientations[comp_ind],
                struct_dims[struct_ind], struct_locs[struct_ind], struct_orients[struct_ind]
            )
            if overlap_bool:
                # print(f"Overlap detected between component {comp_ind} and structure panel {struct_ind}")
                overlap_ind[comp_ind] = True
        return overlap_ind


    def separating_axis_test(self, dim1, loc1, dcm1, dim2, loc2, dcm2):
        
        axes1 = dcm1.T
        axes2 = dcm2.T
        half_sizes1 = np.array(dim1) / 2.0
        half_sizes2 = np.array(dim2) / 2.0
        t = np.array(loc2) - np.array(loc1)
        t_local1 = np.dot(axes1, t)
        R = np.dot(axes1, axes2.T)
        R_abs = np.abs(R) + 1e-6
        # epsilon = 1e-6
        epsilon = 1e-9
        for i in range(3):
            ra = half_sizes1[i]
            rb = np.dot(half_sizes2, R_abs[i, :])
            if abs(t_local1[i]) > ra + rb - epsilon:
                return False
        for j in range(3):
            ra = np.dot(half_sizes1, R_abs[:, j])
            rb = half_sizes2[j]
            if abs(np.dot(t, axes2[j])) > ra + rb - epsilon:
                return False
        for i in range(3):
            for j in range(3):
                ra = half_sizes1[(i + 1) % 3] * R_abs[(i + 2) % 3, j] + half_sizes1[(i + 2) % 3] * R_abs[(i + 1) % 3, j]
                rb = half_sizes2[(j + 1) % 3] * R_abs[i, (j + 2) % 3] + half_sizes2[(j + 2) % 3] * R_abs[i, (j + 1) % 3]
                if abs(t_local1[(i + 2) % 3] * R[(i + 1) % 3, j] - t_local1[(i + 1) % 3] * R[(i + 2) % 3, j]) > ra + rb - 2*epsilon:
                    return False
        return True


    def wire_cost(self, dimensions, locations, types, orientations):
        
        port_dir_base = np.array([-1, 0, 0])
        PCUInd = types.index("PCU")
        PCULoc = locations[PCUInd]
        PCUPortDir = np.matmul(orientations[PCUInd], port_dir_base)
        PCUPortLoc = np.array(PCULoc) + np.multiply(np.array(dimensions[PCUInd]) / 2, PCUPortDir)
        tot_wire_len = 0
        for ind, loc in enumerate(locations):
            port_dir = np.matmul(orientations[ind], port_dir_base)
            comp_port_loc = np.array(loc) + np.multiply(np.array(dimensions[ind]) / 2, port_dir)
            wire_len = np.abs(PCUPortLoc[0] - comp_port_loc[0]) + np.abs(PCUPortLoc[1] - comp_port_loc[1]) + np.abs(PCUPortLoc[2] - comp_port_loc[2])
            tot_wire_len += wire_len
        return tot_wire_len


    def thermal_cost(self, dimensions, locations, heat_disps, struct_dims, struct_locs):

        stefan_boltzmann_const = 5.670374419e-8  # W/m^2K^4
        
        # Key thermal properties - these are the two you need to distinguish
        absorptivity = 0.35   # alpha - for incoming solar radiation
        emissivity   = 0.85   # epsilon - for outgoing thermal radiation
        # NOTE: white paint is a common choice for passive thermal control
        #       because low alpha/high epsilon gives a cool equilibrium temp
        
        # Surface area of structure
        surf_area = 0
        for dim in struct_dims[:8]:  # only shell panels, not shelves
            surf_area += dim[0] * dim[1]
        
        # Average sunlit area (sphere approximation = 1/4 total area)
        average_sunlit_area = surf_area / 4
        
        sun_heat_flux = 1361  # W/m^2 (solar constant at 1 AU)
        
        # Absorbed solar heat (apply absorptivity here, NOT emissivity)
        sc_heat_sun = absorptivity * average_sunlit_area * sun_heat_flux
        
        # Internal dissipation
        sc_heat_comps = sum(heat_disps)
        
        # Total heat in
        sc_heat_in = sc_heat_sun + sc_heat_comps
        
        # Steady state: Q_in = Q_out
        # Q_out = emissivity * stefan_boltzmann * surf_area * T^4
        # Solving for T:
        temp_kelvin = (sc_heat_in / (emissivity * stefan_boltzmann_const * surf_area)) ** 0.25
        target = 273.15 + 10

        temp_diff = abs(temp_kelvin - target)
        
        return temp_diff

        # Qin = np.zeros(len(locations))
        # SA = np.zeros(len(dimensions))
        # for ind, dims in enumerate(dimensions):
        #     SA[ind] = 2 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])
        # comp_pair_list = itertools.combinations(range(len(locations)), 2)
        # for pair in comp_pair_list:
        #     locA = locations[pair[0]]
        #     locB = locations[pair[1]]
        #     SAA = SA[pair[0]]
        #     SAB = SA[pair[1]]
        #     heatDispsA = heat_disps[pair[0]]
        #     heatDispsB = heat_disps[pair[1]]
        #     r = np.sqrt((locA[0] - locB[0]) ** 2 + (locA[1] - locB[1]) ** 2 + (locA[2] - locB[2]) ** 2)
        #     distance_sphere = 4 * np.pi * r ** 2
        #     if SAA < distance_sphere:
        #         Qin[pair[0]] += SAA * heatDispsB / (r ** 2)
        #     else:
        #         Qin[pair[0]] += heatDispsB
        #     if SAB < distance_sphere:
        #         Qin[pair[1]] += SAB * heatDispsA / (r ** 2)
        #     else:
        #         Qin[pair[1]] += heatDispsA
        # Qnet = Qin - np.array(heat_disps)
        # QnetVar = np.var(Qnet)
        # return QnetVar


    def pointing_obj(self, dimensions, locations, types, orientations, pointing):
        
        point_dir = np.array([1, 0, 0])
        new_point_locs = []
        new_point_dims = []
        new_point_orients = []
        for ind, comp in enumerate(types):
            if pointing[ind]:
                comp_point_dir = np.matmul(orientations[ind], point_dir)
                point_loc = np.array(locations[ind]) + np.multiply(np.array(dimensions[ind]) / 2, comp_point_dir)
                point_dim_main = np.multiply(comp_point_dir - point_loc, comp_point_dir)
                point_dim_off = np.abs(np.matmul(orientations[ind], np.array([0, 0.01, 0.01])))
                point_dims = point_dim_main + point_dim_off
                point_loc_center = point_loc + np.multiply(point_dim_main / 2, comp_point_dir)
                new_point_locs.append(point_loc_center)
                new_point_dims.append(point_dims)
                new_point_orients.append(orientations[ind])
        return new_point_locs, new_point_dims, new_point_orients


    def structural_feasibility(self, struct_dims, struct_locs, struct_orients):
        
        num_panels = len(struct_dims)
        if num_panels <= 1:
            return True
        graph = {panel: [] for panel in range(num_panels)}
        for i in range(num_panels):
            for j in range(i + 1, num_panels):
                if self.separating_axis_test(struct_dims[i], struct_locs[i], struct_orients[i], struct_dims[j], struct_locs[j], struct_orients[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        visited = set()
        queue = deque([0])
        while queue:
            panel = queue.popleft()
            if panel not in visited:
                visited.add(panel)
                queue.extend(graph[panel])
        connected = (len(visited) == num_panels)
        return connected


    def constraint_cost(self, dimensions, locations, types, orientations, masses, pointing, struct_dims, struct_locs, struct_orients):
        
        new_point_locs, new_point_dims, new_point_orients = self.pointing_obj(dimensions, locations, types, orientations, pointing)

        overlap_inds = self.overlap_cost_arbitrary(
            dimensions + new_point_dims,
            locations + new_point_locs,
            orientations + new_point_orients,
            struct_dims, struct_locs, struct_orients
        )
        
        return overlap_inds


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
        structure_panels[1].orientation = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

        # side panels
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [z_dim, y_dim, structure_panels[2].thickness]
        structure_panels[2].location = [-x_dim/4, 0, 0]
        structure_panels[2].orientation = np.array([[0,0,1],[0,1,0],[-1,0,0]])


        side_orientation = np.array([[0,0,-1],[0,1,0],[1,0,0]])
        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [z_dim, np.sqrt((x_dim*3/4)**2 + (y_dim/2)**2), structure_panels[3].thickness]
        structure_panels[3].location = [x_dim/8, -y_dim/4, 0]
        angle1 = -np.arctan2(x_dim*3/4, y_dim/2)  # tilt angle
        rot_x1 = np.array([
            [1, 0, 0],
            [0, np.cos(angle1), -np.sin(angle1)],
            [0, np.sin(angle1),  np.cos(angle1)]
        ])
        structure_panels[3].orientation = side_orientation @ rot_x1

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [z_dim, np.sqrt((x_dim*3/4)**2 + (y_dim/2)**2), structure_panels[4].thickness]
        structure_panels[4].location = [x_dim/8, y_dim/4, 0]
        angle2 = np.arctan2(x_dim*3/4, y_dim/2)  # opposite tilt
        rot_x2 = np.array([
            [1, 0, 0],
            [0, np.cos(angle2), -np.sin(angle2)],
            [0, np.sin(angle2),  np.cos(angle2)]
        ])
        structure_panels[4].orientation = side_orientation @ rot_x2

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
        structure_panels[1].orientation = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

        # side panels
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [z_dim, y_dim, structure_panels[2].thickness]
        structure_panels[2].location = [-x_dim/2, 0, 0]
        structure_panels[2].orientation = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [z_dim, y_dim, structure_panels[3].thickness]
        structure_panels[3].location = [x_dim/2, 0, 0]
        structure_panels[3].orientation = np.array([[0,0,-1],[0,1,0],[1,0,0]])

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [x_dim, z_dim, structure_panels[4].thickness]
        structure_panels[4].location = [0, -y_dim/2, 0]
        structure_panels[4].orientation = np.array([[1,0,0],[0,0,1],[0,-1,0]])

        structure_panels[5] = deepcopy(self.base_panel)
        structure_panels[5].dimensions = [x_dim, z_dim, structure_panels[5].thickness]
        structure_panels[5].location = [0, y_dim/2, 0]
        structure_panels[5].orientation = np.array([[1,0,0],[0,0,-1],[0,1,0]])

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
        structure_panels[1].orientation = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

        # side panels
        side_orientation_2 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        structure_panels[2] = deepcopy(self.base_panel)
        structure_panels[2].dimensions = [z_dim, y_dim/2, structure_panels[2].thickness]
        structure_panels[2].location = [-x_dim/2, 0, 0]
        structure_panels[2].orientation = side_orientation_2

        side_orientation_3 = np.array([[0,0,-1],[0,1,0],[1,0,0]])
        structure_panels[3] = deepcopy(self.base_panel)
        structure_panels[3].dimensions = [z_dim, y_dim/2, structure_panels[3].thickness]
        structure_panels[3].location = [x_dim/2, 0, 0]
        structure_panels[3].orientation = side_orientation_3

        structure_panels[4] = deepcopy(self.base_panel)
        structure_panels[4].dimensions = [z_dim, np.sqrt((x_dim/2)**2 + (y_dim/4)**2), structure_panels[4].thickness]
        structure_panels[4].location = [-x_dim/4, -y_dim*3/8, 0]
        angle1 = -np.arctan2(x_dim/2, y_dim/4)  # tilt angle
        rot_x1 = np.array([
            [1, 0, 0],
            [0, np.cos(angle1), -np.sin(angle1)],
            [0, np.sin(angle1),  np.cos(angle1)]
        ])
        structure_panels[4].orientation = side_orientation_2 @ rot_x1

        structure_panels[5] = deepcopy(self.base_panel)
        structure_panels[5].dimensions = [z_dim, np.sqrt((x_dim/2)**2 + (y_dim/4)**2), structure_panels[5].thickness]
        structure_panels[5].location = [-x_dim/4, y_dim*3/8, 0]
        angle2 = np.arctan2(x_dim/2, y_dim/4)  # opposite tilt
        rot_x2 = np.array([
            [1, 0, 0],
            [0, np.cos(angle2), -np.sin(angle2)],
            [0, np.sin(angle2),  np.cos(angle2)]
        ])
        structure_panels[5].orientation = side_orientation_2 @ rot_x2

        structure_panels[6] = deepcopy(self.base_panel)
        structure_panels[6].dimensions = [z_dim, np.sqrt((x_dim/2)**2 + (y_dim/4)**2), structure_panels[6].thickness]
        structure_panels[6].location = [x_dim/4, -y_dim*3/8, 0]
        angle1 = -np.arctan2(x_dim/2, y_dim/4)  # tilt angle
        rot_x1 = np.array([
            [1, 0, 0],
            [0, np.cos(angle1), -np.sin(angle1)],
            [0, np.sin(angle1),  np.cos(angle1)]
        ])
        structure_panels[6].orientation = side_orientation_3 @ rot_x1

        structure_panels[7] = deepcopy(self.base_panel)
        structure_panels[7].dimensions = [z_dim, np.sqrt((x_dim/2)**2 + (y_dim/4)**2), structure_panels[7].thickness]
        structure_panels[7].location = [x_dim/4, y_dim*3/8, 0]
        angle2 = np.arctan2(x_dim/2, y_dim/4)  # opposite tilt
        rot_x2 = np.array([
            [1, 0, 0],
            [0, np.cos(angle2), -np.sin(angle2)],
            [0, np.sin(angle2),  np.cos(angle2)]
        ])
        structure_panels[7].orientation = side_orientation_3 @ rot_x2

        return structure_panels
    

    def get_shelves(self, structure_panels, shape, x_dim, y_dim, z_dim, num_shelves):

        start_ind = 8
        distance_between_shelves = z_dim / (num_shelves + 1)
        for ind in range(int(num_shelves)):
            structure_panels[start_ind+ind] = deepcopy(self.base_panel)
            structure_panels[start_ind+ind].location = [0, 0, -z_dim/2 + distance_between_shelves*(ind+1)]
            structure_panels[start_ind+ind].dimensions = [x_dim, y_dim, structure_panels[start_ind+ind].thickness]
        
        return structure_panels
    

    def get_structural_panels(self, x):

        structure_panels = [None] * (8 + self.max_shelves)

        if x[0] == 0: # triangle
            structure_panels = self.triangle_prism_shell(structure_panels, x[1], x[2], x[3])
        elif x[0] == 1: # rectangle
            structure_panels = self.rectangular_prism_shell(structure_panels, x[1], x[2], x[3])
        elif x[0] == 2: # hexagon
            structure_panels = self.hexagonal_prism_shell(structure_panels, x[1], x[2], x[3])
        else:
            raise ValueError(f"INVALID STRUCTURE SHAPE {x[0]}")
        
        # shelves
        structure_panels = self.get_shelves(structure_panels, x[0], x[1], x[2], x[3], x[4])
        self.structure_panels = structure_panels


    def get_components(self, x):

        # place components
        for i in range(len(self.component_list)):
            compPanel = x[5*i+5]
            compLoc = [x[5*i+6],x[5*i+7]]
            compFace = x[5*i+8]
            compRot = x[5*i+9]

            compOrient = faceAndRot2DCM(faceChoice=compFace, rot=compRot)

            surfNormal = np.array([0,0,1])
            if self.component_list[i].pointing == True: 
                panelChoice = self.structure_panels[int(compPanel)]
                panelChoiceDCM = panelChoice.orientation
                panelChoiceDCM = np.matmul(panelChoiceDCM,np.array([[-1,0,0],[0,1,0],[0,0,-1]]))
                # surfNormal = np.array([0,0,-1])
            elif compPanel >= len(self.structure_panels):  # underside of shelf
                panelChoice = self.structure_panels[int(compPanel - self.max_shelves)]
                panelChoiceDCM = panelChoice.orientation
                panelChoiceDCM = np.matmul(panelChoiceDCM,np.array([[1,0,0],[0,1,0],[0,0,-1]]))
                # surfNormal = np.array([0,0,-1])
            else:
                panelChoice = self.structure_panels[int(compPanel)]
                panelChoiceDCM = panelChoice.orientation
            
            compDCM = align_cuboid_with_plane(compOrient, panelChoiceDCM)

            self.component_list[i].orientation = compDCM
            
            if compFace%3 == 0:
                dimOffset = self.component_list[i].dimensions[2]/2
            elif compFace%3 == 1:
                dimOffset = self.component_list[i].dimensions[1]/2
            elif compFace%3 == 2:
                dimOffset = self.component_list[i].dimensions[0]/2

            panelOffset = panelChoice.thickness/2
            offsetVect = np.matmul(panelChoiceDCM,surfNormal*(dimOffset+panelOffset))
            
            surfLoc = np.matmul(panelChoiceDCM,np.multiply([compLoc[0],compLoc[1],surfNormal[2]],np.array(panelChoice.dimensions)/2))
            compLoc = surfLoc + offsetVect + panelChoice.location
            self.component_list[i].location = compLoc


    def get_panels_and_components(self, x):

        self.get_structural_panels(x)
        self.get_components(x)
        return self.structure_panels, self.component_list
    

    def evaluate(self, x):

        self.get_structural_panels(x)

        for comp in range(self.num_components):
            panel_choice = x[5*comp+5]
            if panel_choice >= len(self.structure_panels):
                panel_choice -= self.max_shelves # shelves have two possibilities
            if self.structure_panels[int(panel_choice)] is None:
                # print(f"PANEL CHOICE INVALID: {panel_choice} for component {comp}")
                # print(f"VALID PANELS: {[i for i, panel in enumerate(self.structure_panels) if panel is not None]}")
                # print(f"Full Design Vector: {x}")
                return [-1]*self.num_objectives, True, 0.0 # constraint violated if a non-existent panel is chosen
            if self.component_list[comp].pointing and panel_choice > 7: # checks if pointing component is on a shelf
                print(f"POINTING COMPONENT {comp} CANNOT BE ON A SHELF")
                return [-1]*self.num_objectives, True, 0.0

        self.get_components(x)
        # print(f"Got past panel selection and component placement!")

            # pull parameters from the components
        locations = []
        dimensions = []
        orientations = []
        types = []
        masses = []
        heat_disps = []
        pointing = []
        for comp in self.component_list:
            locations.append(comp.location)
            orientations.append(comp.orientation)
            # dimensions.append(np.matmul(np.abs(comp.orientation), comp.dimensions))
            dimensions.append(comp.dimensions)
            types.append(comp.type)
            masses.append(comp.mass)
            heat_disps.append(comp.heatDisp)
            pointing.append(comp.pointing)

        struct_masses = []
        struct_dims = []
        struct_locs = []
        struct_orients = []
        for panel in self.structure_panels:
            if panel is not None:
                struct_masses.append(panel.mass)
                # struct_dims.append(np.matmul(np.abs(panel.orientation), panel.dimensions))
                struct_dims.append(panel.dimensions)
                struct_locs.append(panel.location)
                struct_orients.append(panel.orientation)

        # Get the cost from each cost source
        overlap_inds = self.constraint_cost(
            dimensions, locations, types, orientations, masses, pointing,
            struct_dims, struct_locs, struct_orients
        )
        if any(overlap_inds):
            constraint_violated = True
        else:            
            constraint_violated = False

        overlap_score = sum(overlap_inds)/len(self.component_list)

        cm_cost_val = self.center_mass_cost(
            locations, masses, struct_locs, struct_masses
        )
        off_axis_inertia, on_axis_inertia = self.inertia_cost(
            dimensions + struct_dims, locations + struct_locs, masses + struct_masses
        )
        wire_cost_val = self.wire_cost(
            dimensions, locations, types, orientations
        )
        thermal_cost_val = self.thermal_cost(
            dimensions, locations, heat_disps,
            struct_dims, struct_locs
        )

        cost_list = [
            cm_cost_val, off_axis_inertia, on_axis_inertia,
            wire_cost_val, thermal_cost_val
        ]

        # print(f"Found a valid design! Costs: {cost_list}")
        return cost_list, constraint_violated, overlap_score


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
