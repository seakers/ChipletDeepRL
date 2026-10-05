import numpy as np

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

class HypervolumeGrid:

    def __init__(self,refPoint):
        if isinstance(refPoint,list):
            refPoint = np.array(refPoint)
        self.refPoint = refPoint
        self.initializeGrid()

    def initializeGrid(self):
        """
        Function to initialize the hypervolume grid with the reference point
        """
        self.gridSize = 21 # maybe change this to have different resolution closer to the best point
        self.HVMax = np.prod(self.refPoint)

        exp = np.arange(len(self.refPoint)-1,-1,-1)
        self.mult = np.power(self.gridSize,exp)

        self.numPoints = (self.gridSize)**len(self.refPoint)
        self.dominated = np.zeros(int(self.numPoints), dtype=bool)

        self.paretoFrontPoint = np.array([])
        self.paretoFrontSolution = np.array([])

    def updateHV(self,point,solution):
        """
        Function to update the hypervolume grid and pareto front with a new point and index. 
        Index for each call should be unique and able to reference the solution.
        """
        if isinstance(point,list):
            point = np.array(point)
        if isinstance(solution,list):
            solution = np.array(solution)
        pointCalc = self.refPoint - point
        gridPoint = np.floor(pointCalc/self.refPoint * (self.gridSize-1))
        pointIdx = int(np.dot(gridPoint,self.mult))

        if not self.dominated[pointIdx]:
            self.updateParetoFront(point,solution)

            # mark every cell with all coordinates <= gridPoint (same as the old per-cell loop)
            grid = self.dominated.reshape((self.gridSize,) * len(self.refPoint))
            grid[tuple(slice(0, max(int(g) + 1, 0)) for g in gridPoint)] = True

    def updateParetoFront(self,newPoint,solution):
        """
        Add a new cost to the pareto front if it is not dominated
        """
        isDominated = False
        if self.paretoFrontPoint.size:
            isDominated = np.any(np.all(self.paretoFrontPoint < newPoint, axis = 1))
            newDominated = np.all(self.paretoFrontPoint > newPoint, axis = 1)

            self.paretoFrontPoint = self.paretoFrontPoint[~newDominated]
            self.paretoFrontSolution = self.paretoFrontSolution[~newDominated]
        
        if isDominated == False:
            self.paretoFrontPoint = np.vstack([self.paretoFrontPoint, newPoint]) if self.paretoFrontPoint.size else np.array([newPoint])
            self.paretoFrontSolution = np.vstack([self.paretoFrontSolution, solution]) if self.paretoFrontSolution.size else np.array([solution])
            # self.paretoFrontSolution = np.vstack([self.paretoFrontSolution, solution]) if self.paretoFrontSolution.size else solution

    def filterParetoFront(self, objInd, value):   
        """
        Filter the pareto front by the index of the objective and the desired value
        Ex: objInd = 1, value = 0.1 will return all points where the second objective is less than 0.1
        """
        if self.paretoFrontSolution.size:
            mask = self.paretoFrontPoint[:,objInd] < value
            self.paretoFrontPoint = self.paretoFrontPoint[mask]
            self.paretoFrontSolution = self.paretoFrontSolution[mask]

    def contribution(self, point):
        """HV the point would add to the grid, without adding it."""
        gridPoint = np.floor((self.refPoint - np.asarray(point)) / self.refPoint * (self.gridSize - 1))
        if np.any(gridPoint < 0):
            return 0.0
        grid = self.dominated.reshape((self.gridSize,) * len(self.refPoint))
        box = grid[tuple(slice(0, int(g) + 1) for g in gridPoint)]
        return (box.size - np.count_nonzero(box)) / self.numPoints * self.HVMax

    def getHV(self):
        return np.sum(self.dominated)/self.numPoints * self.HVMax