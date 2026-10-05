import numpy as np

class Component:
    """
    A component object containing the component charactaristics

    Input variables and thier desired units/type:

    Necessary Variables:
    mass kg
    dimensions [m,m,m]

    Optional General Variables:
    avgPower W
    peakPower W
    name string
    tempRange [C,C]

    Used for Camera Payloads:
    resolution arcsec
    FOV degrees
    specRange string

    Used for Attitude Determination Components:
    accuracy degrees

    Used for Attitude Controllers:
    momentum Nms
    """     
    def __init__(self, type, mass=None, dimensions=None, in_out='in', **kwargs):
        self.type = type
        self.mass = mass # kg
        self.dimensions = dimensions # m
        self.in_out = in_out

        # get the optional arguments which can differ between components
        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])

class StructPanel:
    """
    An object containing everything to define a structural panel

    inputs:
    dimensions [m,m]: list of two values
    location [m,m,m]: list of three values
    orientation [rad,rad,rad]: list of three values
    density kg*m^-3: float, aluminum by default (2710)
    """
    def __init__(self, dimensions=None, location=None, orientation=None, density=2710., **kwargs):
        self.location = [0, 0, 0] if location is None else location
        self.orientation = np.eye(3) if orientation is None else orientation
        self.density = density
        self.thickness = 0.01 # one centimeter thick
        self.dimensions = list([1, 1] if dimensions is None else dimensions) + [self.thickness]
        # vertices (n, 2) in the panel's local x-y for non-rectangular panels; None for rectangles
        self.polygon = None

        # Optional arguments in case something else needs to be specified
        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])

    @property
    def area(self):
        if self.polygon is None:
            return self.dimensions[0] * self.dimensions[1]
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @property
    def mass(self):
        return self.area * self.thickness * self.density