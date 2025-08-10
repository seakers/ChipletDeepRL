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
    def __init__(self, dimensions=[1,1], location=[0,0,0], orientation=np.array([[1,0,0],[0,1,0],[0,0,1]]), density=2710., **kwargs):
        self.location = location
        self.orientation = orientation
        self.dimensions = dimensions
        self.density = density

        self.thickness = 0.01 # one centimeter thick
        self.dimensions.append(self.thickness)
        self.mass = self.dimensions[0]*self.dimensions[1]*self.thickness*self.density
        

        # Optional arguments in case something else needs to be specified
        for k in kwargs.keys():
            self.__setattr__(k,kwargs[k])