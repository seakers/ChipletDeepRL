import os

from chiplet_model.dse.chiplet_library.base_chiplet import BaseChiplet
from chiplet_model.dse.chiplet_library.gpu_chiplet import GPUChiplet
from chiplet_model.dse.chiplet_library.sparse_chiplet import SparseChiplet
from chiplet_model.dse.chiplet_library.conv_chiplet import ConvChiplet
from chiplet_model.dse.chiplet_library.atten_chiplet import AttenChiplet


class RegisterChiplets:
    def __init__(self):
        self.chiplet_library = {}
    
    def register_chiplets(self):
        self.chiplet_library["gpu"]     = GPUChiplet
        self.chiplet_library["sparse"]  = SparseChiplet
        self.chiplet_library["conv"]    = ConvChiplet
        self.chiplet_library["atten"]   = AttenChiplet

        return self.chiplet_library
