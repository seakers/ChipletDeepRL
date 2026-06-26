from main import *
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import Component, StructPanel
from utils.component_list import getComponents

def initialize_methods_config():
    """Initialize all optimization methods with their properties"""
    methods = [
        OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "orange", True),
    ]
    return methods

methods_config = initialize_methods_config()
storage = initialize_storage(methods_config)
fpath = "results/2026-06-23_12-58-34/run_data/genetic_algorithm_run_0.pkl"
with open(fpath, "rb") as f:
    data = pickle.load(f)
store_results(storage, methods_config[0].name, data)

params = {'date_str': "2026-06-23_12-58-34"}
component_list, transfer_learning_components = getComponents()
base_panel = StructPanel()
eval_function = Chiplet_Configuration_Design(component_list, base_panel)

visualize_configurations(storage, methods_config, params, eval_function, max_designs=1, interactive=True)