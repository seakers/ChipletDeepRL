import matplotlib 
matplotlib.use('agg')
import numpy as np

from utils.design_utils import design_to_chiplet_values
from chiplet_model.dse.lib.chiplet_system import ChipletSystem
from chiplet_model.dse.lib.trace_parser import TraceParser


def runSingleCascade(params, partitions):
    """
    Run a single instance of the Cascade model.
    """
    WORKSPACE = params['workspace']
    TRACE_DIR = params['trace_dir']
    CHIPLET_LIBRARY = params['chiplet_library']
    EXPERIMENT_DIR = params['experiment_dir']
    OUTPUT_DIR = params['output_dir']
    num_components = params['num_components']
    num_comp_types = params['num_comp_types']

    chiplet_nums = design_to_chiplet_values(partitions)

    tp = TraceParser(TRACE_DIR, EXPERIMENT_DIR)

    desired_chiplets = ["gpu"] * chiplet_nums[0] + ["atten"] * chiplet_nums[1] + ["sparse"] * chiplet_nums[2] + ["conv"] * chiplet_nums[3]
    numChannels = 16 # can be changed to be a variable
    agg_kernel_results = []

    for TRACE_ID in range(len(tp.all_traces)):
        # tp.all_traces[TRACE_ID].print_line()
        # print("Working on Trace %i" % TRACE_ID)

        # create SoC and add chiplets to SoC
        cs = ChipletSystem(CHIPLET_LIBRARY, verbose=0)
        cs.configure_system(desired_chiplets, tp.optimization_goal)
        cs.init_system_bandwidth(bw_per_channel=64, num_channels=numChannels)
        # valid = cs.check_valid_system(self.tp.get_trace(TRACE_ID))
        # if not valid == -1:
        kernel_results = cs.characterize_workload(tp.get_trace(TRACE_ID), cut_dim="batch" if tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights", dtype=2)
        agg_kernel_results += kernel_results * tp.all_traces[TRACE_ID].weighted_score # sudo run the workload "weighted_score" times


    # total_exe, total_energy = getTimeAndEnergy(agg_kernel_results, cs.get_num_chiplets())
    num_chiplets = cs.get_num_chiplets()

    kernel_names = [] 
    kernel_exe = []
    kernel_energy = []
    kernel_work = {}
    kernel_breakdown = {}
    chiplet_names = []
    for chiplet_id in range(num_chiplets):
        kernel_work[chiplet_id] = []
        chiplet_names.append("")

    for kernel in agg_kernel_results:
        for chiplet_id in range(num_chiplets):
            if chiplet_id in kernel["chiplets"].keys(): # chiplet participated 
                kernel_work[chiplet_id].append(kernel["chiplets"][chiplet_id]["work"])
                chiplet_names[chiplet_id] = kernel["chiplets"][chiplet_id]["name"]
            else: # chiplet did not participate
                kernel_work[chiplet_id].append(0)
        kernel_exe.append(kernel["total"]["exe_time"])
        kernel_energy.append(kernel["total"]["energy"] )

        if kernel["name"] not in kernel_breakdown:
            kernel_breakdown[kernel["name"]] = 0
        kernel_breakdown[kernel["name"]] += kernel["total"]["exe_time"]
        kernel_names.append(kernel["name"])
    
    # normalize kernel_work
    total_exe = sum(kernel_exe)
    total_energy = sum(kernel_energy)

    # print("Total Time: %0.5fms" % (total_exe))
    # print("Total Energy: %0.5fmJ" % (total_energy))

    return float(total_exe), float(total_energy)

if __name__ == "__main__":
    exe, energy = runSingleCascade()

    
# class CascadeProblem(ElementwiseProblem):
#     def __init__(self, n_var, n_obj, num_components, num_comp_types, TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR):
#         super().__init__(n_var=n_var, 
#                          n_obj=n_obj, 
#                          n_constr=1, 
#                          xl=1, 
#                          xu=num_components+num_comp_types,
#                          vtype=int)
#         self.TRACE_DIR = TRACE_DIR
#         self.CHIPLET_LIBRARY = CHIPLET_LIBRARY
#         self.EXPERIMENT_DIR = EXPERIMENT_DIR
#         self.OUTPUT_DIR = OUTPUT_DIR

#         self.num_components = num_components
#         self.num_comp_types = num_comp_types

#         self.tp = TraceParser(self.TRACE_DIR, self.EXPERIMENT_DIR)
    
#     def _evaluate(self, x, out, *args, **kwargs):
#         x = np.sort(np.array(x, dtype=int))
#         # print(f"Partitions received: {x}")
#         chiplet_nums = np.concatenate((x, [self.num_components+self.num_comp_types])) - np.concatenate(([0], x)) - 1
#         # print(f"Evaluating with chiplet configuration: {chiplet_nums}")
#         desired_chiplets = ["gpu"] * chiplet_nums[0] + ["atten"] * chiplet_nums[1] + ["sparse"] * chiplet_nums[2] + ["conv"] * chiplet_nums[3]
#         numChannels = 16 # can be changed to be a variable
#         agg_kernel_results = []

#         for TRACE_ID in range(len(self.tp.all_traces)):
#             # self.tp.all_traces[TRACE_ID].print_line()
#             # print("Working on Trace %i" % TRACE_ID)

#             # create SoC and add chiplets to SoC
#             cs = ChipletSystem(self.CHIPLET_LIBRARY, verbose=0)
#             cs.configure_system(desired_chiplets, self.tp.optimization_goal)
#             cs.init_system_bandwidth(bw_per_channel=64, num_channels=numChannels)
#             # valid = cs.check_valid_system(self.tp.get_trace(TRACE_ID))
#             # if not valid == -1:
#             kernel_results = cs.characterize_workload(self.tp.get_trace(TRACE_ID), cut_dim="batch" if self.tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights", dtype=2)
#             agg_kernel_results += kernel_results * self.tp.all_traces[TRACE_ID].weighted_score # sudo run the workload "weighted_score" times

#         total_exe, total_energy = self.getTimeAndEnergy(agg_kernel_results, cs.get_num_chiplets())
#         # if total_exe == 0 or total_energy == 0:
#         #     out["F"] = [10e6, 10e6]
#         # else:

#         # context_file = self.OUTPUT_DIR + "/pointContext/" + f"{chiplet_nums[0]}gpu{chiplet_nums[1]}attn{chiplet_nums[2]}sparse{chiplet_nums[3]}conv.txt"
#         # with open(context_file, "w") as f:
#         #     for result in agg_kernel_results:
#         #         result_copy = {key: value for key, value in result.items() if key != "chiplets"}
#         #         result_copy["total"] = {key: f"{value:.1e}" for key, value in result_copy["total"].items()}
#         #         f.write(str(result_copy) + "\n")
#         # print(f"Results saved to {context_file}")

#         # designs_file = "ga_vals/designs.csv"
#         # with open(designs_file, "a") as f:
#         #     f.write(f"{chiplet_nums[0]},{chiplet_nums[1]},{chiplet_nums[2]},{chiplet_nums[3]}\n")

#         # objectives_file = "ga_vals/objectives.csv"
#         # with open(objectives_file, "a") as f:
#         #     f.write(f"{total_exe},{total_energy}\n")
        
#         # print(f"Summary saved to {result_file}")

#         out["F"] = [total_exe, total_energy]
#         # return [total_exe, total_energy] # delete after prototyping


#     def getTimeAndEnergy(self, kernel_results, num_chiplets):
#         kernel_names = [] 
#         kernel_exe = []
#         kernel_energy = []
#         kernel_work = {}
#         kernel_breakdown = {}
#         chiplet_names = []
#         for chiplet_id in range(num_chiplets):
#             kernel_work[chiplet_id] = []
#             chiplet_names.append("")

#         for kernel in kernel_results:
#             for chiplet_id in range(num_chiplets):
#                 if chiplet_id in kernel["chiplets"].keys(): # chiplet participated 
#                     kernel_work[chiplet_id].append(kernel["chiplets"][chiplet_id]["work"])
#                     chiplet_names[chiplet_id] = kernel["chiplets"][chiplet_id]["name"]
#                 else: # chiplet did not participate
#                     kernel_work[chiplet_id].append(0)
#             kernel_exe.append(kernel["total"]["exe_time"])
#             kernel_energy.append(kernel["total"]["energy"] )

#             if kernel["name"] not in kernel_breakdown:
#                 kernel_breakdown[kernel["name"]] = 0
#             kernel_breakdown[kernel["name"]] += kernel["total"]["exe_time"]
#             kernel_names.append(kernel["name"])
        
#         # normalize kernel_work
#         total_exe = sum(kernel_exe)
#         total_energy = sum(kernel_energy)

#         kernel_work_agg = []
#         for chiplet_id in range(num_chiplets):
#             kernel_work_agg.append(np.dot(np.array(kernel_work[chiplet_id]), np.array(kernel_exe)/total_exe))
        
#         # print(f"{len(kernel_work_agg)} frac work per chiplet: [", end=" ")
#         # for frac_work in kernel_work_agg:
#         #     print("%0.2f%%" % (frac_work*100), end=" ")
#         # print("]")
#         # print("Total Time: %0.5fs" % (total_exe))
#         # print("Total Energy: %0.5fJ" % (total_energy))

#         # # Write outputs to a file
#         # outlist = [total_exe*1000, total_energy*10**3]
#         # with open(self.OUTPUT_DIR + "/points.txt", "w", newline='') as f:
#         #     writer = csv.writer(f)
#         #     writer.writerow(outlist)

#         return float(total_exe), float(total_energy)