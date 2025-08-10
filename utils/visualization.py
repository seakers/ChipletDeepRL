import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils.config_utils import getCube


def config_visualization(struct_panels, component_list, date_str, method):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Adjustment for RS
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_aspect('equal')

    # objColor = tuple(np.random.rand(len(component_list), 3))

    for component in component_list:
        xRS, yRS, zRS = getCube(component.dimensions, component.location, component.orientation)
        # ax.plot_surface(xRS, yRS, zRS, color=objColor[i], label=allTypesCompsRS[i])
        ax.plot_surface(xRS, yRS, zRS)
        # point = ax.scatter(allLocsCompsRS[i][0], allLocsCompsRS[i][1], allLocsCompsRS[i][2], color=objColor[i])
        # proxyPointsRS.append(point)

    for panel in struct_panels:
        if panel:
            xPanel, yPanel, zPanel = getCube(panel.dimensions, panel.location, panel.orientation)
            ax.plot_surface(xPanel, yPanel, zPanel, alpha=0.1, color='tab:gray')

    plt.title("Visualization of Configuration RS")
    # plt.legend(proxyPointsRS, allTypesRS, loc='center left', bbox_to_anchor=(1.1, 0.5))

    plt.savefig(f"results/{date_str}/{method}_Config")
    pickle.dump(fig, open(f"results/{date_str}/{method}_Config_Interactive", "wb"))