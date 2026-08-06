"""
Interactive spacecraft design tool for generating imitation-learning demonstrations.

Controls:
  - 5 global sliders (always visible): shape, x_dim, y_dim, z_dim, num_shelves
  - A "component selector" slider that picks which component you're editing
  - 5 per-component sliders for the currently selected component
  - "Save Demo" button appends the current design vector + metrics to a file
  - "Reset" button restores the last saved / initial vector

Discrete variables snap to their allowed 'range' from design_space.
Continuous variables use their [min, max] bounds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from copy import deepcopy

# --- adjust these imports to your project ---
from utils.config_utils import getCube
from utils.evaluation import Chiplet_Configuration_Design

OBJECTIVE_NAMES = [
    "Center of Mass",
    "Off-Axis Inertia",
    "On-Axis Inertia",
    "Wire Cost",
    "Thermal Cost",
]

DEMO_FILE = "demonstrations.jsonl"


class DesignTool:
    def __init__(self, design):
        """
        design: an instance of your evaluation class.
                Must expose:
                  - design.design_space : list of dicts, one per variable in x,
                        each {'type': 'discrete', 'range': [...]}  OR
                              {'type': 'continuous', 'range': [min, max]}
                  - design.evaluate(x) -> (cost_list, constraint_violated, overlap_score)
                  - design.get_panels_and_components(x) -> (struct_panels, component_list)
                  - design.num_components, design.num_objectives, design.max_shelves
        """
        self.design = design
        self.space = design.design_space
        self.num_components = design.num_components
        self.n_vars = len(self.space)

        # start at the low end of each variable's range (valid + deterministic)
        self.x = [self._var_default(i) for i in range(self.n_vars)]

        self.selected_comp = 0  # which component is currently being edited
        self._syncing = False
        self._build_figure()
        self._refresh()  # initial draw + evaluation

    # ---------- design-space helpers ----------
    def _var_bounds(self, i):
        spec = self.space[i]
        rng = spec["range"]
        if spec["type"] == "discrete":
            return min(rng), max(rng), True  # (lo, hi, is_discrete)
        else:
            return rng[0], rng[1], False

    def _var_default(self, i):
        spec = self.space[i]
        if spec["type"] == "discrete":
            return spec["range"][0]
        return spec["range"][0]

    def _snap(self, i, val):
        """Snap a slider value to the nearest allowed value for discrete vars."""
        spec = self.space[i]
        if spec["type"] == "discrete":
            allowed = np.array(spec["range"], dtype=float)
            return float(allowed[np.argmin(np.abs(allowed - val))])
        return float(val)

    # index helpers for the flat vector x
    def _comp_var_index(self, comp, k):
        # global vars are indices 0..4; component vars start at 5
        # order per component: panel, x_loc, y_loc, face, rot
        return 5 + 5 * comp + k

    # ---------- figure construction ----------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.ax3d = self.fig.add_axes([0.05, 0.25, 0.5, 0.7], projection="3d")

        # text panel for objectives / constraints
        self.ax_text = self.fig.add_axes([0.60, 0.55, 0.38, 0.40])
        self.ax_text.axis("off")
        self.text_handle = self.ax_text.text(
            0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10
        )

        # --- global sliders (bottom strip) ---
        self.global_sliders = []
        global_labels = ["shape (0=tri,1=rect,2=hex)",
                         "x_dim", "y_dim", "z_dim", "num_shelves"]
        for i in range(5):
            ax = self.fig.add_axes([0.08, 0.18 - i * 0.035, 0.45, 0.02])
            lo, hi, is_disc = self._var_bounds(i)
            step = 1.0 if is_disc else None
            s = Slider(ax, global_labels[i], lo, hi,
                       valinit=self.x[i], valstep=step)
            s.on_changed(lambda val, idx=i: self._on_global_change(idx, val))
            self.global_sliders.append(s)

        self.global_boxes = {}
        for i in (1, 2):
            ax_box = self.fig.add_axes([0.545, 0.18 - i * 0.035, 0.045, 0.02])
            tb = TextBox(ax_box, "", initial=f"{self.x[i]:.3f}")
            tb.on_submit(lambda text, idx=i: self._on_global_box(idx, text))
            self.global_boxes[i] = tb

        # --- component selector ---
        ax_sel = self.fig.add_axes([0.62, 0.42, 0.30, 0.02])
        self.comp_selector = Slider(
            ax_sel, "component #", 0, self.num_components - 1,
            valinit=0, valstep=1
        )
        self.comp_selector.on_changed(self._on_select_component)

        # --- per-component sliders ---
        self.comp_sliders = []
        comp_labels = ["panel", "x_loc", "y_loc", "face", "rotation"]
        for k in range(5):
            ax = self.fig.add_axes([0.62, 0.36 - k * 0.035, 0.30, 0.02])
            idx = self._comp_var_index(self.selected_comp, k)
            lo, hi, is_disc = self._var_bounds(idx)
            step = 1.0 if is_disc else None
            s = Slider(ax, comp_labels[k], lo, hi,
                       valinit=self.x[idx], valstep=step)
            s.on_changed(lambda val, kk=k: self._on_comp_change(kk, val))
            self.comp_sliders.append(s)

        # number boxes for x_loc (1), y_loc (2), rotation (4)
        self.comp_boxes = {}
        for k in (1, 2, 4):
            ax_box = self.fig.add_axes([0.925, 0.36 - k * 0.035, 0.05, 0.02])
            idx = self._comp_var_index(self.selected_comp, k)
            tb = TextBox(ax_box, "", initial=f"{self.x[idx]:.3f}")
            tb.on_submit(lambda text, kk=k: self._on_comp_box(kk, text))
            self.comp_boxes[k] = tb

        # --- buttons ---
        ax_save = self.fig.add_axes([0.62, 0.10, 0.12, 0.04])
        self.btn_save = Button(ax_save, "Save Demo")
        self.btn_save.on_clicked(self._on_save)

        ax_reset = self.fig.add_axes([0.78, 0.10, 0.12, 0.04])
        self.btn_reset = Button(ax_reset, "Reset Comp")
        self.btn_reset.on_clicked(self._on_reset_component)

    # ---------- callbacks ----------
    def _on_global_change(self, idx, val):
        if self._syncing:
            return
        self.x[idx] = self._snap(idx, val)
        if idx in self.global_boxes:
            self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")
        self._refresh()

    def _on_comp_change(self, k, val):
        if self._syncing:
            return
        idx = self._comp_var_index(self.selected_comp, k)
        self.x[idx] = self._snap(idx, val)
        if k in self.comp_boxes:
            self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")
        self._refresh()

    def _on_select_component(self, val):
        self.selected_comp = int(val)
        comp_labels = ["panel", "x_loc", "y_loc", "face", "rotation"]
        for k, s in enumerate(self.comp_sliders):
            idx = self._comp_var_index(self.selected_comp, k)
            lo, hi, is_disc = self._var_bounds(idx)
            self._syncing = True
            s.valmin, s.valmax = lo, hi
            s.ax.set_xlim(lo, hi)
            s.set_val(self.x[idx])
            self._syncing = False
        # sync the per-component number boxes to the new component
        for k, tb in self.comp_boxes.items():
            idx = self._comp_var_index(self.selected_comp, k)
            tb.set_val(f"{self.x[idx]:.3f}")
        self.fig.canvas.draw_idle()

    def _on_reset_component(self, event):
        for k in range(5):
            idx = self._comp_var_index(self.selected_comp, k)
            self.x[idx] = self._var_default(idx)
        self._on_select_component(self.selected_comp)
        self._refresh()

    def _on_save(self, event):
        cost_list, violated, overlap = self.design.evaluate(list(self.x))
        record = {
            "x": [float(v) for v in self.x],
            "objectives": {name: float(c)
                           for name, c in zip(OBJECTIVE_NAMES, cost_list)},
            "constraint_violated": bool(violated),
            "overlap_score": float(overlap),
        }
        with open(DEMO_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Saved demo -> {DEMO_FILE}")

    # ---------- evaluation + redraw ----------
    def _refresh(self):
        """Rebuild geometry, re-evaluate, and redraw both the 3D view and text panel."""
        x = list(self.x)

        # --- run the real evaluation ---
        try:
            cost_list, violated, overlap = self.design.evaluate(x)
        except Exception as e:
            self._draw_error(str(e))
            return

        # get panels + components for drawing (safe even if constraint violated)
        try:
            struct_panels, comp_list = self.design.get_panels_and_components(x)
        except Exception as e:
            self._draw_error(f"geometry error: {e}")
            return

        self._draw_config(struct_panels, comp_list)
        self._draw_metrics(cost_list, violated, overlap)
        self.fig.canvas.draw_idle()

    def _draw_config(self, struct_panels, comp_list):
        """Interactive analog of config_visualization, using getCube [5]."""
        self.ax3d.cla()

        # structural panels (translucent gray), matching visualization.py style [2]
        for panel in struct_panels:
            if panel is not None:
                xP, yP, zP = getCube(panel.dimensions, panel.location, panel.orientation)
                self.ax3d.plot_surface(xP, yP, zP, alpha=0.1, color="tab:gray")

        # components (highlight the currently selected one)
        for i, comp in enumerate(comp_list):
            xC, yC, zC = getCube(comp.dimensions, comp.location, comp.orientation)
            if i == self.selected_comp:
                self.ax3d.plot_surface(xC, yC, zC, color="tab:red", alpha=0.9)
            else:
                self.ax3d.plot_surface(xC, yC, zC, alpha=0.6)

        self.ax3d.set_xlim(-1, 1)
        self.ax3d.set_ylim(-1, 1)
        self.ax3d.set_zlim(-1, 1)
        self.ax3d.set_title(
            f"Editing component {self.selected_comp} "
            f"({comp_list[self.selected_comp].type})"
        )

    def _draw_metrics(self, cost_list, violated, overlap):
        lines = ["OBJECTIVES", "-" * 28]
        for name, val in zip(OBJECTIVE_NAMES, cost_list):
            lines.append(f"{name:<20s}{val:>8.4f}")
        lines += [
            "",
            "CONSTRAINTS",
            "-" * 28,
            f"{'constraint_violated':<20s}{str(violated):>8s}",
            f"{'overlap_score':<20s}{overlap:>8.4f}",
        ]
        # NOTE: evaluate() returns [-1]*num_objectives when the design is
        # invalid (bad panel choice or pointing component on a shelf) [6],
        # so an all -1 objective row means "infeasible design".
        color = "tab:red" if violated else "black"
        self.text_handle.set_text("\n".join(lines))
        self.text_handle.set_color(color)

    def _draw_error(self, msg):
        self.text_handle.set_text(f"ERROR:\n{msg}")
        self.text_handle.set_color("tab:red")
        self.fig.canvas.draw_idle()

    def _set_slider_silent(self, slider, value):
        """Update a slider's displayed value without firing its callback."""
        self._syncing = True
        slider.set_val(value)
        self._syncing = False

    def _on_global_box(self, idx, text):
        try:
            val = float(text)
        except ValueError:
            self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")  # revert
            return
        lo, hi, _ = self._var_bounds(idx)
        val = float(np.clip(val, lo, hi))
        self.x[idx] = self._snap(idx, val)
        self._set_slider_silent(self.global_sliders[idx], self.x[idx])
        self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")  # show snapped value
        self._refresh()

    def _on_comp_box(self, k, text):
        idx = self._comp_var_index(self.selected_comp, k)
        try:
            val = float(text)
        except ValueError:
            self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")  # revert
            return
        lo, hi, _ = self._var_bounds(idx)
        val = float(np.clip(val, lo, hi))
        self.x[idx] = self._snap(idx, val)
        self._set_slider_silent(self.comp_sliders[k], self.x[idx])
        self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")
        self._refresh()


def launch(design):
    tool = DesignTool(design)
    plt.show()
    return tool


if __name__ == "__main__":
    # --- construct your environment here ---
    # The Chiplet_Configuration_Design constructor takes (component_list,
    # structure_panel) [6], so you need a base StructPanel instance [3].
    from utils.evaluation import Chiplet_Configuration_Design
    from utils.component_classes import StructPanel
    from utils.component_list import getComponents

    component_list, transfer_learning_components = getComponents()
    base_panel = StructPanel()
    eval_function = Chiplet_Configuration_Design(component_list, base_panel)

    launch(eval_function)