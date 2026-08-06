import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

# --- adjust to your project ---
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents

# 8 corner sign combinations for a unit box (body frame)
_CORNER_SIGNS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=float)

# 12 triangles (2 per face), indexing into the 8 corners
_TRI_I = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 0, 0]
_TRI_J = [1, 2, 4, 3, 5, 6, 5, 6, 6, 7, 3, 7]
_TRI_K = [2, 3, 5, 7, 6, 7, 6, 2, 7, 3, 7, 4]


def cuboid_mesh(dimensions, location, orientation):
    """Return (x, y, z, i, j, k) arrays for a Plotly Mesh3d cuboid.

    dimensions: [dx, dy, dz]  (StructPanel stores [w, h, thickness] [3])
    location:   [x, y, z]
    orientation: 3x3 DCM (same convention used by getCube [5])
    """
    half = np.array(dimensions, dtype=float) / 2.0
    corners = _CORNER_SIGNS * half            # (8, 3) body-frame corners
    R = np.asarray(orientation, dtype=float)
    rotated = (R @ corners.T).T               # apply DCM, matching getCube [5]
    world = rotated + np.asarray(location, dtype=float)
    x, y, z = world[:, 0], world[:, 1], world[:, 2]
    return x, y, z, _TRI_I, _TRI_J, _TRI_K


OBJECTIVE_NAMES = [
    "Center of Mass", "Off-Axis Inertia", "On-Axis Inertia",
    "Wire Cost", "Thermal Cost",
]
DEMO_FILE = "demonstrations.jsonl"

# variables that get a linked number input alongside the slider
# global: x_dim (1), y_dim (2); per-component: x_loc (1), y_loc (2), rotation (4)
GLOBAL_NUMERIC = {1, 2}
COMP_NUMERIC = {1, 2, 4}

GLOBAL_LABELS = ["shape (0=tri,1=rect,2=hex)", "x_dim", "y_dim", "z_dim", "num_shelves"]
COMP_LABELS = ["panel", "x_loc", "y_loc", "face", "rotation"]


class DesignApp:
    def __init__(self, design):
        self.design = design
        self.space = design.design_space
        self.num_components = design.num_components
        self.n_vars = len(self.space)
        self.x = [self._default(i) for i in range(self.n_vars)]

        self.app = Dash(__name__)
        self._build_layout()
        self._register_callbacks()

    # ---------- design-space helpers ----------
    def _bounds(self, i):
        spec = self.space[i]
        rng = spec["range"]
        if spec["type"] == "discrete":
            return min(rng), max(rng), True
        return rng[0], rng[1], False

    def _default(self, i):
        return self.space[i]["range"][0]

    def _snap(self, i, val):
        spec = self.space[i]
        if spec["type"] == "discrete":
            allowed = np.array(spec["range"], dtype=float)
            return float(allowed[np.argmin(np.abs(allowed - val))])
        return float(val)

    def _comp_idx(self, comp, k):
        return 5 + 5 * comp + k

    def _slider_step(self, i):
        # discrete vars snap to their spacing; continuous get a fine step
        lo, hi, is_disc = self._bounds(i)
        spec = self.space[i]
        if is_disc:
            r = sorted(spec["range"])
            return (r[1] - r[0]) if len(r) > 1 else 1
        return (hi - lo) / 200.0

    # ---------- layout ----------
    def _make_control(self, var_idx, label, slider_id, input_id, numeric):
        lo, hi, _ = self._bounds(var_idx)
        step = self._slider_step(var_idx)
        children = [
            html.Label(label, style={"fontSize": "12px"}),
            dcc.Slider(
                id=slider_id, min=lo, max=hi, step=step, value=self.x[var_idx],
                marks=None, tooltip={"placement": "bottom", "always_visible": True},
                updatemode="mouseup",  # only fire on release -> avoids lag
            ),
        ]
        if numeric:
            children.append(
                dcc.Input(id=input_id, type="number", value=round(self.x[var_idx], 4),
                          min=lo, max=hi, step=step, debounce=True,
                          style={"width": "100px"})
            )
        return html.Div(children, style={"marginBottom": "8px"})

    def _build_layout(self):
        # global controls
        global_controls = []
        for i in range(5):
            global_controls.append(self._make_control(
                i, GLOBAL_LABELS[i],
                slider_id={"kind": "global-slider", "index": i},
                input_id={"kind": "global-input", "index": i},
                numeric=(i in GLOBAL_NUMERIC),
            ))

        # per-component controls (bound to selected component; start at comp 0)
        comp_controls = []
        for k in range(5):
            idx = self._comp_idx(0, k)
            comp_controls.append(self._make_control(
                idx, COMP_LABELS[k],
                slider_id={"kind": "comp-slider", "index": k},
                input_id={"kind": "comp-input", "index": k},
                numeric=(k in COMP_NUMERIC),
            ))

        self.app.layout = html.Div([
            html.H3("Spacecraft Design Tool"),
            html.Div([
                # left: 3D view
                html.Div([dcc.Graph(id="view3d", style={"height": "700px"})],
                         style={"width": "55%", "display": "inline-block"}),
                # right: controls + metrics
                html.Div([
                    html.H4("Global"),
                    *global_controls,
                    html.Hr(),
                    html.H4("Component"),
                    dcc.Dropdown(
                        id="comp-select",
                        options=[{"label": f"{i}: {c.type}", "value": i}
                                 for i, c in enumerate(self.design.component_list)],
                        value=0, clearable=False,
                    ),
                    *comp_controls,
                    html.Hr(),
                    html.Button("Save Demo", id="save-btn", n_clicks=0),
                    html.Div(id="save-status", style={"color": "green"}),
                    html.Pre(id="metrics", style={"fontFamily": "monospace"}),
                ], style={"width": "43%", "display": "inline-block",
                          "verticalAlign": "top", "paddingLeft": "2%"}),
            ]),
            dcc.Store(id="x-store", data=self.x),
            dcc.Store(id="selected-comp", data=0),
        ])

    # ---------- figure ----------
    def _build_figure(self, x, selected_comp):
        try:
            struct_panels, comp_list = self.design.get_panels_and_components(list(x))
        except Exception as e:
            return go.Figure().update_layout(title=f"geometry error: {e}")

        traces = []
        for panel in struct_panels:
            if panel is not None:
                px, py, pz, i, j, k = cuboid_mesh(
                    panel.dimensions, panel.location, panel.orientation)
                traces.append(go.Mesh3d(
                    x=px, y=py, z=pz, i=i, j=j, k=k,
                    color="lightgray", opacity=0.15, flatshading=True,
                    hoverinfo="skip", showscale=False))

        for ci, comp in enumerate(comp_list):
            cx, cy, cz, i, j, k = cuboid_mesh(
                comp.dimensions, comp.location, comp.orientation)
            is_sel = (ci == selected_comp)
            traces.append(go.Mesh3d(
                x=cx, y=cy, z=cz, i=i, j=j, k=k,
                color="crimson" if is_sel else "steelblue",
                opacity=0.95 if is_sel else 0.7, flatshading=True,
                name=f"{ci}: {comp.type}", showscale=False,
                hovertext=f"{ci}: {comp.type}"))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[-1, 1]), aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=30, b=0), uirevision="keep",
            title=f"Editing component {selected_comp}",
        )
        return fig

    def _format_metrics(self, x):
        try:
            cost_list, violated, overlap = self.design.evaluate(list(x))
        except Exception as e:
            return f"evaluation error: {e}"
        lines = ["OBJECTIVES", "-" * 30]
        for name, val in zip(OBJECTIVE_NAMES, cost_list):
            lines.append(f"{name:<20s}{val:>9.4f}")
        # evaluate() returns [-1]*num_objectives for infeasible designs
        # (bad panel choice, or a pointing component placed on a shelf) [6]
        infeasible = all(v == -1 for v in cost_list)
        lines += ["", "CONSTRAINTS", "-" * 30,
                  f"{'constraint_violated':<20s}{str(bool(violated)):>9s}",
                  f"{'overlap_score':<20s}{overlap:>9.4f}"]
        if infeasible:
            lines += ["", ">> INFEASIBLE DESIGN (all objectives = -1)"]
        return "\n".join(lines)

    # ---------- callbacks ----------
    def _register_callbacks(self):
        app = self.app

        # When the component selector changes, sync the 5 per-component
        # sliders/inputs to that component's current values in x.
        @app.callback(
            Output({"kind": "comp-slider", "index": ALL}, "value"),
            Output({"kind": "comp-input", "index": ALL}, "value"),
            Output({"kind": "comp-slider", "index": ALL}, "min"),
            Output({"kind": "comp-slider", "index": ALL}, "max"),
            Output({"kind": "comp-slider", "index": ALL}, "step"),
            Output("selected-comp", "data"),
            Input("comp-select", "value"),
            State("x-store", "data"),
            prevent_initial_call=True,
        )
        def _on_select_component(comp, x):
            slider_vals, input_vals = [], []
            mins, maxes, steps = [], [], []
            for k in range(5):
                idx = self._comp_idx(comp, k)
                lo, hi, _ = self._bounds(idx)
                slider_vals.append(x[idx])
                mins.append(lo)
                maxes.append(hi)
                steps.append(self._slider_step(idx))
                # inputs only exist for COMP_NUMERIC (x_loc, y_loc, rotation),
                # but Dash still returns one value per matched component in order
                input_vals.append(round(x[idx], 4) if k in COMP_NUMERIC else None)
            # only the numeric-input components actually consume input_vals;
            # filter to those that exist in layout order (k in COMP_NUMERIC)
            numeric_inputs = [round(x[self._comp_idx(comp, k)], 4)
                              for k in sorted(COMP_NUMERIC)]
            return slider_vals, numeric_inputs, mins, maxes, steps, comp

        # Main update: any control (or the selector) changes -> rebuild x,
        # snap values, redraw the Mesh3d view, and re-evaluate the design.
        @app.callback(
            Output("view3d", "figure"),
            Output("metrics", "children"),
            Output("x-store", "data"),
            Input({"kind": "global-slider", "index": ALL}, "value"),
            Input({"kind": "global-input", "index": ALL}, "value"),
            Input({"kind": "comp-slider", "index": ALL}, "value"),
            Input({"kind": "comp-input", "index": ALL}, "value"),
            Input("selected-comp", "data"),
            State("x-store", "data"),
        )
        def _update(global_sliders, global_inputs, comp_sliders,
                    comp_inputs, selected_comp, x):
            x = list(x)
            trigger = ctx.triggered_id

            # --- global sliders ---
            for i, val in enumerate(global_sliders):
                if val is not None:
                    x[i] = self._snap(i, val)

            # --- global numeric inputs (override slider if that box fired) ---
            if isinstance(trigger, dict) and trigger.get("kind") == "global-input":
                gi = sorted(GLOBAL_NUMERIC)
                for pos, i in enumerate(gi):
                    if global_inputs[pos] is not None:
                        lo, hi, _ = self._bounds(i)
                        v = float(np.clip(global_inputs[pos], lo, hi))
                        x[i] = self._snap(i, v)

            # --- per-component sliders (apply to the selected component) ---
            for k, val in enumerate(comp_sliders):
                if val is not None:
                    idx = self._comp_idx(selected_comp, k)
                    x[idx] = self._snap(idx, val)

            # --- per-component numeric inputs ---
            if isinstance(trigger, dict) and trigger.get("kind") == "comp-input":
                ci = sorted(COMP_NUMERIC)
                for pos, k in enumerate(ci):
                    if comp_inputs[pos] is not None:
                        idx = self._comp_idx(selected_comp, k)
                        lo, hi, _ = self._bounds(idx)
                        v = float(np.clip(comp_inputs[pos], lo, hi))
                        x[idx] = self._snap(idx, v)

            self.x = x
            fig = self._build_figure(x, selected_comp)
            metrics = self._format_metrics(x)
            return fig, metrics, x

        # Save the current design vector + metrics to the demo file.
        @app.callback(
            Output("save-status", "children"),
            Input("save-btn", "n_clicks"),
            State("x-store", "data"),
            prevent_initial_call=True,
        )
        def _on_save(n_clicks, x):
            cost_list, violated, overlap = self.design.evaluate(list(x))
            record = {
                "x": [float(v) for v in x],
                "objectives": {name: float(c)
                               for name, c in zip(OBJECTIVE_NAMES, cost_list)},
                "constraint_violated": bool(violated),
                "overlap_score": float(overlap),
            }
            with open(DEMO_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
            return f"Saved demo #{n_clicks} to {DEMO_FILE}"

    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    base_panel = StructPanel()  # default aluminum 1x1 panel, 1 cm thick [3]
    component_list, transfer_learning_components = getComponents()
    design = Chiplet_Configuration_Design(component_list, base_panel)
    DesignApp(design).run()















# """
# Interactive spacecraft design tool for generating imitation-learning demonstrations.

# Controls:
#   - 5 global sliders (always visible): shape, x_dim, y_dim, z_dim, num_shelves
#   - A "component selector" slider that picks which component you're editing
#   - 5 per-component sliders for the currently selected component
#   - "Save Demo" button appends the current design vector + metrics to a file
#   - "Reset" button restores the last saved / initial vector

# Discrete variables snap to their allowed 'range' from design_space.
# Continuous variables use their [min, max] bounds.
# """

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, TextBox
# from copy import deepcopy

# # --- adjust these imports to your project ---
# from utils.config_utils import getCube
# from utils.evaluation import Chiplet_Configuration_Design

# OBJECTIVE_NAMES = [
#     "Center of Mass",
#     "Off-Axis Inertia",
#     "On-Axis Inertia",
#     "Wire Cost",
#     "Thermal Cost",
# ]

# DEMO_FILE = "demonstrations.jsonl"


# class DesignTool:
#     def __init__(self, design):
#         """
#         design: an instance of your evaluation class.
#                 Must expose:
#                   - design.design_space : list of dicts, one per variable in x,
#                         each {'type': 'discrete', 'range': [...]}  OR
#                               {'type': 'continuous', 'range': [min, max]}
#                   - design.evaluate(x) -> (cost_list, constraint_violated, overlap_score)
#                   - design.get_panels_and_components(x) -> (struct_panels, component_list)
#                   - design.num_components, design.num_objectives, design.max_shelves
#         """
#         self.design = design
#         self.space = design.design_space
#         self.num_components = design.num_components
#         self.n_vars = len(self.space)

#         # start at the low end of each variable's range (valid + deterministic)
#         self.x = [self._var_default(i) for i in range(self.n_vars)]

#         self.selected_comp = 0  # which component is currently being edited
#         self._syncing = False
#         self._build_figure()
#         self._refresh()  # initial draw + evaluation

#     # ---------- design-space helpers ----------
#     def _var_bounds(self, i):
#         spec = self.space[i]
#         rng = spec["range"]
#         if spec["type"] == "discrete":
#             return min(rng), max(rng), True  # (lo, hi, is_discrete)
#         else:
#             return rng[0], rng[1], False

#     def _var_default(self, i):
#         spec = self.space[i]
#         if spec["type"] == "discrete":
#             return spec["range"][0]
#         return spec["range"][0]

#     def _snap(self, i, val):
#         """Snap a slider value to the nearest allowed value for discrete vars."""
#         spec = self.space[i]
#         if spec["type"] == "discrete":
#             allowed = np.array(spec["range"], dtype=float)
#             return float(allowed[np.argmin(np.abs(allowed - val))])
#         return float(val)

#     # index helpers for the flat vector x
#     def _comp_var_index(self, comp, k):
#         # global vars are indices 0..4; component vars start at 5
#         # order per component: panel, x_loc, y_loc, face, rot
#         return 5 + 5 * comp + k

#     # ---------- figure construction ----------
#     def _build_figure(self):
#         self.fig = plt.figure(figsize=(14, 8))
#         self.ax3d = self.fig.add_axes([0.05, 0.25, 0.5, 0.7], projection="3d")

#         # text panel for objectives / constraints
#         self.ax_text = self.fig.add_axes([0.60, 0.55, 0.38, 0.40])
#         self.ax_text.axis("off")
#         self.text_handle = self.ax_text.text(
#             0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10
#         )

#         # --- global sliders (bottom strip) ---
#         self.global_sliders = []
#         global_labels = ["shape (0=tri,1=rect,2=hex)",
#                          "x_dim", "y_dim", "z_dim", "num_shelves"]
#         for i in range(5):
#             ax = self.fig.add_axes([0.08, 0.18 - i * 0.035, 0.45, 0.02])
#             lo, hi, is_disc = self._var_bounds(i)
#             step = 1.0 if is_disc else None
#             s = Slider(ax, global_labels[i], lo, hi,
#                        valinit=self.x[i], valstep=step)
#             s.on_changed(lambda val, idx=i: self._on_global_change(idx, val))
#             self.global_sliders.append(s)

#         self.global_boxes = {}
#         for i in (1, 2):
#             ax_box = self.fig.add_axes([0.545, 0.18 - i * 0.035, 0.045, 0.02])
#             tb = TextBox(ax_box, "", initial=f"{self.x[i]:.3f}")
#             tb.on_submit(lambda text, idx=i: self._on_global_box(idx, text))
#             self.global_boxes[i] = tb

#         # --- component selector ---
#         ax_sel = self.fig.add_axes([0.62, 0.42, 0.30, 0.02])
#         self.comp_selector = Slider(
#             ax_sel, "component #", 0, self.num_components - 1,
#             valinit=0, valstep=1
#         )
#         self.comp_selector.on_changed(self._on_select_component)

#         # --- per-component sliders ---
#         self.comp_sliders = []
#         comp_labels = ["panel", "x_loc", "y_loc", "face", "rotation"]
#         for k in range(5):
#             ax = self.fig.add_axes([0.62, 0.36 - k * 0.035, 0.30, 0.02])
#             idx = self._comp_var_index(self.selected_comp, k)
#             lo, hi, is_disc = self._var_bounds(idx)
#             step = 1.0 if is_disc else None
#             s = Slider(ax, comp_labels[k], lo, hi,
#                        valinit=self.x[idx], valstep=step)
#             s.on_changed(lambda val, kk=k: self._on_comp_change(kk, val))
#             self.comp_sliders.append(s)

#         # number boxes for x_loc (1), y_loc (2), rotation (4)
#         self.comp_boxes = {}
#         for k in (1, 2, 4):
#             ax_box = self.fig.add_axes([0.925, 0.36 - k * 0.035, 0.05, 0.02])
#             idx = self._comp_var_index(self.selected_comp, k)
#             tb = TextBox(ax_box, "", initial=f"{self.x[idx]:.3f}")
#             tb.on_submit(lambda text, kk=k: self._on_comp_box(kk, text))
#             self.comp_boxes[k] = tb

#         # --- buttons ---
#         ax_save = self.fig.add_axes([0.62, 0.10, 0.12, 0.04])
#         self.btn_save = Button(ax_save, "Save Demo")
#         self.btn_save.on_clicked(self._on_save)

#         ax_reset = self.fig.add_axes([0.78, 0.10, 0.12, 0.04])
#         self.btn_reset = Button(ax_reset, "Reset Comp")
#         self.btn_reset.on_clicked(self._on_reset_component)

#     # ---------- callbacks ----------
#     def _on_global_change(self, idx, val):
#         if self._syncing:
#             return
#         self.x[idx] = self._snap(idx, val)
#         if idx in self.global_boxes:
#             self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")
#         self._refresh()

#     def _on_comp_change(self, k, val):
#         if self._syncing:
#             return
#         idx = self._comp_var_index(self.selected_comp, k)
#         self.x[idx] = self._snap(idx, val)
#         if k in self.comp_boxes:
#             self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")
#         self._refresh()

#     def _on_select_component(self, val):
#         self.selected_comp = int(val)
#         comp_labels = ["panel", "x_loc", "y_loc", "face", "rotation"]
#         for k, s in enumerate(self.comp_sliders):
#             idx = self._comp_var_index(self.selected_comp, k)
#             lo, hi, is_disc = self._var_bounds(idx)
#             self._syncing = True
#             s.valmin, s.valmax = lo, hi
#             s.ax.set_xlim(lo, hi)
#             s.set_val(self.x[idx])
#             self._syncing = False
#         # sync the per-component number boxes to the new component
#         for k, tb in self.comp_boxes.items():
#             idx = self._comp_var_index(self.selected_comp, k)
#             tb.set_val(f"{self.x[idx]:.3f}")
#         self.fig.canvas.draw_idle()

#     def _on_reset_component(self, event):
#         for k in range(5):
#             idx = self._comp_var_index(self.selected_comp, k)
#             self.x[idx] = self._var_default(idx)
#         self._on_select_component(self.selected_comp)
#         self._refresh()

#     def _on_save(self, event):
#         cost_list, violated, overlap = self.design.evaluate(list(self.x))
#         record = {
#             "x": [float(v) for v in self.x],
#             "objectives": {name: float(c)
#                            for name, c in zip(OBJECTIVE_NAMES, cost_list)},
#             "constraint_violated": bool(violated),
#             "overlap_score": float(overlap),
#         }
#         with open(DEMO_FILE, "a") as f:
#             f.write(json.dumps(record) + "\n")
#         print(f"Saved demo -> {DEMO_FILE}")

#     # ---------- evaluation + redraw ----------
#     def _refresh(self):
#         """Rebuild geometry, re-evaluate, and redraw both the 3D view and text panel."""
#         x = list(self.x)

#         # --- run the real evaluation ---
#         try:
#             cost_list, violated, overlap = self.design.evaluate(x)
#         except Exception as e:
#             self._draw_error(str(e))
#             return

#         # get panels + components for drawing (safe even if constraint violated)
#         try:
#             struct_panels, comp_list = self.design.get_panels_and_components(x)
#         except Exception as e:
#             self._draw_error(f"geometry error: {e}")
#             return

#         self._draw_config(struct_panels, comp_list)
#         self._draw_metrics(cost_list, violated, overlap)
#         self.fig.canvas.draw_idle()

#     def _draw_config(self, struct_panels, comp_list):
#         """Interactive analog of config_visualization, using getCube [5]."""
#         self.ax3d.cla()

#         # structural panels (translucent gray), matching visualization.py style [2]
#         for panel in struct_panels:
#             if panel is not None:
#                 xP, yP, zP = getCube(panel.dimensions, panel.location, panel.orientation)
#                 self.ax3d.plot_surface(xP, yP, zP, alpha=0.1, color="tab:gray")

#         # components (highlight the currently selected one)
#         for i, comp in enumerate(comp_list):
#             xC, yC, zC = getCube(comp.dimensions, comp.location, comp.orientation)
#             if i == self.selected_comp:
#                 self.ax3d.plot_surface(xC, yC, zC, color="tab:red", alpha=0.9)
#             else:
#                 self.ax3d.plot_surface(xC, yC, zC, alpha=0.6)

#         self.ax3d.set_xlim(-1, 1)
#         self.ax3d.set_ylim(-1, 1)
#         self.ax3d.set_zlim(-1, 1)
#         self.ax3d.set_title(
#             f"Editing component {self.selected_comp} "
#             f"({comp_list[self.selected_comp].type})"
#         )

#     def _draw_metrics(self, cost_list, violated, overlap):
#         lines = ["OBJECTIVES", "-" * 28]
#         for name, val in zip(OBJECTIVE_NAMES, cost_list):
#             lines.append(f"{name:<20s}{val:>8.4f}")
#         lines += [
#             "",
#             "CONSTRAINTS",
#             "-" * 28,
#             f"{'constraint_violated':<20s}{str(violated):>8s}",
#             f"{'overlap_score':<20s}{overlap:>8.4f}",
#         ]
#         # NOTE: evaluate() returns [-1]*num_objectives when the design is
#         # invalid (bad panel choice or pointing component on a shelf) [6],
#         # so an all -1 objective row means "infeasible design".
#         color = "tab:red" if violated else "black"
#         self.text_handle.set_text("\n".join(lines))
#         self.text_handle.set_color(color)

#     def _draw_error(self, msg):
#         self.text_handle.set_text(f"ERROR:\n{msg}")
#         self.text_handle.set_color("tab:red")
#         self.fig.canvas.draw_idle()

#     def _set_slider_silent(self, slider, value):
#         """Update a slider's displayed value without firing its callback."""
#         self._syncing = True
#         slider.set_val(value)
#         self._syncing = False

#     def _on_global_box(self, idx, text):
#         try:
#             val = float(text)
#         except ValueError:
#             self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")  # revert
#             return
#         lo, hi, _ = self._var_bounds(idx)
#         val = float(np.clip(val, lo, hi))
#         self.x[idx] = self._snap(idx, val)
#         self._set_slider_silent(self.global_sliders[idx], self.x[idx])
#         self.global_boxes[idx].set_val(f"{self.x[idx]:.3f}")  # show snapped value
#         self._refresh()

#     def _on_comp_box(self, k, text):
#         idx = self._comp_var_index(self.selected_comp, k)
#         try:
#             val = float(text)
#         except ValueError:
#             self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")  # revert
#             return
#         lo, hi, _ = self._var_bounds(idx)
#         val = float(np.clip(val, lo, hi))
#         self.x[idx] = self._snap(idx, val)
#         self._set_slider_silent(self.comp_sliders[k], self.x[idx])
#         self.comp_boxes[k].set_val(f"{self.x[idx]:.3f}")
#         self._refresh()


# def launch(design):
#     tool = DesignTool(design)
#     plt.show()
#     return tool


# if __name__ == "__main__":
#     # --- construct your environment here ---
#     # The Chiplet_Configuration_Design constructor takes (component_list,
#     # structure_panel) [6], so you need a base StructPanel instance [3].
#     from utils.evaluation import Chiplet_Configuration_Design
#     from utils.component_classes import StructPanel
#     from utils.component_list import getComponents

#     component_list, transfer_learning_components = getComponents()
#     base_panel = StructPanel()
#     eval_function = Chiplet_Configuration_Design(component_list, base_panel)

#     launch(eval_function)