# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "networkx==3.6.1",
#     "matplotlib==3.10.8",
#     "polars==1.23.0",
#     "pydantic==2.10.6",
#     "anywidget==0.9.21",
#     "traitlets==5.14.3",
#     "scipy==1.17.1",
# ]
# ///
# Run from project root with:  uv run marimo run tiny_eda.py   (or  uv run marimo edit tiny_eda.py)
# This ensures anywidget and other deps from pyproject.toml are available.

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import networkx as nx
    import matplotlib.pyplot as plt
    import polars as pl
    import random
    import math
    import anywidget
    import traitlets
    from enum import Enum
    from typing import List, Dict, Tuple, Set, Optional
    from pydantic import BaseModel, Field

    # Pydantic Models for Structured Hardware Data
    class BlockType(str, Enum):
        LUT = "LUT"
        BRAM = "BRAM"
        IO = "IO"

    class Coordinate(BaseModel):
        x: int
        y: int
        def to_tuple(self) -> Tuple[int, int]: return (self.x, self.y)

    class Placement(BaseModel):
        block_id: str
        block_type: BlockType
        coord: Coordinate

    class Route(BaseModel):
        u: str
        v: str
        path: List[Coordinate]

    class TimingResult(BaseModel):
        block_id: str
        block_type: BlockType
        arrival: float
        required: float
        slack: float
        is_critical: bool

    return (
        Coordinate,
        Route,
        TimingResult,
        anywidget,
        math,
        mo,
        nx,
        pl,
        plt,
        random,
        traitlets,
    )


@app.cell
def _():
    PALETTE = {
        "primary": "#2563eb",
        "primary_light": "#93c5fd",
        "primary_soft": "#38bdf8",
        "io": "#dc2626",
        "lut": "#16a34a",
        "bram": "#ea580c",
        "critical": "#eab308",
        "congestion_high": "#b91c1c",
        "neutral_fill": "#e2e8f0",
        "neutral_line": "#64748b",
        "ink": "#374151",
        "surface_dark": "#0f172a",
    }
    TRACE_COLORS = [
        PALETTE["primary_soft"],
        PALETTE["io"],
        PALETTE["lut"],
        PALETTE["critical"],
        "#a78bfa",
    ]
    return PALETTE, TRACE_COLORS


@app.cell
def _(anywidget, traitlets):
    class WaveformWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
            const container = document.createElement("div");
            const surfaceDark = model.get("surface_dark") || "#0f172a";
            const primarySoft = model.get("primary_soft") || "#38bdf8";
            container.style.padding = "20px";
            container.style.background = surfaceDark;
            container.style.borderRadius = "12px";
            container.style.color = primarySoft;
            container.style.fontFamily = "'JetBrains Mono', monospace";
            container.style.boxShadow = "0 4px 6px -1px rgb(0 0 0 / 0.1)";

            const title = document.createElement("div");
            title.style.fontSize = "12px";
            title.style.marginBottom = "10px";
            title.style.opacity = "0.8";
            container.appendChild(title);

            const canvas = document.createElement("canvas");
            canvas.width = 800;
            canvas.height = 220;
            canvas.style.width = "100%";
            const ctx = canvas.getContext("2d");

            const defaultColors = ["#38bdf8", "#dc2626", "#16a34a", "#eab308", "#a78bfa"];
            const maxVal = 255;

            function draw() {
                const traceColors = model.get("trace_colors");
                const colors = (traceColors && traceColors.length) ? traceColors : defaultColors;
                const wireNames = model.get("wire_names") || [];
                const wireValues = model.get("wire_values") || [];
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (wireNames.length === 0 || wireValues.length === 0) {
                    title.innerText = "DIGITAL LOGIC ANALYZER (select wires above)";
                    return;
                }

                const n = wireValues[0].length;
                if (n < 1) return;

                const stepWidth = (canvas.width - 20) / n;
                title.innerText = "DIGITAL LOGIC ANALYZER: " + wireNames.join(", ");

                // Grid (primary_soft tint; #38bdf8 = rgb 56,189,248)
                ctx.strokeStyle = "rgba(56, 189, 248, 0.1)";
                ctx.beginPath();
                const gridStep = n <= 25 ? 1 : Math.max(1, Math.floor(n / 15));
                for (let i = 0; i <= n; i += gridStep) {
                    const x = 10 + i * stepWidth;
                    ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
                }
                ctx.stroke();

                // One trace per wire (overlaid, same scale 0–255)
                const traceHeight = 170;
                const labelStep = n <= 20 ? 1 : Math.max(1, Math.floor(n / 12));

                for (let w = 0; w < wireValues.length; w++) {
                    const history = wireValues[w];
                    const name = wireNames[w] || ("W" + w);
                    const color = colors[w % colors.length];
                    ctx.strokeStyle = color;
                    ctx.lineWidth = Math.max(1.2, Math.min(3, stepWidth / 2));
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = color;
                    ctx.beginPath();
                    for (let i = 0; i < history.length; i++) {
                        const x = 10 + i * stepWidth;
                        const y = 190 - (Number(history[i]) % 256) / maxVal * traceHeight;
                        if (i === 0) ctx.moveTo(x, y);
                        else {
                            const prevY = 190 - (Number(history[i-1]) % 256) / maxVal * traceHeight;
                            ctx.lineTo(x, prevY);
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                    ctx.fillStyle = color;
                    ctx.font = (stepWidth > 24 ? "10px" : "8px") + " monospace";
                    if (wireValues.length <= 2) {
                        for (let i = 0; i < history.length; i += labelStep) {
                            const x = 10 + i * stepWidth + 2;
                            const y = 190 - (Number(history[i]) % 256) / maxVal * traceHeight - 2;
                            ctx.fillText(history[i].toString(), x, y);
                        }
                    }
                }

                // Legend at bottom
                ctx.font = "10px monospace";
                let legX = 10;
                for (let w = 0; w < wireNames.length; w++) {
                    ctx.fillStyle = colors[w % colors.length];
                    ctx.fillText(wireNames[w], legX, 212);
                    legX += ctx.measureText(wireNames[w]).width + 16;
                }
            }

            model.on("change:wire_names", draw);
            model.on("change:wire_values", draw);
            model.on("change:trace_colors", draw);
            container.appendChild(canvas);
            el.appendChild(container);
            draw();
        }
        export default { render };
        """
        wire_names = traitlets.List([]).tag(sync=True)
        wire_values = traitlets.List([]).tag(sync=True)
        trace_colors = traitlets.List([]).tag(sync=True)
        surface_dark = traitlets.Unicode("#0f172a").tag(sync=True)
        primary_soft = traitlets.Unicode("#38bdf8").tag(sync=True)

    return (WaveformWidget,)


@app.cell
def _(mo):
    mo.md(r"""
    # 🛠️ TinyEDA: The Architecture Explorer
    ### A Journey from Code to Hardware

    This notebook is for anyone who wants to see how the stages of EDA (synthesis, placement, routing, timing, simulation) connect. We are going to "compile" an **8-bit Fibonacci Calculator** onto a virtual FPGA (Field Programmable Gate Array).

    Follow the steps below to see how abstract math transforms into physical silicon layout. **By the end you will have seen synthesis, placement, routing, timing, and simulation in one flow.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The EDA Pipeline

    The flow has five stages; each consumes the output of the previous one. Real tools (e.g. Vivado, Quartus) do these same stages at scale—this notebook is a miniature version.

    1. **Synthesis** — Logic becomes a netlist (graph of gates and wires).
    2. **Placement** — Each block gets a physical $(x, y)$ slot on the FPGA grid.
    3. **Routing** — Wires get physical paths on the grid.
    4. **STA** — We compute the critical path and maximum clock frequency.
    5. **Simulation** — We run the design cycle-by-cycle and view waveforms.

    Order matters: you must place before you can route (you need block positions to know where wires go), and you need routes before you can do timing.
    """)
    return


@app.cell
def _(mo):
    _diagram = """
    flowchart LR
      Syn[Synthesis] -->|netlist| Plc[Placement]
      Plc -->|placement| Rte[Routing]
      Rte -->|routes| STA[STA]
      STA -->|Fmax| Sim[Simulation]
    """
    mo.mermaid(_diagram)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 📑 Stage 1: Synthesis (High-Level to Gates)

    **The Goal:** Convert a behavioral design description into a structure of wires and gates (a netlist). Choose a design below; the whole of Stage 1 updates to describe that design.
    """)
    return


@app.cell
def _(mo):
    design_selector = mo.ui.dropdown(
        options=["Fibonacci (8-bit)", "Pipelined Adder (2-stage)"],
        value="Fibonacci (8-bit)",
        label="Design",
    )
    _out = mo.vstack([mo.md("#### Select design"), design_selector])
    _out
    return (design_selector,)


@app.cell
def _(design_selector, mo):
    _fib = design_selector.value == "Fibonacci (8-bit)"
    _t_fib = (
        "**What is the HDL design description?** In a real FPGA flow, the *input* to synthesis is an **HDL** design. "
        "Here we start from the recurrence A_next = B, B_next = A + B; the graph below is the netlist.\n\n"
        "#### Block diagram: what the design does\n\n"
        "Data flows top to bottom. **REG_A** holds the previous Fibonacci value, **REG_B** the current one; "
        "the **adder** computes REG_A + REG_B; that sum becomes REG_B, and REG_B becomes REG_A. **RST** resets; **START** can trigger **FIB_MEM**."
    )
    _t_pipe = (
        "**What is the HDL design description?** In a real FPGA flow, the *input* to synthesis is an **HDL** design. "
        "Here we start from a 2-stage pipeline: REG_A gets REG_C, REG_B gets REG_A+1, REG_C gets REG_B+1.\n\n"
        "#### Block diagram: what the design does\n\n"
        "**REG_A** receives REG_C (next cycle). **ADD1** computes REG_A + 1 (constant **ONE**); result to **REG_B**. "
        "**ADD2** computes REG_B + 1; result to **REG_C**. **RST** resets all. No BRAM in this design."
    )
    mo.md(_t_fib if _fib else _t_pipe)
    return


@app.cell
def _(design_selector, mo):
    _fib = design_selector.value == "Fibonacci (8-bit)"
    _mermaid_fib = """
    flowchart TB
      subgraph inputs [Inputs]
        RST[RST]
        START[START]
      end
      subgraph regs [Registers]
        REG_A[REG_A 8-bit]
        REG_B[REG_B 8-bit]
      end
      ADD[Adder A+B]
      FIB[FIB_MEM]
      RST --> REG_A
      RST --> REG_B
      START --> FIB
      REG_A --> ADD
      REG_B --> ADD
      ADD --> REG_B
      REG_B --> REG_A
      REG_B --> FIB
    """
    _mermaid_pipe = """
    flowchart TB
      subgraph inputs [Inputs]
        RST[RST]
        START[START]
        ONE[ONE]
      end
      subgraph stage1 [Stage 1]
        REG_A[REG_A]
        ADD1[ADD1 A+1]
        REG_B[REG_B]
      end
      subgraph stage2 [Stage 2]
        ADD2[ADD2 B+1]
        REG_C[REG_C]
      end
      RST --> REG_A
      RST --> REG_B
      RST --> REG_C
      ONE --> ADD1
      ONE --> ADD2
      REG_A --> ADD1
      ADD1 --> REG_B
      REG_B --> ADD2
      ADD2 --> REG_C
      REG_C --> REG_A
    """
    mo.mermaid(_mermaid_fib if _fib else _mermaid_pipe)
    return


@app.cell
def _(design_selector, mo):
    _fib = design_selector.value == "Fibonacci (8-bit)"
    _verilog_fib = r"""
    #### The same design in Verilog (behavioral)

    Synthesis turns code like this into the netlist graph below.

    ```verilog
    module fib_8bit (
        input  wire       clk, rst, start,
        output wire [7:0] reg_b_out
    );
        reg [7:0] reg_a, reg_b;   // REG_A, REG_B

        always @(posedge clk) begin
            if (rst) begin
                reg_a <= 8'd0;
                reg_b <= 8'd1;
            end else begin
                reg_a <= reg_b;           // A_next = B
                reg_b <= reg_a + reg_b;   // B_next = A + B
            end
        end

        assign reg_b_out = reg_b;   // current Fibonacci number
    endmodule
    ```
    """
    _verilog_pipe = r"""
    #### Same idea in Verilog (behavioral)

    A 2-stage increment pipeline: REG_A gets REG_C, REG_B gets REG_A+1, REG_C gets REG_B+1.

    ```verilog
    module pipe_adder (
        input  wire       clk, rst,
        output wire [7:0] reg_c_out
    );
        reg [7:0] reg_a, reg_b, reg_c;

        always @(posedge clk) begin
            if (rst) begin
                reg_a <= 8'd0; reg_b <= 8'd0; reg_c <= 8'd0;
            end else begin
                reg_a <= reg_c;           // A_next = C
                reg_b <= reg_a + 8'd1;    // B_next = A + 1
                reg_c <= reg_b + 8'd1;    // C_next = B + 1
            end
        end

        assign reg_c_out = reg_c;
    endmodule
    ```
    """
    mo.md(_verilog_fib if _fib else _verilog_pipe)
    return


@app.cell
def _(design_selector, mo):
    _fib = design_selector.value == "Fibonacci (8-bit)"
    _net_fib = r"""
    **What is a netlist?** A netlist is a graph of components (nodes) and connections (edges) that the rest of the flow will place and connect on silicon.

    **Why this graph:** The recurrence becomes registers (REG_A, REG_B), an 8-bit adder (ADD_0..ADD_7 with carry chain), and a memory block (FIB_MEM). Edges show data flow: RST/START to REG_A/REG_B to Adder to REG_B, REG_B to REG_A, REG_B to FIB_MEM.

    **Legend:** **Green** = LUTs (adder + regs). **Orange** = BRAM (FIB_MEM). **Red** = IO (RST, START).
    """
    _net_pipe = r"""
    **What is a netlist?** A netlist is a graph of components (nodes) and connections (edges) that the rest of the flow will place and connect on silicon.

    **Why this graph:** The pipeline becomes three register banks (REG_A, REG_B, REG_C), two 8-bit adders (ADD1, ADD2) each adding the constant 1, and inputs RST, START, ONE. Edges show data flow: REG_A to ADD1 to REG_B to ADD2 to REG_C to REG_A.

    **Legend:** **Green** = LUTs (regs + adders). **Red** = IO (RST, START, ONE). No BRAM in this design.
    """
    mo.md(_net_fib if _fib else _net_pipe)
    return


@app.cell
def _(mo):
    design_selector = mo.ui.dropdown(
        options=["Fibonacci (8-bit)", "Pipelined Adder (2-stage)"],
        value="Fibonacci (8-bit)",
        label="Design",
    )
    _out = mo.vstack([mo.md("#### Select design"), design_selector])
    _out
    return (design_selector,)


@app.cell
def _(design_selector, nx):
    # Synthesis logic: dispatch by design
    _DESIGN_KEY_MAP = {"Fibonacci (8-bit)": "fibonacci", "Pipelined Adder (2-stage)": "pipelined_adder"}

    def _generate_fibonacci_netlist():
        _G = nx.DiGraph()
        _bits = 8
        _reg_a = [f"REG_A_{_i}" for _i in range(_bits)]
        _reg_b = [f"REG_B_{_i}" for _i in range(_bits)]
        _G.add_node("RST", type="input", block_type="IO")
        _G.add_node("START", type="input", block_type="IO")

        for _node in _reg_a + _reg_b:
            _G.add_node(_node, type="gate", block_type="LUT")
            _G.add_edge("RST", _node)

        _adder_gates = [f"ADD_{_i}" for _i in range(_bits)]
        for _i, _gate in enumerate(_adder_gates):
            _G.add_node(_gate, type="gate", block_type="LUT")
            _G.add_edge(_reg_a[_i], _gate, is_sequential=True)
            _G.add_edge(_reg_b[_i], _gate, is_sequential=True)
            if _i > 0: _G.add_edge(_adder_gates[_i-1], _gate)

        for _i in range(_bits):
            _G.add_edge(_adder_gates[_i], _reg_b[_i])
            _G.add_edge(_reg_b[_i], _reg_a[_i], is_sequential=True)

        _G.add_node("FIB_MEM", type="memory", block_type="BRAM")
        for _node in _reg_b: _G.add_edge(_node, "FIB_MEM")
        _G.add_edge("START", "FIB_MEM")
        return _G

    def _generate_pipelined_adder_netlist():
        _G = nx.DiGraph()
        _bits = 8
        _reg_a = [f"REG_A_{_i}" for _i in range(_bits)]
        _reg_b = [f"REG_B_{_i}" for _i in range(_bits)]
        _reg_c = [f"REG_C_{_i}" for _i in range(_bits)]
        _G.add_node("RST", type="input", block_type="IO")
        _G.add_node("START", type="input", block_type="IO")
        _G.add_node("ONE", type="input", block_type="IO")
        for _node in _reg_a + _reg_b + _reg_c:
            _G.add_node(_node, type="gate", block_type="LUT")
            _G.add_edge("RST", _node)
        _add1 = [f"ADD1_{_i}" for _i in range(_bits)]
        _add2 = [f"ADD2_{_i}" for _i in range(_bits)]
        for _i, _g in enumerate(_add1):
            _G.add_node(_g, type="gate", block_type="LUT")
            _G.add_edge("ONE", _g)
            _G.add_edge(_reg_a[_i], _g, is_sequential=True)
            if _i > 0: _G.add_edge(_add1[_i - 1], _g)
        for _i in range(_bits):
            _G.add_edge(_add1[_i], _reg_b[_i])
        for _i, _g in enumerate(_add2):
            _G.add_node(_g, type="gate", block_type="LUT")
            _G.add_edge("ONE", _g)
            _G.add_edge(_reg_b[_i], _g, is_sequential=True)
            if _i > 0: _G.add_edge(_add2[_i - 1], _g)
        for _i in range(_bits):
            _G.add_edge(_add2[_i], _reg_c[_i])
            _G.add_edge(_reg_c[_i], _reg_a[_i], is_sequential=True)
        return _G

    def _layout_fibonacci(G):
        return nx.nx_agraph.graphviz_layout(G, prog="dot")

    def _layout_pipelined_adder(G):
        return nx.nx_agraph.graphviz_layout(G, prog="dot")

    def _layout_fallback(G):
        return nx.nx_agraph.graphviz_layout(G, prog="dot")

    design_key = _DESIGN_KEY_MAP.get(design_selector.value, "fibonacci")
    _generators = {"fibonacci": _generate_fibonacci_netlist, "pipelined_adder": _generate_pipelined_adder_netlist}
    _layout_fns = {"fibonacci": _layout_fibonacci, "pipelined_adder": _layout_pipelined_adder}
    netlist = _generators.get(design_key, _generate_fibonacci_netlist)()
    current_layout_fn = _layout_fns.get(design_key, _layout_fallback)
    return current_layout_fn, design_key, netlist


@app.cell
def _(PALETTE, current_layout_fn, netlist, nx, plt):
    # Netlist with design-specific layout
    _pos = current_layout_fn(netlist)
    _fig, _ax = plt.subplots(figsize=(12, 6))
    _colors = [
        PALETTE["io"] if netlist.nodes[n].get("block_type") == "IO"
        else PALETTE["lut"] if netlist.nodes[n].get("block_type") == "LUT"
        else PALETTE["bram"]
        for n in netlist.nodes
    ]
    nx.draw(
        netlist, _pos,
        with_labels=True,
        node_color=_colors,
        node_size=700,
        font_size=6,
        ax=_ax,
        edge_color=PALETTE["neutral_line"],
        alpha=0.5,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
    )
    _ax.set_title("Synthesized Structural Netlist (8-bit Adder + Registers)")
    _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(design_selector, mo):
    _fib = design_selector.value == "Fibonacci (8-bit)"
    _cap_fib = "*Nodes = hardware blocks; edges = signals. Layout: inputs → REG_A / REG_B → adder chain → FIB_MEM. Physical positions come from Placement (Stage 2).*\n\n**Data flow:** RST/START → REG_A, REG_B → Adder → REG_B; REG_B → REG_A (next cycle); REG_B → FIB_MEM."
    _cap_pipe = "*Nodes = hardware blocks; edges = signals. Layout: inputs → REG_A → ADD1 → REG_B → ADD2 → REG_C → REG_A. Physical positions come from Placement (Stage 2).*\n\n**Data flow:** RST/ONE → ADD1, ADD2; REG_A → ADD1 → REG_B → ADD2 → REG_C → REG_A (next cycle)."
    mo.md(_cap_fib if _fib else _cap_pipe)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 📍 Stage 2: Placement (The Optimization Puzzle)

    **What is placement:** Placement assigns each netlist block to a physical $(x, y)$ slot on the FPGA grid. Good placement shortens wires and reduces routing congestion. Longer wires mean more delay and more congestion, so minimizing wire length helps timing and routability.

    **The Algorithm:** Simulated Annealing. We start from a random placement; we allow random swaps that sometimes increase cost (to escape local minima); acceptance probability decreases as "temperature" drops so the layout settles into a good solution. **Scrub the slider below** to watch the design transition from chaotic noise into a clustered, efficient machine.

    **HPWL (Half-Perimeter Wire Length):** For each **net** (one connection between two blocks), imagine the two blocks at positions $(x_1, y_1)$ and $(x_2, y_2)$. The "bounding box" is the smallest rectangle that contains both points. The **half-perimeter** of that rectangle is $(x_2 - x_1) + (y_2 - y_1)$—that's the Manhattan distance between the two points, and it approximates how much wire we need for that net. We add that length for every net in the design; that sum is our **wirelength cost**. Placement tries to minimize this cost so that connected blocks sit close together and wires stay short.

    **Architecture:** Gates stay in LUT slots; memory stays in BRAM slots. Shaded columns (e.g. 4, 9) = BRAM columns; other cells = LUT slots.

    **Try a smaller board:** Use the control below to force synthesis onto a smaller N×N grid. Fewer slots mean tighter placement and more routing congestion; the design needs at least a 6×6 to fit (24 LUTs + 1 BRAM).
    """)
    return


@app.cell
def _(mo):
    board_size = mo.ui.dropdown(
        options=[6, 7, 8, 9, 10],
        value=10,
        label="Board size (N×N grid)",
    )
    _out = mo.vstack([mo.md("#### Grid size"), board_size])
    _out
    return (board_size,)


@app.cell
def _(board_size, math, netlist, random):
    # Placement Solver (N from board_size)
    def run_placement(graph, N=10, iterations=2000):
        _bram_cols = [4, 9] if N >= 10 else [N // 3, 2 * N // 3]
        _lut_slots = [(x, y) for x in range(N) for y in range(N) if x not in _bram_cols]
        _bram_slots = [(x, y) for x in range(N) for y in range(N) if x in _bram_cols]

        _pl = {}
        _luts = [n for n in graph.nodes if graph.nodes[n]['block_type'] == "LUT"]
        _brams = [n for n in graph.nodes if graph.nodes[n]['block_type'] == "BRAM"]

        random.shuffle(_lut_slots); random.shuffle(_bram_slots)
        for _i, _n in enumerate(_luts): _pl[_n] = _lut_slots[_i]
        for _i, _n in enumerate(_brams): _pl[_n] = _bram_slots[_i]

        def _cost(p):
            return sum(abs(p[u][0]-p[v][0]) + abs(p[u][1]-p[v][1]) for u, v in graph.edges() if u in p and v in p)

        _history = [(_pl.copy(), _cost(_pl))]
        _curr_pl, _curr_cost, _T = _pl.copy(), _cost(_pl), 100.0

        for _s in range(iterations):
            _u = random.choice(list(graph.nodes))
            if _u not in _curr_pl: continue
            _type = graph.nodes[_u]['block_type']
            _slots = _bram_slots if _type == "BRAM" else _lut_slots
            _target = random.choice(_slots)

            _v = next((_n for _n, _c in _curr_pl.items() if _c == _target and _n != _u), None)
            _next_pl = _curr_pl.copy()
            if _v: _next_pl[_u], _next_pl[_v] = _next_pl[_v], _next_pl[_u]
            else: _next_pl[_u] = _target

            _next_cost = _cost(_next_pl)
            if _next_cost < _curr_cost or random.random() < math.exp((_curr_cost - _next_cost)/_T):
                _curr_pl, _curr_cost = _next_pl, _next_cost
            if _s % 40 == 0: _history.append((_curr_pl.copy(), _curr_cost))
            _T *= 0.99

        return _history

    _n = int(board_size.value)
    pl_history = run_placement(netlist, N=_n)
    return (pl_history,)


@app.cell
def _(mo, netlist, pl_history):
    # Placement scrub UI + net selector for HPWL illustration
    _final_pl = pl_history[-1][0]
    _edge_options = [
        f"{u} → {v}" for u, v in netlist.edges()
        if u in _final_pl and v in _final_pl
    ]
    hpwl_net_selector = mo.ui.dropdown(
        options=_edge_options,
        value=_edge_options[0] if _edge_options else None,
        label="Show HPWL (bounding box) for net",
    )
    scrub = mo.ui.slider(0, len(pl_history) - 1, label="Optimization Timeline")
    _out = mo.vstack([
        mo.md("#### 🔍 Algorithm Playback"),
        scrub,
        mo.md("**HPWL:** Half-perimeter of the net's bounding box = |Δx| + |Δy|. Pick a net to highlight on the grid below."),
        hpwl_net_selector,
    ])
    _out
    return hpwl_net_selector, scrub


@app.cell
def _(PALETTE, board_size, hpwl_net_selector, netlist, pl_history, plt, scrub):
    # Render Placement Snapshot and Cost Curve + HPWL illustration
    _n = int(board_size.value)
    _bram_cols = [4, 9] if _n >= 10 else [_n // 3, 2 * _n // 3]
    _curr_pl, _cost = pl_history[scrub.value]
    _fig, (_ax_cost, _ax) = plt.subplots(2, 1, figsize=(6, 6.5), height_ratios=[0.45, 1])

    # Cost curve (wirelength vs step)
    _steps = [i * 40 for i in range(len(pl_history))]
    _costs = [pl_history[i][1] for i in range(len(pl_history))]
    _ax_cost.plot(_steps, _costs, color=PALETTE["primary_soft"], lw=1.5)
    _ax_cost.axvline(x=scrub.value * 40, color=PALETTE["io"], linestyle='--', alpha=0.8)
    _ax_cost.set_ylabel("Wirelength cost")
    _ax_cost.set_xlabel("Optimization step")
    _ax_cost.set_title("Cost vs. step (slider = current step)")
    _ax_cost.grid(True, alpha=0.3)

    # Placement grid (size from board_size)
    _ax.set_xticks(range(_n + 1))
    _ax.set_yticks(range(_n + 1))
    _ax.grid(True, alpha=0.1)
    _ax.set_xlim(-0.5, _n + 0.5)
    _ax.set_ylim(-0.5, _n + 0.5)
    for _x in _bram_cols:
        _ax.add_patch(plt.Rectangle((_x, 0), 1, _n, color=PALETTE["bram"], alpha=0.1))

    # HPWL illustration: bounding box for selected net
    _sel = hpwl_net_selector.value
    if _sel and " → " in _sel:
        _u, _v = _sel.split(" → ", 1)
        if _u in _curr_pl and _v in _curr_pl:
            _x1, _y1 = _curr_pl[_u]
            _x2, _y2 = _curr_pl[_v]
            _minx, _maxx = min(_x1, _x2), max(_x1, _x2)
            _miny, _maxy = min(_y1, _y2), max(_y1, _y2)
            _dx, _dy = abs(_x2 - _x1), abs(_y2 - _y1)
            _hpwl = _dx + _dy
            _ax.add_patch(plt.Rectangle(
                (_minx, _miny), _maxx - _minx + 1, _maxy - _miny + 1,
                fill=True, facecolor=PALETTE["primary_soft"], alpha=0.12,
                edgecolor=PALETTE["primary"], linewidth=1.5, linestyle="--", zorder=0,
            ))
            _ax.plot([_x1 + 0.5, _x2 + 0.5], [_y1 + 0.5, _y2 + 0.5], color=PALETTE["primary"], lw=1, ls=":", alpha=0.7, zorder=0)
            _midx, _midy = (_minx + _maxx + 1) / 2, (_miny + _maxy + 1) / 2
            _ax.text(_midx, _midy, f"HPWL = {_hpwl}\n|Δx|+|Δy|", fontsize=8, ha="center", va="center", color=PALETTE["primary"], fontweight="bold")

    for _node, (_px, _py) in _curr_pl.items():
        _c = PALETTE["bram"] if netlist.nodes[_node]['block_type'] == "BRAM" else PALETTE["lut"]
        _ax.add_patch(plt.Rectangle((_px+0.1, _py+0.1), 0.8, 0.8, color=_c, ec='k', alpha=0.9, lw=0.5))
        _ax.text(_px+0.5, _py+0.5, _node.split('_')[-1], ha='center', va='center', fontsize=6, fontweight='bold')

    _ax.set_title(f"Step {scrub.value*40} | Wirelength Cost (Energy): {_cost}")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Each rectangle is one block at its assigned slot. Scrub the slider to see cost decrease as the layout improves. Use the **HPWL** dropdown to highlight one net's bounding box: the dashed rectangle is the smallest box containing both endpoints; HPWL = |Δx| + |Δy| is the half-perimeter (Manhattan distance) and is that net's contribution to the wirelength cost.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 🌉 Stage 3: Routing (Interconnect)

    **What is routing:** Routing assigns a physical path (a sequence of wire segments on the grid) to each net so that every connection is realized. It's hard because there are many nets, a limited set of segments, and no two signals can share the same segment without shorting. We use a **negotiated congestion** approach: paths "negotiate" by changing the cost of shared resources.

    **The algorithm (Pathfinder-style), step by step:**

    1. **Model the chip as a graph.** The placement grid becomes a graph: each cell is a vertex, each adjacent pair is an edge. Every edge has a **cost** (we start with cost 1 everywhere).

    2. **Route one net at a time.** For each net (source block → target block), we find a **shortest path** on this graph using **A\***: from the source cell, the algorithm expands to neighboring cells in order of lowest cost-to-reach, until it reaches the target. The path is the sequence of edges along that shortest route.

    3. **Increase cost on used edges (negotiation).** After routing a net, we **add a penalty** (e.g. 1.5) to the cost of every edge that net used. So the next net sees those edges as more expensive and is encouraged to use other segments when possible. That's "negotiated congestion": early nets get cheap paths; later nets avoid crowding the same wires.

    4. **Repeat.** We repeat steps 2–3 for every net. In a full Pathfinder flow, we would then check for overused edges and **re-iterate** (rip up and reroute) until no edge is over capacity. Here we do a single pass so you see the pattern.

    **What the plot shows:** Each line is one net's path on the placement grid. The dropdown lets you highlight a single net to see exactly which segments that connection uses.
    """)
    return


@app.cell
def _(PALETTE, plt):
    # Static diagram: cost update (step 3) — one edge before/after a net uses it
    _fig, _ax = plt.subplots(figsize=(4, 1.2))
    _ax.set_xlim(0, 3)
    _ax.set_ylim(0, 1)
    _ax.axis("off")
    _ax.plot([0.3, 2.7], [0.5, 0.5], color=PALETTE["ink"], lw=3, solid_capstyle="round")
    _ax.text(0.3, 0.2, "Before: 1.0", fontsize=9, ha="left")
    _ax.text(1.5, 0.85, "Net uses edge", fontsize=9, ha="center", style="italic")
    _ax.text(2.7, 0.2, "After: 2.5", fontsize=9, ha="right")
    _fig.suptitle("Step 3: after routing a net, add 1.5 to every edge it used", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Coordinate, Route, board_size, netlist, nx, pl_history):
    # Routing Solver (N from board_size)
    def route_design(graph, pl, N=10):
        _grid = nx.grid_2d_graph(N, N)
        for _u_g, _v_g in _grid.edges(): _grid.edges[_u_g, _v_g]['weight'] = 1.0
        _routes = []
        for _u, _v in graph.edges():
            if _u not in pl or _v not in pl: continue
            _path = nx.astar_path(_grid, pl[_u], pl[_v], weight='weight')
            for _i in range(len(_path)-1):
                _e = tuple(sorted((_path[_i], _path[_i+1])))
                _grid.edges[_e[0], _e[1]]['weight'] += 1.5 
            _routes.append(Route(u=_u, v=_v, path=[Coordinate(x=p[0], y=p[1]) for p in _path]))
        return _routes

    _n = int(board_size.value)
    final_pl = pl_history[-1][0]
    final_routes = route_design(netlist, final_pl, N=_n)
    return final_pl, final_routes


@app.cell
def _(final_routes, mo):
    _net_options = [f"{r.u} → {r.v}" for r in final_routes]
    route_selector = mo.ui.dropdown(
        options=_net_options,
        value=_net_options[0] if _net_options else None,
        label="Inspect one net's path (A* result)",
    )
    _n_routes = len(final_routes)
    first_k_slider = mo.ui.slider(0, _n_routes, value=0, label="Show first K nets (0 = use dropdown)")
    _out = mo.vstack([
        mo.md("#### See one net's path"),
        mo.md("Pick a net to see the exact sequence of grid segments chosen by A* (step 2). Or use the slider to show the first K nets (K-th in bold)."),
        route_selector,
        first_k_slider,
    ])
    _out
    return first_k_slider, route_selector


@app.cell
def _(
    PALETTE,
    board_size,
    final_pl,
    final_routes,
    first_k_slider,
    plt,
    route_selector,
):
    # Two-panel: left = path view, right = congestion heatmap (edge usage)
    _n = int(board_size.value)
    _fig, (_ax_path, _ax_cong) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Right panel: congestion heatmap (edge usage count)
    _edge_usage = {}
    for _r in final_routes:
        _pts = [_p.to_tuple() for _p in _r.path]
        for _i in range(len(_pts) - 1):
            _e = tuple(sorted((_pts[_i], _pts[_i + 1])))
            _edge_usage[_e] = _edge_usage.get(_e, 0) + 1
    _max_use = max(_edge_usage.values()) if _edge_usage else 1
    _usage_colors = {1: PALETTE["primary"], 2: PALETTE["bram"]}
    _ax_cong.set_xticks(range(_n + 1))
    _ax_cong.set_yticks(range(_n + 1))
    _ax_cong.set_aspect("equal")
    _ax_cong.set_xlim(-0.5, _n + 0.5)
    _ax_cong.set_ylim(-0.5, _n + 0.5)
    _ax_cong.grid(True, alpha=0.25)
    for _node, (_x, _y) in final_pl.items():
        _ax_cong.add_patch(plt.Rectangle((_x + 0.15, _y + 0.15), 0.7, 0.7, color=PALETTE["neutral_fill"], alpha=0.2, ec=PALETTE["neutral_line"], lw=0.5))
    for (_a, _b), _cnt in _edge_usage.items():
        _x1, _y1 = _a[0] + 0.5, _a[1] + 0.5
        _x2, _y2 = _b[0] + 0.5, _b[1] + 0.5
        _col = _usage_colors.get(_cnt, PALETTE["congestion_high"]) if _cnt <= 2 else PALETTE["congestion_high"]
        _ax_cong.plot([_x1, _x2], [_y1, _y2], color=_col, lw=2 + _cnt * 0.5, zorder=1)
    _ax_cong.set_title("Congestion: edge color = number of nets using that segment")

    # --- Left panel: path view (selected net or first K nets)
    _ax_path.set_xticks(range(_n + 1))
    _ax_path.set_yticks(range(_n + 1))
    _ax_path.grid(True, alpha=0.25)
    _ax_path.set_aspect("equal")
    _ax_path.set_xlim(-0.5, _n + 0.5)
    _ax_path.set_ylim(-0.5, _n + 0.5)
    for _node, (_x, _y) in final_pl.items():
        _ax_path.add_patch(plt.Rectangle((_x + 0.15, _y + 0.15), 0.7, 0.7, color=PALETTE["neutral_fill"], alpha=0.2, ec=PALETTE["neutral_line"], lw=0.5))

    _k = first_k_slider.value
    if _k > 0:
        # Show first K nets; K-th (index K-1) bold
        _routes_to_show = final_routes[: _k]
        _bold_idx = _k - 1
        for _idx, _r in enumerate(_routes_to_show):
            _pts = [_p.to_tuple() for _p in _r.path]
            _xs = [p[0] + 0.5 for p in _pts]
            _ys = [p[1] + 0.5 for p in _pts]
            _bold = _idx == _bold_idx
            _ax_path.plot(_xs, _ys, color=PALETTE["primary"], lw=4 if _bold else 1, alpha=0.9 if _bold else 0.35, zorder=3)
        if _routes_to_show:
            _ax_path.set_title(f"First {_k} nets (net {_k} bold); order shows routing sequence")
    else:
        # Dropdown: one net with step numbers
        _selected = route_selector.value
        _route = next((r for r in final_routes if f"{r.u} → {r.v}" == _selected), None)
        if _route is not None:
            _pts = [_p.to_tuple() for _p in _route.path]
            _xs = [p[0] + 0.5 for p in _pts]
            _ys = [p[1] + 0.5 for p in _pts]
            for _x, _y in _pts:
                _ax_path.add_patch(plt.Circle((_x + 0.5, _y + 0.5), 0.24, color=PALETTE["primary"], zorder=2, ec="white", lw=1))
            _ax_path.plot(_xs, _ys, color=PALETTE["primary"], lw=4, zorder=3, solid_capstyle="round")
            _ax_path.plot(_xs, _ys, color=PALETTE["primary_light"], lw=1.5, zorder=4, alpha=0.9)
            for _i, (_x, _y) in enumerate(_pts):
                _ax_path.text(_x + 0.5, _y + 0.5, str(_i), fontsize=9, fontweight="bold", ha="center", va="center", color="white", zorder=5)
            _ax_path.annotate(f"from {_route.u}", xy=(_xs[0], _ys[0]), xytext=(_xs[0] - 0.5, _ys[0] + 0.7), fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_line"], lw=1))
            _ax_path.annotate(f"to {_route.v}", xy=(_xs[-1], _ys[-1]), xytext=(_xs[-1] + 0.5, _ys[-1] + 0.7), fontsize=8, ha="center", arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_line"], lw=1))
            _ax_path.set_title(f"Net: {_route.u} → {_route.v}  ({len(_pts) - 1} segments; numbers = step order)")
        else:
            _ax_path.set_title("Select a net from the dropdown above")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Left: one net's path (dropdown) or first K nets (slider); right: congestion — edge color = number of nets using that segment (negotiated congestion).*
    *One net only: the path from source to target is the sequence of grid cells A* chose. Numbers 0, 1, 2, … are the step order along that path (each step = one edge). Change the dropdown to see a different net.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## ⏱️ Stage 4: Static Timing Analysis (STA)

    **Why we care about timing:** The clock tells every register when to capture new values. For that to work, signals must *arrive* at each register *before* the next clock edge (setup constraint). So we need to know: what is the latest time any signal arrives? That latest arrival defines the **minimum clock period** we can use. STA answers that without simulating every cycle.

    **What STA does (in steps):**

    1. **Combinational only.** We ignore edges that cross a register (e.g. REG_B → REG_A goes through a clock); those start a new "timing path" on the next cycle. What remains is a DAG of combinational logic and wires between registers and I/O.

    2. **Propagate arrival times.** Start from primary inputs and register outputs (time 0). For each block, set its arrival time = max(predecessors' arrival) + delay of that block/wire. We use a simple delay model: each block adds a fixed delay (e.g. LUT vs BRAM), each wire adds a small delay. This sweeps forward until every node has an arrival time.

    3. **Critical path and Fmax.** The **critical path** is the path that achieves the **maximum arrival time** anywhere in the design. That value (in ns) is the shortest clock period that can safely capture all signals—so **maximum frequency** $F_{max} = 1 / \text{(critical path delay)}$.

    **Slack:** For each block, *slack* = (required time) − (arrival time). Here we take "required time" as the same for everyone (the max arrival). So the block that *has* the max arrival has slack ≈ 0; that block is on the critical path. Other blocks have positive slack (they "finish early"). Zero or near-zero slack = critical; positive slack = timing margin.

    **Combinational path:** A path with *no registers in between*—signals go from one register (or input) through logic and wires to the next register (or output). Everything on that path must settle within one clock period; the longest such path sets $F_{max}$.
    """)
    return


@app.cell
def _(PALETTE, plt):
    # Small diagram: one timing path with arrival times (conceptual)
    _fig, _ax = plt.subplots(figsize=(5, 1.8))
    _ax.set_xlim(0, 5)
    _ax.set_ylim(0, 1.5)
    _ax.axis("off")
    _nodes = ["Input/Reg", "Block A", "Block B", "Reg"]
    _xs = [0.5, 1.5, 2.8, 4.2]
    for _i, (_x, _l) in enumerate(zip(_xs, _nodes)):
        _ax.plot(_x, 0.8, "o", color=PALETTE["primary"], markersize=10)
        _ax.text(_x, 0.5, _l, fontsize=8, ha="center")
        _ax.text(_x, 1.15, f"t={0.5 + _i * 0.6}", fontsize=7, ha="center", style="italic")
    for _i in range(len(_xs) - 1):
        _ax.annotate("", xy=(_xs[_i + 1], 0.8), xytext=(_xs[_i], 0.8), arrowprops=dict(arrowstyle="->", color=PALETTE["neutral_line"], lw=1.5))
    _ax.text(2.35, 0.2, "arrival time increases along path → max sets clock period", fontsize=8, ha="center")
    _fig.suptitle("STA idea: propagate arrival times along combinational paths; the max is the critical path delay", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(TimingResult, final_routes, netlist, nx):
    # STA Solver
    def run_sta(graph, rts):
        _dag = nx.DiGraph()
        _dag.add_nodes_from(graph.nodes(data=True))
        for _u, _v, _data in graph.edges(data=True):
            if not _data.get('is_sequential', False): _dag.add_edge(_u, _v)

        _at = {n: 0.0 for n in _dag.nodes}
        _order = list(nx.topological_sort(_dag))
        for _u in _order:
            _at[_u] += 4.5 if _dag.nodes[_u].get('block_type') == "BRAM" else 1.0
            for _v in _dag.successors(_u):
                _at[_v] = max(_at[_v], _at[_u] + 0.6)

        _max_at = max(_at.values()) if _at else 0
        _res = [TimingResult(
            block_id=_n, block_type=graph.nodes[_n]['block_type'],
            arrival=_at[_n], required=_max_at, slack=_max_at - _at[_n],
            is_critical=(_max_at - _at[_n]) < 0.1
        ) for _n in graph.nodes]
        return _res, _max_at

    timing_results, fmax_delay = run_sta(netlist, final_routes)
    return fmax_delay, timing_results


@app.cell
def _(fmax_delay, mo):
    _delay_ns = max(fmax_delay, 0.01)
    mo.stat(
        label="Maximum Clock Frequency",
        value=f"{1000/_delay_ns:.1f} MHz",
        caption=f"Critical path delay: {fmax_delay:.2f} ns → $F_{{max}}$ = 1 / delay. The longest combinational path in the design (STA propagation) sets this limit.",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    *Below: the netlist with the **critical path** in gold (blocks and edges that have the smallest slack); then a bar chart of **slack per block** — the block(s) with slack ≈ 0 are on the critical path and limit $F_{max}$.*
    """)
    return


@app.cell
def _(PALETTE, current_layout_fn, netlist, nx, plt, timing_results):
    # Critical path highlighted on netlist — same layout as Stage 1
    _critical_ids = {r.block_id for r in timing_results if r.is_critical}
    _critical_edges = [(u, v) for u, v in netlist.edges() if u in _critical_ids and v in _critical_ids]
    _fig, _ax = plt.subplots(figsize=(12, 6))
    _pos = current_layout_fn(netlist)
    _colors = []
    for n in netlist.nodes:
        if n in _critical_ids:
            _colors.append(PALETTE["critical"])
        elif netlist.nodes[n].get("block_type") == "IO":
            _colors.append(PALETTE["io"])
        elif netlist.nodes[n].get("block_type") == "LUT":
            _colors.append(PALETTE["lut"])
        else:
            _colors.append(PALETTE["bram"])
    nx.draw(
        netlist, _pos,
        with_labels=True,
        node_color=_colors,
        node_size=700,
        font_size=6,
        ax=_ax,
        edge_color=PALETTE["neutral_line"],
        width=1.0,
        alpha=0.5,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
    )
    if _critical_edges:
        nx.draw_networkx_edges(
            netlist, _pos, edgelist=_critical_edges, ax=_ax,
            edge_color=PALETTE["critical"], width=3.0, alpha=0.9,
            arrows=True,
            arrowsize=14,
            connectionstyle="arc3,rad=0.1",
        )
    _ax.set_title("Netlist with critical path highlighted (gold) — dataflow layout")
    _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(PALETTE, plt, timing_results):
    # Slack summary: bar chart of slack per block (smallest slack = most critical)
    _sorted_res = sorted(timing_results, key=lambda r: r.slack)
    _n_show = min(15, len(_sorted_res))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    if _n_show > 0:
        _subset = _sorted_res[:_n_show]
        _ax.barh(
            range(_n_show),
            [r.slack for r in _subset],
            color=[PALETTE["critical"] if r.is_critical else PALETTE["primary_soft"] for r in _subset],
            alpha=0.9,
        )
        _ax.set_yticks(range(_n_show))
        _ax.set_yticklabels([r.block_id for r in _subset], fontsize=7)
    _ax.set_xlabel("Slack")
    _ax.set_title("Slack by block (lowest = most critical; gold = on critical path)")
    _ax.axvline(x=0, color=PALETTE["neutral_line"], linestyle="--", alpha=0.5)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 🚀 Stage 5: Hardware Simulation (Waveforms)

    **What we're simulating:** Depends on the selected design (use the **Design** dropdown in Stage 1). For **Fibonacci**: REG_A ← REG_B, REG_B ← REG_A + REG_B. For **Pipelined Adder**: REG_A ← REG_C, REG_B ← REG_A+1, REG_C ← REG_B+1 (2-stage increment pipeline).

    **Wires:** The table and waveform show the signals for the selected design. Choose which wires to plot from the multiselect.
    """)
    return


@app.cell
def _(design_key):
    # Precompute simulation: dispatch by design
    def _sim_fibonacci():
        _rows = []
        _reg_a, _reg_b = 0, 1
        for _c in range(25):
            _a_next = _reg_b
            _b_next = (_reg_a + _reg_b) % 256
            _rows.append({"Cycle": _c, "REG_A": _reg_a, "REG_B": _reg_b, "A_next": _a_next, "B_next": _b_next})
            _reg_a, _reg_b = _a_next, _b_next
        return _rows, ["REG_A", "REG_B", "A_next", "B_next"]

    def _sim_pipelined_adder():
        _rows = []
        _ra, _rb, _rc = 0, 0, 0
        for _c in range(25):
            _a_next = _rc
            _b_next = (_ra + 1) % 256
            _c_next = (_rb + 1) % 256
            _rows.append({"Cycle": _c, "REG_A": _ra, "REG_B": _rb, "REG_C": _rc, "A_next": _a_next, "B_next": _b_next, "C_next": _c_next})
            _ra, _rb, _rc = _a_next, _b_next, _c_next
        return _rows, ["REG_A", "REG_B", "REG_C", "A_next", "B_next", "C_next"]

    _simulators = {"fibonacci": _sim_fibonacci, "pipelined_adder": _sim_pipelined_adder}
    _rows, _cols = _simulators.get(design_key, _sim_fibonacci)()
    sim_table = _rows
    wire_cols = _cols
    return sim_table, wire_cols


@app.cell
def _(mo, sim_table, wire_cols):
    _n = max(0, len(sim_table) - 1)
    cycle_slider = mo.ui.slider(0, _n, value=0, label="Cycle")
    wire_selector = mo.ui.multiselect(
        options=wire_cols,
        value=wire_cols[:2] if len(wire_cols) >= 2 else wire_cols,
        label="Wires to show on waveform",
    )
    _out = mo.vstack([
        mo.md("#### Select cycle and wires"),
        cycle_slider,
        wire_selector,
    ])
    _out
    return cycle_slider, wire_selector


@app.cell
def _(
    PALETTE,
    TRACE_COLORS,
    WaveformWidget,
    cycle_slider,
    mo,
    pl,
    sim_table,
    wire_cols,
    wire_selector,
):
    # Waveform: selected wires from sim_table; colors from palette
    _selected = wire_selector.value or (wire_cols[:2] if wire_cols else [])
    _wire_names = [w for w in _selected if w in sim_table[0]] if sim_table else []
    _wire_values = [[r[w] for r in sim_table] for w in _wire_names]
    waveform = WaveformWidget(
        wire_names=_wire_names,
        wire_values=_wire_values,
        trace_colors=TRACE_COLORS,
        surface_dark=PALETTE["surface_dark"],
        primary_soft=PALETTE["primary_soft"],
    )

    _c = max(0, min(int(cycle_slider.value), len(sim_table) - 1)) if sim_table else 0
    _row = sim_table[_c] if _c < len(sim_table) else {}

    cycle_df = pl.DataFrame(sim_table)

    _stat_wires = [w for w in wire_cols if w in _row]
    _out = mo.vstack([
        mo.hstack([
            mo.stat(label="Cycle", value=str(_c)),
            *[mo.stat(label=f"{w} (at selected cycle)", value=str(_row.get(w, ""))) for w in _stat_wires],
        ], justify="space-around"),
        mo.md("#### Logic Analyzer (select wires above to map arbitrary signals)"),
        waveform,
        mo.md("#### Cycle table (all wires)"),
        cycle_df,
        mo.md("*Table shows all wires per cycle. Use the multiselect to choose which wires appear on the waveform.*"),
    ])
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Glossary

    | Term | Definition |
    |------|------------|
    | **HDL** | Hardware Description Language (e.g. Verilog, VHDL); the “source code” that describes the design. Synthesis compiles HDL into a netlist. |
    | **Netlist** | Graph of components (nodes) and connections (edges) that the flow places and routes on silicon. |
    | **IO** | Input/Output; pins or blocks that connect the design to the outside (red in figures; e.g. RST, START). |
    | **LUT** | Look-Up Table; programmable logic block (green in figures). |
    | **BRAM** | Block RAM; on-chip memory (orange in figures). |
    | **Placement** | Assigning each block to a physical $(x, y)$ slot on the FPGA grid. |
    | **HPWL** | Half-Perimeter Wire Length. For each net, the Manhattan distance between its two endpoints; sum over all nets is the wirelength cost we minimize during placement. |
    | **Routing** | Assigning physical wire paths to each net on the grid. |
    | **Pathfinder** | Negotiated congestion routing: find paths (e.g. A*), increase cost on used edges, repeat until nets spread onto legal routes. |
    | **Congestion** | Overuse of wire segments when multiple nets share the same resource. |
    | **STA** | Static Timing Analysis; computes arrival times and critical path. |
    | **Critical path** | Longest combinational path; sets the minimum clock period. |
    | **Slack** | Required time minus arrival time; zero slack = critical. |
    | **Fmax** | Maximum clock frequency = 1 / critical path delay. |
    | **Cycle-accurate simulation** | Simulation that advances state once per clock cycle. |
    """)
    return


if __name__ == "__main__":
    app.run()
