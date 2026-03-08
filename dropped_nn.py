# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "altair==6.0.0",
#     "numpy==2.4.2",
#     "pandas==3.0.1",
#     "torch==2.10.0",
#     "scipy==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import json
    import html
    import marimo as mo
    import numpy as np
    import pandas as pd
    import re
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import urllib.request
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr

    return (
        F,
        Path,
        alt,
        html,
        json,
        math,
        mo,
        nn,
        pd,
        pearsonr,
        random,
        re,
        torch,
        urllib,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Dropped a Neural Net — Jane Street Puzzle

    Jane Street posted a puzzle called [Dropped a Neural Net](https://huggingface.co/spaces/jane-street/droppedaneuralnet)
    on HuggingFace Spaces. The challenge: a neural network's weights were "dropped" (scattered),
    and you have to reassemble them in the correct order.

    This notebook reverse-engineers the space to understand:
    1. What the original model architecture looks like
    2. How the weights are distributed across piece files
    3. How to reconstruct the model

    ---
    ## Setup
    """)
    return


@app.cell
def _(Path):
    space_id = "jane-street/droppedaneuralnet"
    space_url = f"https://huggingface.co/spaces/{space_id}"
    api_url = f"https://huggingface.co/api/spaces/{space_id}"
    tree_url = f"{api_url}/tree/main"
    readme_url = f"{space_url}/raw/main/README.md"
    app_url = f"{space_url}/raw/main/app.py"
    data_dir = Path("data") / "droppedaneuralnet"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Space ID: {space_id}")
    print(f"Space URL: {space_url}")
    return api_url, app_url, data_dir, readme_url, tree_url


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Fetching from HuggingFace

    Fetch space metadata, the file tree, README, and app source via the HuggingFace API and raw file URLs.
    """)
    return


@app.cell
def _(api_url, app_url, html, json, re, readme_url, tree_url, urllib):
    def fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    def fetch_json(url: str) -> dict | list:
        return json.loads(fetch_text(url))
    def strip_frontmatter(readme_text: str) -> str:
        frontmatter_match = re.match(r"^---\s*\n.*?\n---\s*\n?", readme_text, flags=re.DOTALL)
        if frontmatter_match:
            return readme_text[frontmatter_match.end():]
        return readme_text
    def extract_code_blocks(text: str) -> list[str]:
        return re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    def dehtml(s: str) -> str:
        return html.unescape(re.sub(r"<[^>]+>", "", s))
    space_meta = fetch_json(api_url)
    file_tree = fetch_json(tree_url)
    readme_raw = fetch_text(readme_url)
    readme_body = strip_frontmatter(readme_raw)
    app_source = fetch_text(app_url)
    return file_tree, readme_body


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Space README
    """)
    return


@app.cell
def _(mo, readme_body):
    print(readme_body)
    mo.md(readme_body)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## File Tree
    """)
    return


@app.cell
def _(file_tree, pd):
    file_df = pd.DataFrame(file_tree)
    file_df["size_kb"] = file_df["size"] / 1024
    file_df = file_df.sort_values("size", ascending=False)
    print(f"Files in repository ({len(file_df)} total):")
    print(file_df[["path", "type", "size_kb"]].to_string())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Data Reading

    The space provides two data assets: `historical_data.csv` (tabular features/labels)
    and a `pieces/` directory of `.pth` files containing individual weight tensors.
    """)
    return


@app.cell
def _(data_dir, pd):
    csv_path = data_dir / "historical_data.csv"
    historical_data_df = pd.read_csv(csv_path)
    print(f"Historical data shape: {historical_data_df.shape}")
    print(f"\nColumns: {historical_data_df.columns.tolist()}")
    return (historical_data_df,)


@app.cell
def _(data_dir):
    pieces_dir = data_dir / "pieces"
    piece_files = sorted(pieces_dir.glob("piece_*.pth"))
    print(f"Found {len(piece_files)} piece files locally")
    total_size = sum(f.stat().st_size for f in piece_files)
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB\n")
    print("Piece files (first 15):")
    for f in piece_files[:15]:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<20} {size_kb:>10.2f} KB")
    if len(piece_files) > 15:
        print(f"  ... and {len(piece_files) - 15} more files")
    return (piece_files,)


@app.cell
def _(pd, piece_files, torch):
    print("Analyzing PyTorch piece files...\n")
    shape_counts = {}
    for fpath in piece_files:
        try:
            state_dict = torch.load(fpath, weights_only=False, map_location="cpu")
            if isinstance(state_dict, dict):
                for key, val in state_dict.items():
                    if isinstance(val, torch.Tensor):
                        shape = tuple(val.shape)
                        if shape not in shape_counts:
                            shape_counts[shape] = 0
                        shape_counts[shape] += 1
        except Exception as e:
            print(f"{fpath.name}: Error - {e}")
    print("Tensor shape distribution:")
    shape_df = pd.DataFrame([
        {"shape": str(shape), "count": count}
        for shape, count in sorted(shape_counts.items(), key=str)
    ])
    print(shape_df.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Learned Model Description

    PyTorch stores weight matrices as `(out_features, in_features)`, so we have:
    - 1 piece with shape `(1, 48)`: the **final layer** (`LastLayer.layer`, 48 → 1)
    - 48 pieces with shape `(96, 48)`: **expander** layers (`Block.inp`, 48 → 96)
    - 48 pieces with shape `(48, 96)`: **compressor** layers (`Block.out`, 96 → 48)

    Each block is a residual unit: `x_out = x + out(relu(inp(x)))`
    """)
    return


@app.cell
def _(piece_files, torch):
    pieces = {i: torch.load(pf, weights_only=False, map_location="cpu") for i, pf in enumerate(piece_files)}
    inp_pieces = {i: sd for i, sd in pieces.items() if sd["weight"].shape == (96, 48)}
    out_pieces = {i: sd for i, sd in pieces.items() if sd["weight"].shape == (48, 96)}
    last_pieces = {i: sd for i, sd in pieces.items() if sd["weight"].shape == (1, 48)}
    print(f"inp (expander, 96×48): {len(inp_pieces)}, out (compressor, 48×96): {len(out_pieces)}, last (1×48): {len(last_pieces)}")
    return inp_pieces, last_pieces, out_pieces


@app.cell
def _(F, historical_data_df, last_pieces, pearsonr, torch):
    X = torch.tensor(
        historical_data_df[[f"measurement_{i}" for i in range(48)]].values,
        dtype=torch.float32,
    )
    pred_true = historical_data_df["pred"].values
    last_sd = next(iter(last_pieces.values()))
    baseline = F.linear(X, last_sd["weight"], last_sd["bias"]).squeeze(-1).detach().numpy()
    print(f"Baseline r (output layer alone on raw input): {pearsonr(baseline, pred_true)[0]:.4f}")
    return X, last_sd, pred_true


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Reconstruction Strategy

    **Three-phase approach:**
    1. **Hungarian pairing**: Score all 48×48 possible (inp, out) pairs on raw data,
       then use the Hungarian algorithm for optimal 1-1 matching. This decouples the
       pairing problem from the ordering problem — reducing search from 2304 to 48 choices/step.
    2. **Beam search ordering** (K=100): With 48 paired blocks, maintain K candidate orderings.
       At each step, each beam tries all remaining blocks (only 48 choices vs 2304).
    3. **SA + exhaustive local search**: Refine ordering with SA (200k iterations allowing
       both position swaps and out-piece re-pairings to fix any Hungarian errors), then
       sweep all possible swaps until no improvement remains.
    """)
    return


@app.cell
def _(F, torch):
    def block_forward(x: torch.Tensor, inp_sd: dict, out_sd: dict) -> torch.Tensor:
        h = F.relu(F.linear(x, inp_sd["weight"], inp_sd["bias"]))
        return x + F.linear(h, out_sd["weight"], out_sd["bias"])

    return (block_forward,)


@app.cell
def _(
    F,
    X,
    block_forward,
    inp_pieces,
    last_sd,
    out_pieces,
    pearsonr,
    pred_true,
):
    # Full greedy: try all remaining (inp, out) pairs at each step.
    # This finds pairings AND ordering simultaneously (r≈0.70).
    _remaining_inp = dict(inp_pieces)
    _remaining_out = dict(out_pieces)
    _cx = X.clone()
    greedy_order = []

    for _step in range(48):
        _best_r, _best_pair = -1.0, None
        for _i, _inp_sd in _remaining_inp.items():
            for _j, _out_sd in _remaining_out.items():
                _x_cand = block_forward(_cx, _inp_sd, _out_sd)
                _pred_cand = F.linear(_x_cand, last_sd["weight"], last_sd["bias"]).squeeze(-1).detach().numpy()
                _r = pearsonr(_pred_cand, pred_true)[0]
                if _r > _best_r:
                    _best_r, _best_pair = _r, (_i, _j)
        _i_star, _j_star = _best_pair
        _cx = block_forward(_cx, _remaining_inp[_i_star], _remaining_out[_j_star])
        greedy_order.append((_i_star, _j_star))
        _remaining_inp.pop(_i_star)
        _remaining_out.pop(_j_star)
        print(f"Step {_step+1:2d}: inp={_i_star:3d}, out={_j_star:3d}, r={_best_r:.4f}")
    return


@app.cell
def _(F, X, block_forward, inp_pieces, last_sd, out_pieces, pred_true, torch):
    import time as _time

    # Config
    _K_joint = 30   # Joint beam width
    _K_pair = 200   # CD: beam width for pairing search
    _K_ord = 50     # CD: beam width for ordering search
    _N_b = min(1000, len(X))
    _Xs = X[:_N_b]
    _ys = pred_true[:_N_b]
    _lw = last_sd["weight"]
    _lb = last_sd["bias"]
    _yt = torch.tensor(_ys, dtype=torch.float32)
    _ytc = _yt - _yt.mean()
    _ytn = _ytc.norm()

    # Pre-stack weights
    _all_out_keys = sorted(out_pieces.keys())
    _all_out_W = torch.stack([out_pieces[k]["weight"] for k in _all_out_keys])
    _all_out_b = torch.stack([out_pieces[k]["bias"] for k in _all_out_keys])
    _out_k2i = {k: idx for idx, k in enumerate(_all_out_keys)}

    _all_inp_keys = sorted(inp_pieces.keys())
    _all_inp_W = torch.stack([inp_pieces[k]["weight"] for k in _all_inp_keys])
    _all_inp_b = torch.stack([inp_pieces[k]["bias"] for k in _all_inp_keys])
    _inp_k2i = {k: idx for idx, k in enumerate(_all_inp_keys)}

    def _pr(pred: torch.Tensor) -> torch.Tensor:
        pc = pred - pred.mean(-1, keepdim=True)
        return (pc * _ytc).sum(-1) / (pc.norm(dim=-1) * _ytn + 1e-10)

    def _eval(order: list) -> float:
        with torch.no_grad():
            cx = _Xs.clone()
            for i, j in order:
                cx = block_forward(cx, inp_pieces[i], out_pieces[j])
            return float(_pr(F.linear(cx, _lw, _lb).squeeze(-1).unsqueeze(0))[0])

    # ==================== Phase 1: Joint Beam Search ====================
    _t0 = _time.time()
    _beams = [(0.0, _Xs.clone(), frozenset(inp_pieces.keys()), frozenset(out_pieces.keys()), [])]

    with torch.no_grad():
        for _step in range(48):
            _cands = []
            for _bidx, (_sc, _cx, _ri, _ro, _ch) in enumerate(_beams):
                _ri_l = sorted(_ri)
                _ro_l = sorted(_ro)
                if not _ro_l:
                    continue
                _ro_idx = [_out_k2i[k] for k in _ro_l]
                _oW = _all_out_W[_ro_idx]
                _ob = _all_out_b[_ro_idx]
                _n_o = len(_ro_l)
                for _ik in _ri_l:
                    _h = F.relu(F.linear(_cx, inp_pieces[_ik]["weight"], inp_pieces[_ik]["bias"]))
                    _d = torch.einsum('nr,mor->mno', _h, _oW) + _ob[:, None, :]
                    _xn = _cx.unsqueeze(0) + _d
                    _pa = F.linear(_xn.reshape(-1, 48), _lw, _lb).view(_n_o, _N_b)
                    _rs = _pr(_pa)
                    for _oi in range(_n_o):
                        _cands.append((_rs[_oi].item(), _bidx, _ik, _ro_l[_oi]))
            _cands.sort(key=lambda c: -c[0])
            _new = []
            for _r, _bi, _ik, _ok in _cands[:_K_joint]:
                _old = _beams[_bi]
                _cx2 = block_forward(_old[1], inp_pieces[_ik], out_pieces[_ok])
                _new.append((_r, _cx2, _old[2] - {_ik}, _old[3] - {_ok}, _old[4] + [(_ik, _ok)]))
            _beams = _new
            if (_step + 1) % 12 == 0 or _step == 47:
                print(f"Joint {_step+1:2d}: r={_beams[0][0]:.4f} ({_time.time()-_t0:.0f}s)")

    _best_order = _beams[0][4]
    _best_r = _eval(_best_order)
    print(f"Phase 1: r={_best_r:.6f}")

    # ==================== Phase 2: Coordinate Descent ====================

    def _beam_out(inp_seq: list, K: int) -> tuple:
        """Fix inp sequence, beam search out assignment."""
        beams = [(0.0, _Xs.clone(), frozenset(out_pieces.keys()), [])]
        with torch.no_grad():
            for step in range(48):
                cands = []
                isd = inp_pieces[inp_seq[step]]
                for bidx, (sc, cx, rem, ch) in enumerate(beams):
                    ro_l = sorted(rem)
                    ro_idx = [_out_k2i[k] for k in ro_l]
                    oW = _all_out_W[ro_idx]
                    ob = _all_out_b[ro_idx]
                    n_o = len(ro_l)
                    h = F.relu(F.linear(cx, isd["weight"], isd["bias"]))
                    d = torch.einsum('nr,mor->mno', h, oW) + ob[:, None, :]
                    xn = cx.unsqueeze(0) + d
                    pa = F.linear(xn.reshape(-1, 48), _lw, _lb).view(n_o, _N_b)
                    rs = _pr(pa)
                    for oi in range(n_o):
                        cands.append((rs[oi].item(), bidx, ro_l[oi]))
                cands.sort(key=lambda c: -c[0])
                new = []
                for r, bi, ok in cands[:K]:
                    old = beams[bi]
                    cx2 = block_forward(old[1], isd, out_pieces[ok])
                    new.append((r, cx2, old[2] - {ok}, old[3] + [ok]))
                beams = new
        return list(zip(inp_seq, beams[0][3])), beams[0][0]

    def _beam_inp(out_seq: list, K: int) -> tuple:
        """Fix out sequence, beam search inp assignment."""
        beams = [(0.0, _Xs.clone(), frozenset(inp_pieces.keys()), [])]
        with torch.no_grad():
            for step in range(48):
                cands = []
                osd = out_pieces[out_seq[step]]
                for bidx, (sc, cx, rem, ch) in enumerate(beams):
                    ri_l = sorted(rem)
                    ri_idx = [_inp_k2i[k] for k in ri_l]
                    iW = _all_inp_W[ri_idx]
                    ib = _all_inp_b[ri_idx]
                    n_i = len(ri_l)
                    h_all = F.relu(
                        torch.bmm(cx.unsqueeze(0).expand(n_i, -1, -1), iW.permute(0, 2, 1))
                        + ib[:, None, :]
                    )
                    delta = F.linear(h_all.reshape(-1, 96), osd["weight"], osd["bias"]).view(n_i, _N_b, 48)
                    xn = cx.unsqueeze(0) + delta
                    pa = F.linear(xn.reshape(-1, 48), _lw, _lb).view(n_i, _N_b)
                    rs = _pr(pa)
                    for ii in range(n_i):
                        cands.append((rs[ii].item(), bidx, ri_l[ii]))
                cands.sort(key=lambda c: -c[0])
                new = []
                for r, bi, ik in cands[:K]:
                    old = beams[bi]
                    cx2 = block_forward(old[1], inp_pieces[ik], osd)
                    new.append((r, cx2, old[2] - {ik}, old[3] + [ik]))
                beams = new
        return list(zip(beams[0][3], out_seq)), beams[0][0]

    def _beam_reorder(pairs: list, K: int) -> tuple:
        """Fix pairings, beam search ordering."""
        aiw = torch.stack([inp_pieces[i]["weight"] for i, j in pairs])
        aib = torch.stack([inp_pieces[i]["bias"] for i, j in pairs])
        aow = torch.stack([out_pieces[j]["weight"] for i, j in pairs])
        aob = torch.stack([out_pieces[j]["bias"] for i, j in pairs])
        beams = [(0.0, _Xs.clone(), frozenset(range(48)), [])]
        with torch.no_grad():
            for step in range(48):
                cands = []
                for bidx, (sc, cx, rem, ch) in enumerate(beams):
                    rem_l = sorted(rem)
                    n = len(rem_l)
                    iw, ib_ = aiw[rem_l], aib[rem_l]
                    ow, ob_ = aow[rem_l], aob[rem_l]
                    h = F.relu(torch.bmm(cx.unsqueeze(0).expand(n, -1, -1), iw.permute(0, 2, 1)) + ib_[:, None, :])
                    d = torch.bmm(h, ow.permute(0, 2, 1)) + ob_[:, None, :]
                    xn = cx.unsqueeze(0) + d
                    pa = F.linear(xn.reshape(-1, 48), _lw, _lb).view(n, _N_b)
                    rs = _pr(pa)
                    for ki, k in enumerate(rem_l):
                        cands.append((rs[ki].item(), bidx, k))
                cands.sort(key=lambda c: -c[0])
                new = []
                for r, bi, k in cands[:K]:
                    old = beams[bi]
                    i, j = pairs[k]
                    cx2 = block_forward(old[1], inp_pieces[i], out_pieces[j])
                    new.append((r, cx2, old[2] - {k}, old[3] + [k]))
                beams = new
        return [pairs[k] for k in beams[0][3]], beams[0][0]

    print("\n--- Coordinate Descent ---")
    for _cd_it in range(5):
        _changed = False
        _new, _nr = _beam_out([p[0] for p in _best_order], _K_pair)
        if _nr > _best_r + 1e-8:
            _best_order, _best_r = _new, _nr
            _changed = True
            print(f"CD {_cd_it}: opt-out → r={_best_r:.6f} ({_time.time()-_t0:.0f}s)")
        _new, _nr = _beam_reorder(_best_order, _K_ord)
        if _nr > _best_r + 1e-8:
            _best_order, _best_r = _new, _nr
            _changed = True
            print(f"CD {_cd_it}: opt-order → r={_best_r:.6f} ({_time.time()-_t0:.0f}s)")
        _new, _nr = _beam_inp([p[1] for p in _best_order], _K_pair)
        if _nr > _best_r + 1e-8:
            _best_order, _best_r = _new, _nr
            _changed = True
            print(f"CD {_cd_it}: opt-inp → r={_best_r:.6f} ({_time.time()-_t0:.0f}s)")
        _new, _nr = _beam_reorder(_best_order, _K_ord)
        if _nr > _best_r + 1e-8:
            _best_order, _best_r = _new, _nr
            _changed = True
            print(f"CD {_cd_it}: opt-order → r={_best_r:.6f} ({_time.time()-_t0:.0f}s)")
        if _best_r > 0.999999:
            print("Solved!")
            break
        if not _changed:
            print(f"CD {_cd_it}: converged at r={_best_r:.6f}")
            break

    beam_best_order = _best_order
    print(f"\nFinal beam+CD: r={_best_r:.6f}, time={_time.time()-_t0:.0f}s")
    return (beam_best_order,)


@app.cell
def _(
    F,
    X,
    beam_best_order,
    block_forward,
    inp_pieces,
    last_sd,
    math,
    out_pieces,
    pearsonr,
    pred_true,
    random,
    torch,
):
    _N_sa = min(1000, len(X))
    _X_sub = X[:_N_sa]
    _pred_sub = pred_true[:_N_sa]

    def _sa_states(ordering: list, x_data: torch.Tensor) -> list:
        states = [x_data]
        for i, j in ordering:
            states.append(block_forward(states[-1], inp_pieces[i], out_pieces[j]))
        return states

    def _sa_r(states: list) -> float:
        _p = F.linear(states[-1], last_sd["weight"], last_sd["bias"]).squeeze(-1).detach().numpy()
        return pearsonr(_p, _pred_sub)[0]

    def _recompute(ordering: list, states: list, pos: int) -> list:
        new = states[:pos + 1]
        for k in range(pos, len(ordering)):
            i, j = ordering[k]
            new.append(block_forward(new[-1], inp_pieces[i], out_pieces[j]))
        return new

    def _local_search_2swap(ordering: list) -> tuple:
        """Run 2-swap local search until convergence. Returns (best_ordering, best_r)."""
        _states = _sa_states(ordering, _X_sub)
        _best = list(ordering)
        _best_r = _sa_r(_states)
        _improved = True
        while _improved:
            _improved = False
            _best_delta = 0.0
            _best_move = None
            for _a in range(48):
                for _b in range(_a + 1, 48):
                    # Position swap
                    _cand = list(_best)
                    _cand[_a], _cand[_b] = _cand[_b], _cand[_a]
                    _cs = _recompute(_cand, _states, _a)
                    _cr = _sa_r(_cs)
                    if _cr - _best_r > _best_delta:
                        _best_delta = _cr - _best_r
                        _best_move = (_cand, _cs, _cr)
                    # Out-piece swap
                    _cand2 = list(_best)
                    _ia, _ja = _cand2[_a]
                    _ib, _jb = _cand2[_b]
                    _cand2[_a] = (_ia, _jb)
                    _cand2[_b] = (_ib, _ja)
                    _cs2 = _recompute(_cand2, _states, _a)
                    _cr2 = _sa_r(_cs2)
                    if _cr2 - _best_r > _best_delta:
                        _best_delta = _cr2 - _best_r
                        _best_move = (_cand2, _cs2, _cr2)
                    # Inp-piece swap
                    _cand3 = list(_best)
                    _ia, _ja = _cand3[_a]
                    _ib, _jb = _cand3[_b]
                    _cand3[_a] = (_ib, _ja)
                    _cand3[_b] = (_ia, _jb)
                    _cs3 = _recompute(_cand3, _states, _a)
                    _cr3 = _sa_r(_cs3)
                    if _cr3 - _best_r > _best_delta:
                        _best_delta = _cr3 - _best_r
                        _best_move = (_cand3, _cs3, _cr3)
            if _best_move is not None:
                _best, _states, _best_r = _best_move
                _improved = True
                print(f"    2swap improved: r={_best_r:.6f}")
        return _best, _best_r

    # --- Phase 1: SA from beam search result ---
    _cur = list(beam_best_order)
    _states = _sa_states(_cur, _X_sub)
    _cur_r = _sa_r(_states)
    _best = list(_cur)
    _best_r = _cur_r
    print(f"Starting from beam: r={_cur_r:.6f}")

    _T = 0.1
    _n_iter = 100000
    _no_improve = 0

    for _it in range(_n_iter):
        if _best_r > 0.999999:
            print(f"  SA early exit at iter {_it}: r={_best_r:.6f}")
            break
        _a, _b = sorted(random.sample(range(48), 2))
        _move_type = random.random()
        if _move_type < 0.5:
            # Position swap
            _cand = list(_cur)
            _cand[_a], _cand[_b] = _cand[_b], _cand[_a]
        elif _move_type < 0.8:
            # Out-piece swap
            _cand = list(_cur)
            _ia, _ja = _cand[_a]
            _ib, _jb = _cand[_b]
            _cand[_a] = (_ia, _jb)
            _cand[_b] = (_ib, _ja)
        else:
            # Inp-piece swap
            _cand = list(_cur)
            _ia, _ja = _cand[_a]
            _ib, _jb = _cand[_b]
            _cand[_a] = (_ib, _ja)
            _cand[_b] = (_ia, _jb)

        _cand_states = _recompute(_cand, _states, _a)
        _cand_r = _sa_r(_cand_states)

        _delta = _cand_r - _cur_r
        if _delta > 0 or random.random() < math.exp(_delta / max(_T, 1e-10)):
            _cur = _cand
            _states = _cand_states
            _cur_r = _cand_r
            _no_improve = 0
            if _cur_r > _best_r:
                _best_r = _cur_r
                _best = list(_cur)
        else:
            _no_improve += 1

        _T *= 0.99995
        if _no_improve > 5000:
            _T = max(_T, 0.02)
            _no_improve = 0

        if (_it + 1) % 20000 == 0:
            print(f"  SA iter {_it+1:6d}: cur_r={_cur_r:.6f}, best_r={_best_r:.6f}, T={_T:.6f}")

    print(f"SA done: best_r={_best_r:.6f}")

    # --- Phase 2: 2-swap local search ---
    print("Running 2-swap local search...")
    _best, _best_r = _local_search_2swap(_best)
    print(f"After local search: r={_best_r:.6f}")

    # --- Phase 3: Iterated Local Search (perturb + local search) ---
    if _best_r < 0.999999:
        print("Running Iterated Local Search...")
        for _restart in range(20):
            if _best_r > 0.999999:
                break
            _perturbed = list(_best)
            _n_perturb = random.randint(3, 6)
            for _ in range(_n_perturb):
                _a, _b = random.sample(range(48), 2)
                _mv = random.random()
                if _mv < 0.4:
                    _perturbed[_a], _perturbed[_b] = _perturbed[_b], _perturbed[_a]
                elif _mv < 0.7:
                    _ia, _ja = _perturbed[_a]
                    _ib, _jb = _perturbed[_b]
                    _perturbed[_a] = (_ia, _jb)
                    _perturbed[_b] = (_ib, _ja)
                else:
                    _ia, _ja = _perturbed[_a]
                    _ib, _jb = _perturbed[_b]
                    _perturbed[_a] = (_ib, _ja)
                    _perturbed[_b] = (_ia, _jb)
            _p_order, _p_r = _local_search_2swap(_perturbed)
            if _p_r > _best_r:
                _best = _p_order
                _best_r = _p_r
                print(f"  ILS restart {_restart+1}: improved to r={_best_r:.6f}")

    # --- Phase 4: Insertion local search ---
    if _best_r < 0.999999:
        print("Running insertion local search...")
        _ins_states = _sa_states(_best, _X_sub)
        _ins_r = _sa_r(_ins_states)
        _ins_improved = True
        _ins_sweep = 0
        while _ins_improved and _ins_sweep < 10:
            _ins_improved = False
            _ins_sweep += 1
            _ins_best_delta = 0.0
            _ins_best_move = None
            for _src in range(48):
                for _dst in range(48):
                    if _src == _dst:
                        continue
                    _cand = list(_best)
                    _block = _cand.pop(_src)
                    _cand.insert(_dst, _block)
                    _cs = _sa_states(_cand, _X_sub)
                    _cr = _sa_r(_cs)
                    if _cr - _ins_r > _ins_best_delta:
                        _ins_best_delta = _cr - _ins_r
                        _ins_best_move = (_cand, _cs, _cr)
            if _ins_best_move is not None:
                _best, _ins_states, _ins_r = _ins_best_move
                _ins_improved = True
                print(f"  Insertion sweep {_ins_sweep}: r={_ins_r:.6f}")
            else:
                print(f"  Insertion sweep {_ins_sweep}: converged")
        if _ins_sweep > 1:
            _best, _ins_r = _local_search_2swap(_best)
            print(f"  Final 2-swap: r={_ins_r:.6f}")

    ordered_pairs = _best

    # Verify on full data
    _full = _sa_states(ordered_pairs, X)
    _full_pred = F.linear(_full[-1], last_sd["weight"], last_sd["bias"]).squeeze(-1).detach().numpy()
    _full_r = pearsonr(_full_pred, pred_true)[0]
    print(f"\nFinal r on full data: {_full_r:.6f}")
    return (ordered_pairs,)


@app.cell
def _(mo, ordered_pairs):
    _pairs_str = "\n".join(f"  {k+1:2d}: inp_piece={i:3d}, out_piece={j:3d}" for k, (i, j) in enumerate(ordered_pairs))
    mo.md(f"**Reconstructed block order ({len(ordered_pairs)} blocks):**\n```\n{_pairs_str}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Model Validation

    Rebuild the full `DroppedNN` from the greedy ordering and compare its predictions
    against the true `pred` column. A Pearson r > 0.99 confirms the reconstruction is correct.
    """)
    return


@app.cell
def _(F, nn):
    class Block(nn.Module):
        def __init__(self, inp_sd: dict, out_sd: dict):
            super().__init__()
            self.inp = nn.Linear(48, 96)
            self.out = nn.Linear(96, 48)
            self.inp.weight.data = inp_sd["weight"]
            self.inp.bias.data = inp_sd["bias"]
            self.out.weight.data = out_sd["weight"]
            self.out.bias.data = out_sd["bias"]

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.out(F.relu(self.inp(x)))

    class DroppedNN(nn.Module):
        def __init__(self, blocks: list, last_sd: dict):
            super().__init__()
            self.blocks = nn.Sequential(*blocks)
            self.last = nn.Linear(48, 1)
            self.last.weight.data = last_sd["weight"]
            self.last.bias.data = last_sd["bias"]

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.last(self.blocks(x)).squeeze(-1)

    return Block, DroppedNN


@app.cell
def _(
    Block,
    DroppedNN,
    X,
    historical_data_df,
    inp_pieces,
    last_sd,
    ordered_pairs,
    out_pieces,
    pearsonr,
    torch,
):
    model = DroppedNN(
        [Block(inp_pieces[i], out_pieces[j]) for i, j in ordered_pairs],
        last_sd,
    )
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
    _r_pred = pearsonr(preds, historical_data_df["pred"].values)[0]
    _r_true = pearsonr(preds, historical_data_df["true"].values)[0]
    print(f"r vs pred: {_r_pred:.6f}")
    print(f"r vs true: {_r_true:.6f}")
    return (preds,)


@app.cell
def _(alt, historical_data_df, pd, preds):
    _scatter_df = pd.DataFrame({
        "model_output": preds,
        "pred": historical_data_df["pred"].values,
    })
    scatter_chart = (
        alt.Chart(_scatter_df.sample(min(2000, len(_scatter_df))))
        .mark_point(opacity=0.4, size=15)
        .encode(
            x=alt.X("pred:Q", title="True pred (target)"),
            y=alt.Y("model_output:Q", title="Model output"),
            tooltip=["pred:Q", "model_output:Q"],
        )
        .properties(title="Model output vs target pred — tight diagonal = correct reconstruction", width=500, height=400)
        .interactive()
    )
    scatter_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
