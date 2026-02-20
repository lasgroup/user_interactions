# sdpo_signal_analysis.py
#
# Computes and visualizes the SDPO per-token signal (log-ratio) for a set of
# prompt/feedback cases.
#
# Pipeline:
#   1. Load cases from a JSON file (see auxiliary/signal_analysis_cases.json).
#   2. For each case, generate a completion y under the base context x.
#   3. Score y under three contexts:
#        - P(y | x)                  — base
#        - P(y | x, o_unrelated)     — unrelated follow-up (should be ~0)
#        - P(y | x, o_followup)      — relevant follow-up  (should have structure)
#   4. Compute per-token signals: log P(y|x,o) - log P(y|x).
#   5. Save all signals + metadata to <out_dir>/sdpo_signals.json.
#   6. Produce heatmap visualizations saved to <out_dir>/.
#
# Usage:
#   python sdpo_signal_analysis.py \
#       --cases_json auxiliary/signal_analysis_cases.json \
#       --model Qwen/Qwen3-8B \
#       --out_dir ./signal_analysis_out
#
# All options can also be set via environment variables (see shell script).

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer



@dataclass
class Case:
    name: str
    raw_prompt: str
    unrelated_followup: str
    followup_message: str
    tag: str


def load_cases(path: str, n_cases: int, seed: int) -> Tuple[List[Case], dict]:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    base: List[Case] = [
        Case(
            name=c["id"],
            raw_prompt=c["raw_prompt"],
            unrelated_followup=c["unrelated_followup"],
            followup_message=c["feedback_followup"],
            tag=c.get("tag", "unknown"),
        )
        for c in spec["cases"]
    ]

    suffix_pool = spec.get("defaults", {}).get("variant_suffix_pool", [""])
    rng = random.Random(seed)

    cases: List[Case] = []
    while len(cases) < n_cases:
        c = rng.choice(base)
        suffix = rng.choice(suffix_pool)
        cases.append(Case(
            name=f"{c.name}_v{len(cases):02d}",
            raw_prompt=c.raw_prompt + suffix,
            unrelated_followup=c.unrelated_followup,
            followup_message=c.followup_message,
            tag=c.tag,
        ))
    return cases, spec


def build_context(tok, raw_prompt: str) -> str:
    msgs = [{"role": "user", "content": raw_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_context_with_hindsight(tok, raw_prompt: str, followup: str) -> str:
    block = (
        "\n\n[HINDSIGHT CONTEXT]\n"
        "The following is a user response to your previous, insufficient attempt. "
        "Improve your response to the user prompt.\n"
        f"Future User Message: {followup.strip()}"
    )
    msgs = [{"role": "user", "content": raw_prompt + block}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def get_per_token_logps(
    model,
    tokenizer,
    context_texts: List[str],
    completion_ids_list: List[List[int]],
    max_prompt_length: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        per_tok_logps : (B, C)  log p(y_t | context, y_<t)
        y_ids         : (B, C)  completion token ids
        y_mask        : (B, C)  1 for real tokens, 0 for padding
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    enc = tokenizer(
        text=context_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    ).to(device)

    y_tensors = [torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids_list]
    y_masks   = [torch.ones_like(t, dtype=torch.long) for t in y_tensors]

    y_ids  = torch.nn.utils.rnn.pad_sequence(y_tensors, batch_first=True, padding_value=pad_id)
    y_mask = torch.nn.utils.rnn.pad_sequence(y_masks,   batch_first=True, padding_value=0)

    input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
    attn_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

    logits   = model(input_ids=input_ids, attention_mask=attn_mask).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets  = input_ids[:, 1:]

    P = enc["input_ids"].shape[1]
    C = y_ids.shape[1]
    per_tok_logps = logprobs[:, P - 1: P - 1 + C, :].gather(
        -1, targets[:, P - 1: P - 1 + C].unsqueeze(-1)
    ).squeeze(-1)

    return per_tok_logps, y_ids, y_mask



def token_strings_from_ids(tokenizer, ids: List[int]) -> List[str]:
    """Map token ids to readable string pieces, aligned 1:1."""
    ids = [int(i) for i in ids]
    pieces = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    return [p if isinstance(p, str) else str(p) for p in pieces]


def decode_token_pieces(tokenizer, token_ids: List[int]) -> List[str]:
    """Decode each token id individually to its literal text."""
    return [
        tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for tid in token_ids
    ]



def build_matrix(series_list: List[np.ndarray], width: int) -> np.ndarray:
    """Truncate each series to `width`, NaN-pad the rest."""
    mat = np.full((len(series_list), width), np.nan, dtype=np.float32)
    for i, v in enumerate(series_list):
        v = np.asarray(v, dtype=np.float32)[:width]
        mat[i, :len(v)] = v
    return mat


def _make_cmap_and_norm(vmin: float, vmax: float):
    cmap = plt.get_cmap("RdBu").copy()  # negative=red, positive=blue
    cmap.set_bad(color="white")
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return cmap, norm


def _style_axes(ax, ncols: int, xtick_step: int = 25):
    ax.set_yticks([])
    xticks = np.arange(0, ncols, xtick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("black")
    ax.tick_params(axis="x", which="both", width=1.8, length=8, labelsize=14)
    ax.tick_params(axis="y", which="both", width=0, length=0)


def _cbar(fig, im, ax, vmin: float, vmax: float):
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_linewidth(2.2)
    cbar.ax.tick_params(width=1.8, length=8, labelsize=14)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels([f"{vmin:g}", "0", f"+{vmax:g}"])
    return cbar


def plot_heatmap(mat: np.ndarray, outpath: str, *, vmin=-15.0, vmax=3.0, dpi=220, figsize=(12.0, 2.7)):
    mat = np.clip(mat, vmin, vmax)
    cmap, norm = _make_cmap_and_norm(vmin, vmax)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    _style_axes(ax, ncols=mat.shape[1])
    _cbar(fig, im, ax, vmin, vmax)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {outpath}")


def plot_stacked(mat_top: np.ndarray, mat_bottom: np.ndarray, outpath: str,
                 *, vmin=-15.0, vmax=3.0, dpi=220, figsize=(12.0, 5.4)):
    mat_top    = np.clip(mat_top, vmin, vmax)
    mat_bottom = np.clip(mat_bottom, vmin, vmax)
    cmap, norm = _make_cmap_and_norm(vmin, vmax)

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1.0, 0.045], wspace=0.12, hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[:, 1])

    im = ax1.imshow(mat_top,    aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax2.imshow(mat_bottom,      aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    _style_axes(ax1, ncols=mat_top.shape[1])
    _style_axes(ax2, ncols=mat_bottom.shape[1])
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x", which="both", length=0)

    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(2.2)
    cbar.ax.tick_params(width=1.8, length=8, labelsize=14)
    cbar.set_ticks([vmin, 0.0, vmax])
    cbar.set_ticklabels([f"{vmin:g}", "0", f"+{vmax:g}"])

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {outpath}")


def plot_side_by_side(mat_left: np.ndarray, mat_right: np.ndarray, outpath: str,
                      *, vmin=-15.0, vmax=3.0, dpi=220, figsize=(12.0, 2.7)):
    mat_left  = np.clip(mat_left, vmin, vmax)
    mat_right = np.clip(mat_right, vmin, vmax)
    cmap, norm = _make_cmap_and_norm(vmin, vmax)

    fig, (axL, axR) = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
    axL.imshow(mat_left,  aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    imR = axR.imshow(mat_right, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    _style_axes(axL, ncols=mat_left.shape[1])
    _style_axes(axR, ncols=mat_right.shape[1])
    _cbar(fig, imR, axR, vmin, vmax)

    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {outpath}")


def _normalize_piece_for_boxes(piece: str):
    """
    Return (leading_spaces, display_text) for token box rendering.
    Leading spaces are extracted so inter-box spacing is correct.
    Newlines are shown as ⏎, empty tokens as ∅.
    """
    if piece is None:
        return 0, "∅"
    piece = piece.replace("\r\n", "\n").replace("\n", "⏎")
    m = re.match(r"^[ \t]+", piece)
    lead = m.group(0) if m else ""
    lead_spaces = lead.count(" ") + 4 * lead.count("\t")
    core = piece[len(lead):]
    if core == "":
        core = "␠" if lead_spaces > 0 else "∅"
        lead_spaces = 0
    core = core.replace("\t", "⇥")
    return lead_spaces, core


def _text_color_for_bg(rgba):
    r, g, b, _a = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.62 else "white"


def plot_token_blocks(
    case: dict,
    tokenizer,
    outpath: str,
    *,
    case_index: int,
    n_tokens: int = 15,
    vmin: float = -15.0,
    vmax: float = 3.0,
    dpi: int = 220,
    fontsize: int = 14,
    fontfamily: str = "monospace",
):
    """
    Render the first `n_tokens` tokens as colored boxes for one case,
    showing both the unrelated and follow-up signal rows.
    """
    if "completion_token_ids" not in case:
        raise KeyError("case['completion_token_ids'] missing — regenerate sdpo_signals.json.")

    token_ids = case["completion_token_ids"]
    sig_unrel  = np.asarray(case["sig_unrel"],  dtype=np.float32)
    sig_follow = np.asarray(case["sig_follow"], dtype=np.float32)

    L = min(len(token_ids), len(sig_unrel), len(sig_follow))
    token_ids  = token_ids[:L]
    sig_unrel  = sig_unrel[:L]
    sig_follow = sig_follow[:L]

    pieces     = decode_token_pieces(tokenizer, token_ids)[:n_tokens]
    vals_unrel  = np.clip(sig_unrel[:n_tokens],  vmin, vmax)
    vals_follow = np.clip(sig_follow[:n_tokens], vmin, vmax)
    normed = [_normalize_piece_for_boxes(p) for p in pieces]

    cmap, norm = _make_cmap_and_norm(vmin, vmax)

    # Measure needed width via a scratch figure
    def _measure_width():
        tmp = plt.figure(figsize=(8.0, 2.0), dpi=dpi)
        ax  = tmp.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        tmp.canvas.draw()
        renderer = tmp.canvas.get_renderer()

        def mw(s):
            t = ax.text(0, 0, s, fontsize=fontsize, fontfamily=fontfamily, alpha=0.0)
            tmp.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            t.remove()
            return bb.width, bb.height

        sw, _ = mw(" ")
        margin, label_gap, pad_px, base_gap = 18, 12, 6, 1

        lbl1 = f"Case {case_index} — Unrelated — first {n_tokens} tokens:"
        lbl2 = f"Case {case_index} — Follow-up — first {n_tokens} tokens:"
        lw = max(mw(lbl1)[0], mw(lbl2)[0])

        x = margin + lw + label_gap
        max_h = 0.0
        for lead, txt in normed:
            tw, th = mw(txt)
            bw = tw + 2 * pad_px
            bh = th + 2 * (pad_px * 0.6)
            max_h = max(max_h, bh)
            x += lead * sw + bw + base_gap

        plt.close(tmp)
        return x + margin + 12, (max_h + 44) * 2 + 22, sw

    total_w, total_h, space_w = _measure_width()
    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
    gs  = fig.add_gridspec(nrows=2, ncols=1, hspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.set_axis_off()
    ax2.set_axis_off()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def mon_ax(ax, s):
        t = ax.text(0, 0, s, fontsize=fontsize, fontfamily=fontfamily, alpha=0.0, transform=ax.transAxes)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb.width, bb.height

    margin, label_gap, pad_px, base_gap = 18, 12, 6, 1
    lbl1 = f"Case {case_index} — Unrelated — first {n_tokens} tokens:"
    lbl2 = f"Case {case_index} — Follow-up — first {n_tokens} tokens:"
    lw = max(mon_ax(ax1, lbl1)[0], mon_ax(ax1, lbl2)[0])

    def draw_row(ax, label, values):
        bb = ax.get_window_extent(renderer=renderer)
        aw, ah = bb.width, bb.height

        def p2a(x, y):
            return x / aw, y / ah

        ax.text(*p2a(margin, ah - margin), label, transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize + 2, fontfamily=fontfamily, color="black")

        x_px = margin + lw + label_gap
        y_px = ah - margin - 28
        for (lead, txt), v in zip(normed, values):
            x_px += lead * space_w
            tw, th = mon_ax(ax, txt)
            bw = tw + 2 * pad_px
            bh = th + 2 * (pad_px * 0.6)
            rgba = list(cmap(norm(float(v))))
            rgba[3] = 0.95
            ax.text(*p2a(x_px, y_px), txt, transform=ax.transAxes,
                    ha="left", va="top", fontsize=fontsize, fontfamily=fontfamily,
                    color=_text_color_for_bg(rgba),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=rgba,
                              edgecolor=(0, 0, 0, 0.12), linewidth=1.0))
            x_px += bw + base_gap

    draw_row(ax1, lbl1, vals_unrel)
    draw_row(ax2, lbl2, vals_follow)
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {outpath}")



def parse_args():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_cases = os.path.join(repo_root, "auxiliary", "signal_analysis_cases.json")

    p = argparse.ArgumentParser(description="Compute and visualize SDPO per-token signal.")
    p.add_argument("--model", type=str,
                   default=os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    p.add_argument("--cases_json", type=str,
                   default=os.environ.get("CASES_JSON", default_cases))
    p.add_argument("--out_dir", type=str,
                   default=os.environ.get("OUT_DIR", "signal_analysis_out"))
    p.add_argument("--n_cases", type=int,
                   default=int(os.environ.get("N_CASES", "24")))
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("SEED", "123")))
    p.add_argument("--max_cols", type=int, default=120,
                   help="Heatmap width in tokens (longer completions are truncated)")
    p.add_argument("--vmin", type=float, default=-15.0)
    p.add_argument("--vmax", type=float, default=3.0)
    p.add_argument("--token_case_index", type=int, default=1,
                   help="Which case index to use for the token-block plot")
    p.add_argument("--token_n", type=int, default=15,
                   help="How many tokens to show in the token-block plot")
    return p.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    cases, spec = load_cases(args.cases_json, args.n_cases, seed=args.seed)
    defaults = spec.get("defaults", {})
    max_new_tokens = int(defaults.get("max_new_tokens", 256))
    temperature    = float(defaults.get("temperature", 1.0))
    do_sample      = bool(defaults.get("do_sample", True))

    min_y_tokens     = 80
    max_tries        = 6

    print(f"Model:      {args.model}")
    print(f"Cases JSON: {args.cases_json}  ({len(cases)} cases)")
    print(f"Out dir:    {args.out_dir}")
    print(f"max_new_tokens={max_new_tokens}  temperature={temperature}  do_sample={do_sample}")
    print()


    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()


    results = {
        "meta": {
            "model": args.model,
            "seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "min_y_tokens": min_y_tokens,
            "cases_json": args.cases_json,
        },
        "cases": [],
    }

    sig_unrel_list:  List[np.ndarray] = []
    sig_follow_list: List[np.ndarray] = []

    for i, c in enumerate(cases):
        print(f"===== {i + 1}/{len(cases)}: {c.name} ({c.tag}) =====")

        x_context = build_context(tok, c.raw_prompt)
        xo_unrel  = build_context_with_hindsight(tok, c.raw_prompt, c.unrelated_followup)
        xo_follow = build_context_with_hindsight(tok, c.raw_prompt, c.followup_message)

        # Generate y under x; resample until long enough
        model_inputs = tok(x_context, return_tensors="pt").to(model.device)
        prompt_len = model_inputs.input_ids.shape[1]

        completion_ids = None
        completion_text = None

        for attempt in range(1, max_tries + 1):
            gen_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            new_ids = gen_ids[0][prompt_len:]
            if new_ids.shape[0] >= min_y_tokens:
                # Keep exact generated ids — do not re-encode
                completion_ids  = new_ids.detach().cpu().to(torch.long).tolist()
                completion_text = tok.decode(new_ids, skip_special_tokens=True)
                break
            if attempt == max_tries:
                print(f"  [SKIP] completion too short after {max_tries} tries")

        if completion_ids is None:
            continue

        # Score under all three contexts
        logps_x,     y_ids, y_mask = get_per_token_logps(model, tok, [x_context],   [completion_ids])
        logps_unrel, _,     _      = get_per_token_logps(model, tok, [xo_unrel],    [completion_ids])
        logps_follow,_,     _      = get_per_token_logps(model, tok, [xo_follow],   [completion_ids])

        mask = y_mask[0].bool()
        base_lp   = logps_x[0][mask].detach().cpu()
        unrel_lp  = logps_unrel[0][mask].detach().cpu()
        follow_lp = logps_follow[0][mask].detach().cpu()

        sig_unrel  = (unrel_lp  - base_lp).to(torch.float32).numpy()
        sig_follow = (follow_lp - base_lp).to(torch.float32).numpy()

        # Align token ids/strings with the mask
        token_ids_masked = y_ids[0][mask].detach().cpu().to(torch.long).tolist()
        token_strs_masked = token_strings_from_ids(tok, token_ids_masked)

        if not (len(token_ids_masked) == len(sig_unrel) == len(sig_follow)):
            raise RuntimeError(
                f"Alignment mismatch: tokens={len(token_ids_masked)} "
                f"sig_unrel={len(sig_unrel)} sig_follow={len(sig_follow)}"
            )

        print(f"  tokens: {len(token_ids_masked)}"
              f"  |sig_unrel| mean: {np.abs(sig_unrel).mean():.4f}"
              f"  sig_follow mean: {sig_follow.mean():+.4f}")

        sig_unrel_list.append(sig_unrel)
        sig_follow_list.append(sig_follow)

        results["cases"].append({
            "name": c.name,
            "tag": c.tag,
            "raw_prompt": c.raw_prompt,
            "unrelated_followup": c.unrelated_followup,
            "followup_message": c.followup_message,
            "completion_text": completion_text,
            "n_tokens": len(token_ids_masked),
            "completion_token_ids": token_ids_masked,
            "completion_tokens": token_strs_masked,
            "sig_unrel":  sig_unrel.tolist(),
            "sig_follow": sig_follow.tolist(),
        })

    if not results["cases"]:
        print("No cases processed. Exiting.")
        return


    json_path = os.path.join(args.out_dir, "sdpo_signals.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] wrote: {json_path}")


    M_unrel  = build_matrix(sig_unrel_list,  width=args.max_cols)
    M_follow = build_matrix(sig_follow_list, width=args.max_cols)
    vmin, vmax = args.vmin, args.vmax

    plot_heatmap(M_unrel,  os.path.join(args.out_dir, "unrelated.png"),   vmin=vmin, vmax=vmax)
    plot_heatmap(M_follow, os.path.join(args.out_dir, "followup.png"),    vmin=vmin, vmax=vmax)
    plot_stacked(M_unrel, M_follow, os.path.join(args.out_dir, "stacked.png"),         vmin=vmin, vmax=vmax)
    plot_side_by_side(M_unrel, M_follow, os.path.join(args.out_dir, "side_by_side.png"), vmin=vmin, vmax=vmax)

    idx = args.token_case_index
    if 0 <= idx < len(results["cases"]):
        plot_token_blocks(
            results["cases"][idx],
            tok,
            os.path.join(args.out_dir, f"case{idx}_tokens.png"),
            case_index=idx,
            n_tokens=args.token_n,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        print(f"[WARN] token_case_index={idx} out of range for {len(results['cases'])} cases; skipping token plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
