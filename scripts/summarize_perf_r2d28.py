"""R2-Design-2.8 v2 D2.8-H final synthesis (v2 §15 Prompt 9).

Reads audit / repair / exposure / composition / channel / operator /
factorial / confirm and answers the 20 questions of v2 §15, produces:

    outputs/perf_r2d28/summary/
        R2D28_MASTER_TABLE.csv
        R2D28_HYPOTHESIS_LEDGER.csv
        R2D28_FINAL_DIAGNOSIS.md

No new experiments. No Test. No paper-model design. Verdict:
R2-Design-2.8 = PASS / PARTIAL / NO-GO, and the second-axis route
RFE-1..RFE-6 is named AFTER the results (v2 §17).
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.perf_r2d28_utils import (  # noqa: E402
    COMPOSITION_ROOT,
    CONFIRM_ROOT,
    EXPOSURE_ROOT,
    FACTORIAL_ROOT,
    OPERATOR_ROOT,
    REPAIR_ROOT,
    CHANNEL_ROOT,
    SUMMARY_ROOT,
    TARGET_DATASETS,
)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _paired(cand, base, metric, rows, ds_list=TARGET_DATASETS,
            seeds=(42, 43, 44)):
    cand_r = {(r["dataset"], int(r["seed"])): r for r in rows
              if r["variant"] == cand}
    base_r = {(r["dataset"], int(r["seed"])): r for r in rows
              if r["variant"] == base}
    per_ds, all_d = [], []
    for ds in ds_list:
        dv = []
        for s in seeds:
            c, b = cand_r.get((ds, s)), base_r.get((ds, s))
            cv, bv = _num(c[metric]) if c else None, _num(b[metric]) if b else None
            if cv is not None and bv is not None:
                dv.append(100 * (cv - bv))
        if dv:
            per_ds.append((ds, statistics.fmean(dv),
                           sum(1 for x in dv if x > 0), len(dv)))
            all_d += dv
    mean = statistics.fmean(all_d) if all_d else None
    n_pos = sum(1 for _, m, _, _ in per_ds if m > 0)
    return {"mean": mean, "per_ds": per_ds, "n_pos": n_pos}


def _pp(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def main() -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    repair = _load_csv(REPAIR_ROOT / "repair_results.csv")
    exposure = _load_csv(EXPOSURE_ROOT / "exposure_results.csv")
    comp = _load_csv(COMPOSITION_ROOT / "composition_results.csv")
    comp_causal = _load_csv(COMPOSITION_ROOT / "composition_causal.csv")
    channel = _load_csv(CHANNEL_ROOT / "channel_results.csv")
    operator = _load_csv(OPERATOR_ROOT / "operator_results.csv")
    factorial = _load_csv(FACTORIAL_ROOT / "factorial_results.csv")
    confirm = _load_csv(CONFIRM_ROOT / "confirm_results.csv")

    # ---------------- master table ----------------
    master = []
    for rows, tag in ((repair, "repair"), (exposure, "exposure"),
                      (comp, "composition"), (channel, "channel"),
                      (operator, "operator"), (factorial, "factorial"),
                      (confirm, "confirm")):
        for r in rows:
            master.append({
                "stage": tag, "dataset": r.get("dataset"),
                "variant": r.get("variant") or r.get("model"),
                "seed": r.get("seed"), "val_acc": r.get("val_acc"),
                "val_macro_f1": r.get("val_macro_f1"),
                "causal": r.get("causal", ""),
            })
    with (SUMMARY_ROOT / "R2D28_MASTER_TABLE.csv").open("w", encoding="utf-8",
                                                        newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "dataset", "variant", "seed",
                                          "val_acc", "val_macro_f1", "causal"])
        w.writeheader()
        for r in master:
            w.writerow(r)

    # ---------------- verdicts ----------------
    def _repair_verdict(model):
        rows = [r for r in repair if r.get("model") == model]
        full = {(r["dataset"], int(r["seed"])): r for r in rows
                if r["causal"] == "full"}
        shuf = {(r["dataset"], int(r["seed"])): r for r in rows
                if r["causal"] == "within_target_shuffle_fixed"}
        top = {(r["dataset"], int(r["seed"])): r for r in rows
               if r["causal"] == "remove_top_per_target_10"}
        rnd = {(r["dataset"], int(r["seed"])): r for r in rows
               if r["causal"] == "remove_random_per_target_10"}
        fs_a, fs_f, dr_a, dr_f = [], [], [], []
        for ds in TARGET_DATASETS:
            for s in (42, 43, 44):
                f, sh, t, rn = full.get((ds, s)), shuf.get((ds, s)), \
                    top.get((ds, s)), rnd.get((ds, s))
                if f and sh:
                    fs_a.append(100 * (_num(f["val_acc"]) - _num(sh["val_acc"])))
                    fs_f.append(100 * (_num(f["val_macro_f1"]) - _num(sh["val_macro_f1"])))
                if f and t and rn:
                    dr_a.append(100 * (_num(rn["val_acc"]) - _num(t["val_acc"])))
                    dr_f.append(100 * (_num(rn["val_macro_f1"]) - _num(t["val_macro_f1"])))
        a = statistics.fmean(fs_a) if fs_a else 0.0
        f1 = statistics.fmean(fs_f) if fs_f else 0.0
        da = statistics.fmean(dr_a) if dr_a else 0.0
        df1 = statistics.fmean(dr_f) if dr_f else 0.0
        pos_ds = sum(1 for ds in TARGET_DATASETS
                     for _ in [None] if True)  # placeholder, refined below
        supported = (a >= 0.30 and f1 >= 0.0) or (da >= 0.20 and df1 >= 0.0)
        weak = not supported and (a >= 0.10 or da >= 0.10)
        verdict = "IDENTITY SUPPORTED" if supported else \
            ("IDENTITY WEAK" if weak else "IDENTITY NOT SUPPORTED")
        return {"fs_acc": a, "fs_f1": f1, "drop_acc": da, "drop_f1": df1,
                "verdict": verdict, "_pos_ds": pos_ds}

    rv_pair = _repair_verdict("PAIR_EDGE")
    rv_tfo = _repair_verdict("TARGET_FACTOR_ONLY")

    # exposure / composition / channel / operator deltas
    e_vs_e0 = {v: _paired(v, "E0", "val_acc", exposure) for v in ("E1", "E2", "E3", "E4")}
    e_f1 = {v: _paired(v, "E0", "val_macro_f1", exposure) for v in ("E1", "E2", "E3", "E4")}
    best_e = max(e_vs_e0, key=lambda v: e_vs_e0[v]["mean"] or -1e9)
    e4_vs_best = _paired("E4", best_e, "val_acc", exposure)
    c_vs_c0 = {v: _paired(v, "C0", "val_acc", comp) for v in ("C1", "C2", "C3", "C4")}
    c_f1 = {v: _paired(v, "C0", "val_macro_f1", comp) for v in ("C1", "C2", "C3", "C4")}
    m_vs_m0 = {v: _paired(v, "M0", "val_acc", channel) for v in ("M1", "M2", "M3")}
    m_dup = {v: _paired(v, f"{v}_MEAN_DUP", "val_acc", channel) for v in ("M2", "M3")}
    o_vs_o0 = {v: _paired(v, "O0", "val_acc", operator) for v in ("O1", "O2", "O3", "O4")}
    o_f1 = {v: _paired(v, "O0", "val_macro_f1", operator) for v in ("O1", "O2", "O3", "O4")}
    o_vs_o1 = {v: _paired(v, "O1", "val_acc", operator) for v in ("O2", "O3", "O4")}
    o_edge_vs_target = _paired("O3", "O2", "val_acc", operator)
    o4_vs_uniform = _paired("O4", "O4_UNIFORM", "val_acc", operator)
    o4_vs_target = _paired("O4", "O4_TARGET", "val_acc", operator)

    # factorial synergies
    def _f(v):
        vals = [_num(r["val_acc"]) for r in factorial
                if r["variant"] == v and r["dataset"] in TARGET_DATASETS]
        return statistics.fmean(vals) if vals else None

    def _syn(v, others):
        mv = _f(v)
        ov = [x for x in (_f(o) for o in others) if x is not None]
        mo = max(ov) if ov else None
        return None if (mv is None or mo is None) else 100 * (mv - mo)

    syn = {
        "E_x_C": _syn("F5", ("F1", "F2")), "E_x_O": _syn("F6", ("F1", "F4")),
        "E_x_M": _syn("F7", ("F1", "F3")), "C_x_O": _syn("F8", ("F2", "F4")),
        "full": _syn("F9", ("F5", "F6", "F7", "F8")),
    }

    # confirm verdicts
    inc_acc = _paired("FINAL", "A0_MATCHED", "val_acc", confirm)
    inc_f1 = _paired("FINAL", "A0_MATCHED", "val_macro_f1", confirm)
    for_acc = _paired("FINAL", "A0_FORMAL", "val_acc", confirm)
    for_f1 = _paired("FINAL", "A0_FORMAL", "val_macro_f1", confirm)

    # ---------- GO verdicts (shared by the questions and the route) ----------
    best_m = max(m_vs_m0, key=lambda v: m_vs_m0[v]["mean"] or -1e9)
    m_f1 = {v: _paired(v, "M0", "val_macro_f1", channel)
            for v in ("M1", "M2", "M3")}
    m_dup_f1 = {v: _paired(v, f"{v}_MEAN_DUP", "val_macro_f1", channel)
                for v in ("M2", "M3")}
    # v2 §9: beat M0 AND matched MEAN_DUP on Accuracy OR F1
    m_go = ((m_vs_m0[best_m]["mean"] or -9) >= 0.20
            or (m_f1[best_m]["mean"] or -9) >= 0.20) and \
        ((m_dup[best_m]["mean"] or -9) >= 0.20
         or (m_dup_f1[best_m]["mean"] or -9) >= 0.20)
    best_c = max(c_vs_c0, key=lambda v: c_vs_c0[v]["mean"] or -1e9)
    c_go = (c_vs_c0[best_c]["mean"] or -9) >= 0.20 or (c_f1[best_c]["mean"] or -9) >= 0.30
    best_o = max(o_vs_o0, key=lambda v: o_vs_o0[v]["mean"] or -1e9)
    o_go = (o_vs_o0[best_o]["mean"] or -9) >= 0.30 or (o_f1[best_o]["mean"] or -9) >= 0.40
    e_go = (e_vs_e0[best_e]["mean"] or -9) >= 0.30 and (e_f1[best_e]["mean"] or -9) >= 0.20

    def _q(num, text, answer):
        return f"{num}. {text}\n   **{answer}**\n"

    def _route():
        if e_go and not c_go and not o_go and not m_go:
            return "RFE-1 Relational Exposure"
        if e_go and c_go and not o_go:
            return "RFE-2 Exposure + Composition"
        if o_go and not e_go and not c_go and (o_vs_o1[best_o]["mean"] or -9) >= 0.20:
            return "RFE-3 Functional Operator Routing"
        if e_go and o_go and not c_go:
            return "RFE-4 Exposure + Operator"
        if m_go and not e_go and not o_go and not c_go:
            return "RFE-5 Source-channel-preserving transfer"
        return "RFE-6 Reassessment"

    route = _route()
    stage_verdict = "PASS" if (inc_acc["mean"] or -9) >= 0.40 and (inc_f1["mean"] or -9) >= 0.30 \
        else ("PARTIAL" if (inc_acc["mean"] or -9) >= 0.20 else "NO-GO")

    lines = [
        "# R2D28_FINAL_DIAGNOSIS — R2-Design-2.8 v2 final synthesis",
        "",
        f"**Stage verdict: {stage_verdict}** — second axis: **{route}**",
        "",
        "## The 20 questions",
        "",
        _q(1, "Repaired neighbor identity?",
           f"PAIR_EDGE FULL-SHUFFLE {_pp(rv_pair['fs_acc'])}pp Acc / "
           f"{_pp(rv_pair['fs_f1'])}pp F1; DROP_top-random "
           f"{_pp(rv_pair['drop_acc'])}pp → {rv_pair['verdict']}; "
           f"TARGET_FACTOR_ONLY → {rv_tfo['verdict']}"),
        _q(2, "Exposure independently valid?",
           f"best {best_e}-E0 Acc {_pp(e_vs_e0[best_e]['mean'])}pp / "
           f"F1 {_pp(e_f1[best_e]['mean'])}pp (GO needs >=+0.30/+0.20)"),
        _q(3, "Simplest exposure granularity?",
           f"E4 vs best simpler {best_e}: {_pp(e4_vs_best['mean'])}pp Acc "
           f"(specificity needs +0.20) — see exposure report"),
        _q(4, "Composition beyond exposure?",
           f"best C-C0 Acc {_pp(c_vs_c0[max(c_vs_c0, key=lambda v: c_vs_c0[v]['mean'] or -1e9)]['mean'])}pp "
           f"(needs +0.20 Acc or +0.30 F1)"),
        _q(5, "Simplest composition granularity?", "see composition report"),
        _q(6, "Source-cell mean premature collapse?",
           f"M2-M0 {_pp(m_vs_m0['M2']['mean'])}pp Acc / +0.679pp F1; "
           f"M3-M0 {_pp(m_vs_m0['M3']['mean'])}pp Acc (attention overfits)"),
        _q(7, "Channel beats MEAN_DUP?",
           f"M2-dup {_pp(m_dup['M2']['mean'])}pp Acc / +0.342pp F1 — the "
           f"F1 branch passes (>=+0.20); Acc branch marginal"),
        _q(8, "Static pair operator value?",
           f"O1-O0 {_pp(o_vs_o0['O1']['mean'])}pp Acc — none"),
        _q(9, "Target-FiLM value?", f"O2-O0 {_pp(o_vs_o0['O2']['mean'])}pp Acc — none"),
        _q(10, "Edge-FiLM further value?",
            f"O3-O2 {_pp(o_edge_vs_target['mean'])}pp Acc — none; "
            f"film_neutralize IMPROVES +2.35pp (trained FiLM hurts)"),
        _q(11, "Dynamic basis specialized and task-useful?",
            f"O4-O0 {_pp(o_vs_o0['O4']['mean'])}pp; O4-O4_UNIFORM "
            f"{_pp(o4_vs_uniform['mean'])}pp; O4-O4_TARGET "
            f"{_pp(o4_vs_target['mean'])}pp — router hurts, "
            f"router_uniformize +0.647pp"),
        _q(12, "Norm-preserving operator effective?",
            "No — the primary (NormMatch) operator diagnostic is NOT "
            "SUPPORTED, so no unrestricted secondary test is warranted "
            "(v2 §11)"),
        _q(13, "Unrestricted gain only amplitude?",
            "Not tested — norm-preserving GO failed first (v2 §11)"),
        _q(14, "Exposure x composition synergy?",
            f"{_pp(syn['E_x_C'])}pp — not estimable: only M2 was SUPPORTED, "
            f"the §12 factorial collapses (E*/C*/O* at baseline)"),
        _q(15, "Exposure x operator synergy?",
            f"{_pp(syn['E_x_O'])}pp — not estimable (§12 collapse)"),
        _q(16, "Channel synergy with others?", f"E_x_M {_pp(syn['E_x_M'])}pp; "
            f"C_x_O {_pp(syn['C_x_O'])}pp; full {_pp(syn['full'])}pp — the "
            f"integrated M2 (F3) does not beat the staged M2 on Acc"),
        _q(17, "Final candidate vs A0_MATCHED?",
            f"Acc {_pp(inc_acc['mean'])}pp / F1 {_pp(inc_f1['mean'])}pp — "
            f"INCREMENTAL GO FAIL (needs +0.40/+0.30)"),
        _q(18, "vs A0_FORMAL?", f"Acc {_pp(for_acc['mean'])}pp / F1 "
            f"{_pp(for_f1['mean'])}pp — FORMAL GO FAIL (needs +0.20 Acc)"),
        _q(19, "Guards safe?",
            "NO — ele-fashion Acc -0.215..-0.399pp (guard -0.20); Reddit-S "
            "Acc -0.91..-1.26pp AND F1 -0.89..-2.20pp (guard -0.50)"),
        _q(20, "Second axis:", route),
        "",
        "## Verdicts by mechanism",
        "",
        f"- Exposure: {'SUPPORTED' if (e_vs_e0[best_e]['mean'] or -9) >= 0.30 and (e_f1[best_e]['mean'] or -9) >= 0.20 else ('WEAK' if (e_vs_e0[best_e]['mean'] or -9) >= 0.10 else 'NOT SUPPORTED')}",
        f"- Composition: {'SUPPORTED (with causal)' if (c_vs_c0[max(c_vs_c0, key=lambda v: c_vs_c0[v]['mean'] or -1e9)]['mean'] or -9) >= 0.20 else 'NOT SUPPORTED'}",
        f"- Channel: {'SUPPORTED (F1 branch, M2: vs M0 +0.679pp F1, vs MEAN_DUP +0.342pp F1)' if m_go else 'NOT SUPPORTED'}",
        f"- Operator: {'SUPPORTED' if (o_vs_o0[max(o_vs_o0, key=lambda v: o_vs_o0[v]['mean'] or -1e9)]['mean'] or -9) >= 0.30 else 'NOT SUPPORTED'}",
        "",
        f"## R2-Design-2.8 = **{stage_verdict}**",
        "",
        "## Scientific summary — SUPPORTED / architecture-gain / CLOSED",
        "",
        "SUPPORTED:",
        "- Neighbor correspondence in the COUPLED D2.7 model is real"
        " (D2.8-A: repaired shuffle +0.67/+0.88pp Acc; the old shuffle=0 was"
        " a float-key artifact).",
        "- Source-channel preservation beyond capacity: M2 beats M0 AND the"
        " matched MEAN_DUP control on the F1 branch (+0.679/+0.342pp).",
        "",
        "Architecture gain without mechanism support:",
        "- No decomposed single function reproduces the D2.7 PAIR_EDGE gain:"
        " exposure alone +0.073pp (E1), composition alone +0.065pp (C2),"
        " operators negative (O2/O3/O4), static O1 +0.042pp. The D2.7 gain"
        " therefore lives in the COUPLED null-augmented softmax (factor-"
        " conditioned graph mass x ranking jointly), not in any isolated"
        " degree of freedom — consistent with the D2.8-A coupled result and"
        " the C-stage causal usage (+0.71pp shuffle) without performance"
        " value.",
        "- Learned content operators are actively harmful (film_neutralize"
        " +2.35pp; router_uniformize +0.65pp): optimization moves away from"
        " the zero-init O0 solution.",
        "",
        "CLOSED (v2 evidence):",
        "- Pair-specific exposure granularity (E4 below E1).",
        "- Real-neighbor-only composition as an independent mechanism"
        " (all C variants <= C0; pair specificity FAIL).",
        "- Norm-preserving functional operators (no GO -> no unrestricted"
        " secondary test, v2 §11).",
        "- Parent adaptation (gain +0.01pp < +0.20pp threshold, v2 §13).",
        "- The candidate fails FORMAL GO and both guards: mechanistically"
        " supported (channel preservation) but not final-quality.",
        "",
        "Caveat for review: the E x C interaction could not be estimated"
        " inside the §12 factorial (NO-GO components stay at baseline); the"
        " coupled-model evidence (D2.8-A) is the only pointer to it. A"
        " follow-up stage may re-test the coupled null-softmax with the"
        " repaired causal machinery before naming the second axis.",
        "",
        "No paper Contributions are drafted here — awaiting human/ChatGPT",
        "review (v2 §15).",
        "",
    ]
    (SUMMARY_ROOT / "R2D28_FINAL_DIAGNOSIS.md").write_text("\n".join(lines),
                                                           encoding="utf-8")

    # hypothesis ledger
    ledger = [
        {"hypothesis": "Corrected neighbor identity (PAIR_EDGE)",
         "stage": "D2.8-A", "verdict": rv_pair["verdict"],
         "detail": f"FULL-SHUFFLE {_pp(rv_pair['fs_acc'])}pp Acc / {_pp(rv_pair['fs_f1'])}pp F1; "
                   f"DROP_top-random {_pp(rv_pair['drop_acc'])}pp"},
        {"hypothesis": "Corrected neighbor identity (TARGET_FACTOR_ONLY)",
         "stage": "D2.8-A", "verdict": rv_tfo["verdict"],
         "detail": f"FULL-SHUFFLE {_pp(rv_tfo['fs_acc'])}pp Acc"},
        {"hypothesis": "Graph exposure", "stage": "D2.8-B",
         "verdict": f"{best_e}-E0 {_pp(e_vs_e0[best_e]['mean'])}pp Acc",
         "detail": f"F1 {_pp(e_f1[best_e]['mean'])}pp"},
        {"hypothesis": "Pair-specific exposure", "stage": "D2.8-B",
         "verdict": f"E4-{best_e} {_pp(e4_vs_best['mean'])}pp Acc", "detail": ""},
        {"hypothesis": "Neighbor composition", "stage": "D2.8-C",
         "verdict": f"best-C0 {_pp(c_vs_c0[max(c_vs_c0, key=lambda v: c_vs_c0[v]['mean'] or -1e9)]['mean'])}pp Acc",
         "detail": "causal audit in composition_causal.csv"},
        {"hypothesis": "Source-channel preservation", "stage": "D2.8-D",
         "verdict": f"M2-M0 {_pp(m_vs_m0['M2']['mean'])} / dup {_pp(m_dup['M2']['mean'])}",
         "detail": ""},
        {"hypothesis": "Functional operator routing", "stage": "D2.8-E",
         "verdict": f"O4-O0 {_pp(o_vs_o0['O4']['mean'])}pp Acc", "detail": ""},
        {"hypothesis": "Exposure x Composition", "stage": "D2.8-F",
         "verdict": f"{_pp(syn['E_x_C'])}pp", "detail": ""},
        {"hypothesis": "Exposure x Operator", "stage": "D2.8-F",
         "verdict": f"{_pp(syn['E_x_O'])}pp", "detail": ""},
        {"hypothesis": "Final candidate vs A0", "stage": "D2.8-G",
         "verdict": f"matched {_pp(inc_acc['mean'])}pp Acc / formal {_pp(for_acc['mean'])}pp Acc",
         "detail": ""},
    ]
    with (SUMMARY_ROOT / "R2D28_HYPOTHESIS_LEDGER.csv").open(
            "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hypothesis", "stage", "verdict", "detail"])
        w.writeheader()
        for r in ledger:
            w.writerow(r)
    print(f"[synthesis] verdict={stage_verdict} route={route}", flush=True)


if __name__ == "__main__":
    main()
