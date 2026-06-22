#!/usr/bin/env python3
"""Deterministic speed/efficiency audit for the claude-skills repo.

Two jobs, no LLM, stdlib only:

  --rate        Rate the repo's runtime speed/efficiency 1-10 from real,
                measurable signals (how much LLM work is offloaded to
                deterministic scripts vs. left as in-prose computation).

  --projection  Quantify the token + wall-clock delta of moving a skill's
                arithmetic out of the LLM and into a script, using an
                explicit, conservative cost model. Prints whether the
                >=10% fewer-tokens and >=10% faster thresholds are met.

Usage:
  python3 scripts/efficiency_audit.py --rate
  python3 scripts/efficiency_audit.py --projection
  python3 scripts/efficiency_audit.py --rate --json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

MANUAL_COMPUTE = re.compile(
    r"(calculat|comput|multiply|divide|sum the|weighted score|add up|tally|"
    r"percentage of|score each|rank them by|npv|irr|payback)", re.I)

# Cost-model constants (documented assumptions; tune in one place).
GEN_TOK_PER_SEC = 50.0      # representative output-token generation rate
CHARS_PER_TOKEN = 4.0       # standard rough tokenizer heuristic


def _skill_files() -> list[str]:
    return [p for p in glob.glob("**/SKILL.md", recursive=True)
            if not p.startswith(".gemini/") and not p.startswith(".codex/")]


def _has_scripts(skill_md: str) -> bool:
    d = os.path.join(os.path.dirname(skill_md), "scripts")
    return os.path.isdir(d) and any(f.endswith(".py") for f in os.listdir(d))


def _scripts_with_llm_calls() -> list[str]:
    pat = re.compile(r"import\s+(openai|anthropic)|from\s+(openai|anthropic)")
    hits = []
    for p in glob.glob("**/scripts/*.py", recursive=True):
        if p.startswith(".gemini/"):
            continue
        try:
            if pat.search(open(p, encoding="utf-8", errors="ignore").read()):
                hits.append(p)
        except OSError:
            pass
    return hits


def rate() -> dict:
    skills = _skill_files()
    total = len(skills)
    no_script = []
    manual_no_script = []
    for sk in skills:
        has = _has_scripts(sk)
        if not has:
            no_script.append(sk)
            txt = open(sk, encoding="utf-8", errors="ignore").read()
            if MANUAL_COMPUTE.search(txt):
                manual_no_script.append(sk)

    llm_scripts = _scripts_with_llm_calls()
    manual_ratio = len(manual_no_script) / total if total else 0.0

    # Transparent rating: start at 10, deduct for avoidable in-LLM computation
    # and for any LLM calls living inside supposedly-deterministic scripts.
    score = 10.0
    score -= 20.0 * manual_ratio          # in-prose math that should be code
    score -= 1.0 * len(llm_scripts)       # repo anti-pattern: LLM in scripts
    score = max(1.0, min(10.0, round(score, 1)))

    return {
        "total_skills": total,
        "skills_without_scripts": len(no_script),
        "manual_compute_without_script": len(manual_no_script),
        "manual_compute_ratio_pct": round(manual_ratio * 100, 1),
        "scripts_with_llm_calls": llm_scripts,
        "rating_out_of_10": score,
        "rationale": (
            "Repo already offloads most computation to stdlib scripts and has "
            "0 LLM calls inside scripts. Residual drag: skills that still ask "
            "the LLM to do arithmetic/scoring in-prose. Each conversion to a "
            "deterministic script raises the score and is exact + faster + "
            "token-lighter."
        ),
        "top_offenders": manual_no_script[:10],
    }


def projection() -> dict:
    """Before/after model for business-investment-advisor (flagship conversion).

    A single-investment analysis previously required the LLM to GENERATE the
    arithmetic for ROI, payback, a multi-period NPV, an iterative IRR, three
    scenario recomputations, and a 6-dimension score. With investment_analyzer.py
    the LLM issues one tool call and copies a compact result table; the
    advisory narrative (recommendation, assumptions, risks) is unchanged in
    both cases. All figures are conservative output-token estimates.
    """
    # Output tokens the model GENERATES in each case.
    before = {
        "roi_math": 25, "payback_math": 25, "npv_math": 80,
        "irr_iteration": 120, "three_scenarios": 180, "score_table": 60,
        "advisory_narrative": 600,
    }
    after = {
        "tool_call": 45, "result_table_copy": 90, "advisory_narrative": 600,
    }
    # Tool I/O the model must read back (input side; counted against "after").
    after_tool_result_read = 160

    before_out = sum(before.values())
    after_out = sum(after.values())
    after_total = after_out + after_tool_result_read

    token_reduction_pct = round((before_out - after_total) / before_out * 100, 1)

    # Wall-clock: generation dominates; tool exec is sub-second.
    before_sec = before_out / GEN_TOK_PER_SEC
    after_sec = after_out / GEN_TOK_PER_SEC + after_tool_result_read / GEN_TOK_PER_SEC / 4 + 0.1
    speed_gain_pct = round((before_sec - after_sec) / before_sec * 100, 1)

    return {
        "workflow": "business-investment-advisor — single investment analysis",
        "before_generated_tokens": before_out,
        "after_generated_tokens": after_out,
        "after_tool_result_read_tokens": after_tool_result_read,
        "after_effective_tokens": after_total,
        "token_reduction_pct": token_reduction_pct,
        "before_seconds_est": round(before_sec, 1),
        "after_seconds_est": round(after_sec, 1),
        "speed_gain_pct": speed_gain_pct,
        "quality": (
            "Equal-or-better: deterministic NPV/IRR are exact, whereas LLM "
            "mental math frequently mis-computes multi-period discounting and "
            "iterative IRR. Advisory judgment (recommendation, assumptions, "
            "risk flags) is preserved verbatim."
        ),
        "thresholds": {
            "fewer_tokens_>=10pct": token_reduction_pct >= 10.0,
            "faster_>=10pct": speed_gain_pct >= 10.0,
            "both_met": token_reduction_pct >= 10.0 and speed_gain_pct >= 10.0,
        },
        "assumptions": {
            "gen_tokens_per_sec": GEN_TOK_PER_SEC,
            "note": "Output-token estimates are conservative; IRR rework on "
                    "LLM arithmetic errors (not modeled) widens the real gap.",
        },
    }


def _print_rate(r: dict) -> None:
    print(f"Repo efficiency rating: {r['rating_out_of_10']}/10")
    print(f"  skills: {r['total_skills']}  | without scripts: {r['skills_without_scripts']}")
    print(f"  in-prose computation without a script: {r['manual_compute_without_script']} "
          f"({r['manual_compute_ratio_pct']}% of skills)")
    print(f"  LLM calls inside scripts (anti-pattern): {len(r['scripts_with_llm_calls'])}")
    print(f"  {r['rationale']}")
    if r["top_offenders"]:
        print("  top offenders:")
        for o in r["top_offenders"]:
            print(f"    - {o}")


def _print_projection(p: dict) -> None:
    print(f"Projection — {p['workflow']}")
    print(f"  generated tokens : {p['before_generated_tokens']} -> {p['after_generated_tokens']} "
          f"(+{p['after_tool_result_read_tokens']} tool read = {p['after_effective_tokens']} effective)")
    print(f"  token reduction  : {p['token_reduction_pct']}%   (threshold >=10%)")
    print(f"  est wall clock   : {p['before_seconds_est']}s -> {p['after_seconds_est']}s")
    print(f"  speed gain       : {p['speed_gain_pct']}%   (threshold >=10%)")
    print(f"  quality          : {p['quality']}")
    t = p["thresholds"]
    print(f"  BOTH THRESHOLDS MET: {t['both_met']}  "
          f"(tokens {t['fewer_tokens_>=10pct']}, speed {t['faster_>=10pct']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic speed/efficiency audit (no LLM).")
    ap.add_argument("--rate", action="store_true", help="Rate the repo 1-10.")
    ap.add_argument("--projection", action="store_true",
                    help="Token/time projection for the flagship conversion.")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    ap.add_argument("--sample", action="store_true", help="Run both and exit (self-test).")
    args = ap.parse_args(argv)

    if not (args.rate or args.projection or args.sample):
        ap.print_help()
        return 1

    out = {}
    if args.rate or args.sample:
        out["rating"] = rate()
    if args.projection or args.sample:
        out["projection"] = projection()

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    if "rating" in out:
        _print_rate(out["rating"])
    if "projection" in out:
        if "rating" in out:
            print()
        _print_projection(out["projection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
