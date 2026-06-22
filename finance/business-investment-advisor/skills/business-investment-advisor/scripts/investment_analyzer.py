#!/usr/bin/env python3
"""Deterministic investment analysis for the business-investment-advisor skill.

Replaces in-context LLM arithmetic (ROI, payback, NPV, IRR, scenario cases,
the 1-5 scoring rubric, and IRR-ranked budget allocation) with exact,
stdlib-only computation. The LLM runs this once and reads a compact result
table instead of generating step-by-step math — fewer tokens, faster, and
correct (LLMs systematically mis-compute multi-period NPV and iterative IRR).

No third-party dependencies. No network. No LLM calls.

Usage:
  # Single investment (Mode 1)
  python3 investment_analyzer.py single \
      --cost 50000 --annual-cash-flow 22000 --years 5 --discount-rate 0.12 \
      [--salvage 5000] [--score 4,5,3,4,3,4] [--output text|json]

  # Cash flows can vary per year instead of a flat annual figure:
  python3 investment_analyzer.py single --cost 50000 \
      --cash-flows 10000,15000,20000,20000,20000 --discount-rate 0.12

  # Compare / allocate a fixed budget across options (Mode 2)
  python3 investment_analyzer.py allocate --budget 100000 --options options.json

  # Self-test
  python3 investment_analyzer.py --sample
"""
from __future__ import annotations

import argparse
import json
import sys


# --------------------------------------------------------------------------
# Core finance primitives (deterministic)
# --------------------------------------------------------------------------
def npv(rate: float, cash_flows: list[float], initial_investment: float) -> float:
    """Net present value. cash_flows are end-of-period inflows for t=1..N."""
    total = -abs(initial_investment)
    for t, cf in enumerate(cash_flows, start=1):
        total += cf / ((1.0 + rate) ** t)
    return total


def irr(cash_flows: list[float], initial_investment: float) -> float | None:
    """Internal rate of return via bisection on a sign-bracketed interval.

    Returns the rate where NPV == 0, or None when no sign change exists in a
    plausible range (e.g. the project never pays back). Bisection is used
    instead of Newton's method because it cannot diverge and needs no
    derivative — robust for the irregular cash-flow shapes this skill sees.
    """
    lo, hi = -0.9999, 10.0  # -99.99% .. +1000%
    f_lo = npv(lo, cash_flows, initial_investment)
    f_hi = npv(hi, cash_flows, initial_investment)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None  # no sign change -> no real IRR in range
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid, cash_flows, initial_investment)
        if abs(f_mid) < 1e-9:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return (lo + hi) / 2.0


def roi(total_returns: float, cost: float) -> float:
    """Simple ROI percentage over the analysis period (ignores time value)."""
    if cost == 0:
        return 0.0
    return (total_returns - cost) / cost * 100.0


def payback_period(cost: float, cash_flows: list[float]) -> float | None:
    """Years to recover the initial investment from cumulative cash flow.

    Interpolates within the year recovery occurs. Returns None if never
    recovered over the supplied horizon.
    """
    cumulative = 0.0
    for t, cf in enumerate(cash_flows, start=1):
        prev = cumulative
        cumulative += cf
        if cumulative >= cost:
            shortfall = cost - prev
            fraction = shortfall / cf if cf else 0.0
            return (t - 1) + fraction
    return None


# --------------------------------------------------------------------------
# Scoring rubric (matches SKILL.md: 6 dimensions, 1-5 each, 6-30 total)
# --------------------------------------------------------------------------
SCORE_DIMENSIONS = [
    "roi", "payback", "strategic_fit", "risk", "reversibility", "cash_flow_impact",
]


def score_verdict(total: int) -> str:
    if total <= 12:
        return "DON'T DO IT"
    if total <= 20:
        return "NEEDS MORE ANALYSIS"
    return "STRONG INVESTMENT"


def recommendation(npv_value: float, irr_value: float | None,
                   hurdle: float, payback: float | None, years: float) -> str:
    if npv_value < 0:
        return "Do not proceed"
    if payback is not None and payback > years:
        return "Do not proceed"
    if irr_value is not None and irr_value < hurdle:
        return "Proceed with conditions"
    if payback is not None and payback > 0.8 * years:
        return "Proceed with conditions"
    return "Proceed"


# --------------------------------------------------------------------------
# Mode 1 — single investment
# --------------------------------------------------------------------------
def analyze_single(cost: float, cash_flows: list[float], discount_rate: float,
                   hurdle: float, salvage: float, score: list[int] | None) -> dict:
    flows = list(cash_flows)
    if salvage:
        flows[-1] += salvage
    years = len(flows)
    total_returns = sum(flows)
    npv_value = npv(discount_rate, flows, cost)
    irr_value = irr(flows, cost)
    pb = payback_period(cost, flows)
    roi_value = roi(total_returns, cost)

    # Downside (-40% inflows) and upside (+20% inflows) scenarios.
    down_flows = [cf * 0.6 for cf in flows]
    up_flows = [cf * 1.2 for cf in flows]
    scenarios = {
        "downside_minus_40pct": {
            "npv": round(npv(discount_rate, down_flows, cost), 2),
            "irr": _pct(irr(down_flows, cost)),
            "payback_years": _round(payback_period(cost, down_flows)),
        },
        "base": {
            "npv": round(npv_value, 2),
            "irr": _pct(irr_value),
            "payback_years": _round(pb),
        },
        "upside_plus_20pct": {
            "npv": round(npv(discount_rate, up_flows, cost), 2),
            "irr": _pct(irr(up_flows, cost)),
            "payback_years": _round(payback_period(cost, up_flows)),
        },
    }

    result = {
        "mode": "single",
        "inputs": {
            "total_investment": cost,
            "cash_flows": flows,
            "years": years,
            "discount_rate": discount_rate,
            "hurdle_rate": hurdle,
            "salvage": salvage,
        },
        "metrics": {
            "roi_pct": round(roi_value, 1),
            "payback_years": _round(pb),
            "npv": round(npv_value, 2),
            "irr_pct": _pct(irr_value),
            "annual_net_cash_flow_avg": round(total_returns / years, 2) if years else 0.0,
        },
        "scenarios": scenarios,
        "recommendation": recommendation(npv_value, irr_value, hurdle, pb, years),
        "flags": _flags(npv_value, irr_value, hurdle, pb, years),
    }

    if score:
        if len(score) != len(SCORE_DIMENSIONS):
            raise ValueError(
                f"--score needs {len(SCORE_DIMENSIONS)} values "
                f"({','.join(SCORE_DIMENSIONS)}); got {len(score)}"
            )
        total = sum(score)
        result["score"] = {
            "by_dimension": dict(zip(SCORE_DIMENSIONS, score)),
            "total": total,
            "max": 30,
            "verdict": score_verdict(total),
        }
    return result


def _flags(npv_value, irr_value, hurdle, pb, years) -> list[str]:
    out = []
    if npv_value < 0:
        out.append("Negative NPV — investment destroys value at this discount rate.")
    if pb is None:
        out.append("Never pays back over the supplied horizon.")
    elif pb > years:
        out.append("Payback exceeds useful life — recommend against.")
    elif pb > 0.8 * years:
        out.append("Payback > 80% of useful life — marginal at best.")
    if irr_value is not None and irr_value < hurdle:
        out.append(f"IRR {irr_value*100:.1f}% below hurdle {hurdle*100:.0f}%.")
    return out


# --------------------------------------------------------------------------
# Mode 2 — budget allocation
# --------------------------------------------------------------------------
def analyze_allocation(budget: float, options: list[dict]) -> dict:
    enriched = []
    for opt in options:
        cost = float(opt["cost"])
        flows = _resolve_flows(opt)
        rate = float(opt.get("discount_rate", 0.12))
        n = npv(rate, flows, cost)
        r = irr(flows, cost)
        pb = payback_period(cost, flows)
        enriched.append({
            "name": opt.get("name", "unnamed"),
            "cost": cost,
            "npv": round(n, 2),
            "irr_pct": _pct(r),
            "payback_years": _round(pb),
            "_irr_sort": r if r is not None else -1.0,
            "_quick_win": pb is not None and pb < 0.5,
            "_npv": n,
        })

    # Quick wins (payback < 6 months) first, then by IRR descending.
    ranked = sorted(enriched, key=lambda o: (not o["_quick_win"], -o["_irr_sort"]))

    funded, remaining, allocation = [], budget, []
    for o in ranked:
        decision = "SKIP"
        reason = ""
        if o["_npv"] < 0:
            reason = "negative NPV — fund only with a named strategic reason"
        elif o["cost"] <= remaining:
            decision = "FUND"
            remaining -= o["cost"]
            funded.append(o["name"])
        else:
            reason = "exceeds remaining budget"
        allocation.append({
            "name": o["name"], "cost": o["cost"], "npv": o["npv"],
            "irr_pct": o["irr_pct"], "payback_years": o["payback_years"],
            "quick_win": o["_quick_win"], "decision": decision, "reason": reason,
        })

    return {
        "mode": "allocate",
        "budget": budget,
        "budget_remaining": round(remaining, 2),
        "funded": funded,
        "ranked_allocation": allocation,
    }


def _resolve_flows(opt: dict) -> list[float]:
    if "cash_flows" in opt:
        return [float(x) for x in opt["cash_flows"]]
    annual = float(opt["annual_cash_flow"])
    years = int(opt["years"])
    return [annual] * years


# --------------------------------------------------------------------------
# helpers + rendering
# --------------------------------------------------------------------------
def _round(v, nd=2):
    return None if v is None else round(v, nd)


def _pct(rate):
    return None if rate is None else round(rate * 100.0, 1)


def render_single_text(r: dict) -> str:
    m = r["metrics"]
    lines = [
        f"RECOMMENDATION: {r['recommendation']}",
        "",
        "THE NUMBERS",
        f"  Total Investment        ${r['inputs']['total_investment']:,.0f}",
        f"  Annual Net Cash Flow    ${m['annual_net_cash_flow_avg']:,.0f} (avg)",
        f"  Payback Period          {('%.2f yrs' % m['payback_years']) if m['payback_years'] is not None else 'never'}",
        f"  ROI (period)            {m['roi_pct']}%",
        f"  NPV @ {r['inputs']['discount_rate']*100:.0f}%            ${m['npv']:,.2f}",
        f"  IRR                     {('%.1f%%' % m['irr_pct']) if m['irr_pct'] is not None else 'n/a'}",
    ]
    if "score" in r:
        s = r["score"]
        lines.append(f"  Investment Score        {s['total']}/30 — {s['verdict']}")
    lines += ["", "SCENARIOS (NPV / IRR / payback)"]
    for k, sc in r["scenarios"].items():
        irr_s = ("%.1f%%" % sc["irr"]) if sc["irr"] is not None else "n/a"
        pb_s = ("%.2f yrs" % sc["payback_years"]) if sc["payback_years"] is not None else "never"
        lines.append(f"  {k:<22} ${sc['npv']:>12,.0f}   {irr_s:<7}  {pb_s}")
    if r["flags"]:
        lines += ["", "FLAGS"] + [f"  - {f}" for f in r["flags"]]
    return "\n".join(lines)


def render_alloc_text(r: dict) -> str:
    lines = [f"BUDGET ${r['budget']:,.0f}  |  remaining ${r['budget_remaining']:,.0f}",
             f"FUNDED: {', '.join(r['funded']) or '(none)'}", "",
             "RANKED ALLOCATION (quick wins first, then IRR desc)"]
    for a in r["ranked_allocation"]:
        irr_s = ("%.1f%%" % a["irr_pct"]) if a["irr_pct"] is not None else "n/a"
        tag = " [quick win]" if a["quick_win"] else ""
        line = f"  {a['decision']:<5} {a['name']:<24} ${a['cost']:>10,.0f}  IRR {irr_s:<7} NPV ${a['npv']:>12,.0f}{tag}"
        if a["reason"]:
            line += f"  ({a['reason']})"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def run_sample() -> int:
    print("=== Mode 1: single investment ===")
    r = analyze_single(
        cost=50000, cash_flows=[22000] * 5, discount_rate=0.12,
        hurdle=0.15, salvage=5000, score=[4, 5, 3, 4, 3, 4],
    )
    print(render_single_text(r))
    print("\n=== Mode 2: budget allocation ===")
    a = analyze_allocation(100000, [
        {"name": "Equipment", "cost": 50000, "annual_cash_flow": 22000, "years": 5},
        {"name": "Automation", "cost": 30000, "annual_cash_flow": 28000, "years": 3},
        {"name": "Office remodel", "cost": 40000, "annual_cash_flow": 6000, "years": 5},
    ])
    print(render_alloc_text(a))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deterministic investment analysis (ROI/payback/NPV/IRR/scoring/allocation).")
    p.add_argument("--sample", action="store_true", help="Run a built-in self-test and exit.")
    sub = p.add_subparsers(dest="mode")

    s = sub.add_parser("single", help="Evaluate one investment.")
    s.add_argument("--cost", type=float, required=True)
    s.add_argument("--annual-cash-flow", type=float)
    s.add_argument("--cash-flows", type=_parse_floats, help="Comma-separated per-year inflows.")
    s.add_argument("--years", type=int)
    s.add_argument("--discount-rate", type=float, default=0.12)
    s.add_argument("--hurdle-rate", type=float, default=0.15)
    s.add_argument("--salvage", type=float, default=0.0)
    s.add_argument("--score", type=_parse_ints, help="6 comma-separated 1-5 scores.")
    s.add_argument("--output", "--format", dest="output",
                   choices=["text", "json"], default="text",
                   help="Output format (--format alias matches sibling finance tools).")

    a = sub.add_parser("allocate", help="Rank/allocate a fixed budget across options.")
    a.add_argument("--budget", type=float, required=True)
    a.add_argument("--options", required=True, help="Path to JSON list of options.")
    a.add_argument("--output", "--format", dest="output",
                   choices=["text", "json"], default="text",
                   help="Output format (--format alias matches sibling finance tools).")

    args = p.parse_args(argv)

    if args.sample:
        return run_sample()

    if args.mode == "single":
        if args.cash_flows:
            flows = args.cash_flows
        elif args.annual_cash_flow is not None and args.years:
            flows = [args.annual_cash_flow] * args.years
        else:
            p.error("provide either --cash-flows OR (--annual-cash-flow and --years)")
        try:
            r = analyze_single(args.cost, flows, args.discount_rate,
                               args.hurdle_rate, args.salvage, args.score)
        except ValueError as e:
            p.error(str(e))
        print(json.dumps(r, indent=2) if args.output == "json" else render_single_text(r))
        return 0

    if args.mode == "allocate":
        with open(args.options, encoding="utf-8") as fh:
            options = json.load(fh)
        r = analyze_allocation(args.budget, options)
        print(json.dumps(r, indent=2) if args.output == "json" else render_alloc_text(r))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
