"""Small, reproducible synthetic probe of the production visualization judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv
from plotnine import (
    aes,
    element_text,
    geom_col,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_fill_gradient,
    scale_x_discrete,
    theme,
    theme_minimal,
)
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from src.validation.eval_models import VisualizationOutput
from src.validation.evaluators.visualization import ValidateVisualizationQuality

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "visual_judge_probe"
SITES = ["North", "East", "South", "West"]
VALUES = [12, 19, 27, 15]
CASES = ("control", "type", "swap", "data", "aesthetics", "encoding")
EXPECTED = {
    "control": {"type_correct": True, "encoding_correct": True, "data_correct": True, "aesthetics_correct": True},
    "type": {"type_correct": False, "encoding_correct": True, "data_correct": True, "aesthetics_correct": True},
    "swap": {"type_correct": True, "encoding_correct": True, "data_correct": False, "aesthetics_correct": True},
    "data": {"type_correct": True, "encoding_correct": True, "data_correct": False, "aesthetics_correct": True},
    "aesthetics": {"type_correct": True, "encoding_correct": True, "data_correct": True, "aesthetics_correct": False},
    "encoding": {"type_correct": True, "encoding_correct": False, "data_correct": False, "aesthetics_correct": False},
}


def plot_for(case: str) -> ggplot:
    """Use the same synthetic observations with one deliberate change per case."""
    values = VALUES.copy()
    sites = SITES.copy()
    if case == "swap":
        sites[0], sites[2] = sites[2], sites[0]
    elif case == "data":
        values = [v * 4 for v in values]  # Sum of four days, labelled as mean.
    frame = pd.DataFrame({"site": pd.Categorical(sites, categories=SITES, ordered=True), "minutes": values})
    chart = ggplot(frame, aes("site", "minutes"))
    if case == "type":
        chart += geom_line(aes(group=1), color="#17669c", size=1.2)
        chart += geom_point(color="#17669c", size=3)
    elif case == "encoding":
        # Keep the values, but encode them only through color. Equal bar heights
        # incorrectly imply equal quantities on the labelled quantitative axis.
        frame["bar_height"] = 1
        chart = ggplot(frame, aes("site", "bar_height", fill="minutes"))
        chart += geom_col(width=0.65)
        chart += scale_fill_gradient(low="#d7e9f5", high="#17669c", name="Outage (minutes/day)")
    else:
        chart += geom_col(fill="#17669c", width=0.65)
    chart += scale_x_discrete(limits=SITES)
    chart += labs(title="Mean daily outage by substation", x="Substation", y="Outage (minutes/day)")
    chart += theme_minimal(base_size=13)
    chart += theme(figure_size=(8, 5), axis_title=element_text(size=14), axis_text=element_text(size=12))
    if case == "aesthetics":
        chart += labs(title="", x="", y="")
        chart += theme(axis_text=element_text(size=6))
    return chart


def render() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        chart = plot_for(case)
        chart.save(OUT / f"{case}.png", width=8, height=5, dpi=125, verbose=False)
        chart.save(OUT / f"{case}.pdf", width=8, height=5, verbose=False)


def run(cases: list[str], efforts: list[str]) -> None:
    load_dotenv(ROOT / ".env")
    model = OpenAIResponsesModel("gpt-6-luna")
    results_path = OUT / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    for effort in efforts:
        for case in cases:
            settings = OpenAIResponsesModelSettings(openai_reasoning_effort=effort)
            if effort == "none":
                settings["temperature"] = 0.0
            evaluator = ValidateVisualizationQuality(
                artifact_base_path=OUT,
                model=model,
                model_settings=settings,
            )
            ctx = SimpleNamespace(
                expected_output=[VisualizationOutput(image_path="control.png")],
                attributes={"visualization_image_path": str(OUT / f"{case}.png")},
            )
            result = evaluator.evaluate_sync(ctx)  # type: ignore[arg-type]
            actual = {name: bool(result[name].value) for name in EXPECTED[case]}
            rationale = result["data_correct"].reason
            row = {
                "case": case,
                "effort": effort,
                "expected": EXPECTED[case],
                "actual": actual,
                "matched": {name: actual[name] == target for name, target in EXPECTED[case].items()},
                "rationale": rationale,
            }
            rows.append(row)
            results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--efforts", nargs="+", choices=("none", "low", "medium"), default=["none"])
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    render()
    if not args.render_only:
        run(args.cases, args.efforts)
