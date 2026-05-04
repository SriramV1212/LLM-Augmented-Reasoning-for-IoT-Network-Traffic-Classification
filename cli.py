#!/usr/bin/env python3
"""CLI for LLM-Augmented IoT traffic classification (ML + explainer, or zero-shot LLM)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from rich.console import Console

from llm_agent.config import ClassifyMode, load_config
from llm_agent.evaluator import evaluate_csv, write_eval_artifacts
from llm_agent.merger import merge_results
from llm_agent.openrouter_client import OpenRouterClient
from llm_agent.track_a_zero_shot import classify_flow_zero_shot
from llm_agent.track_b_ml_explainer import load_ml_models, run_track_b

console = Console()


def _add_openrouter_log_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log OpenRouter request/response summaries (URL, model, timings, message previews) to stderr",
    )
    ap.add_argument(
        "--debug-api",
        action="store_true",
        help="Longer previews in API logs; DEBUG level for llm_agent.openrouter_client",
    )


def _configure_openrouter_logging(args: argparse.Namespace) -> None:
    debug = bool(getattr(args, "debug_api", False))
    verbose = bool(getattr(args, "verbose", False)) or debug
    if not verbose:
        return
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("llm_agent.openrouter_client").setLevel(level)
    for name in ("openai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _classification_mode_type(value: str) -> ClassifyMode:
    """CLI ``--mode``: ``ml-agent`` / ``ml+agent`` (default) or ``zeroshot``."""
    t = str(value).strip().lower().replace("+", "_").replace("-", "_")
    if t in ("ml_agent", "mlagent"):
        return "ml_agent"
    if t in ("zeroshot", "zero_shot", "llm"):
        return "zeroshot"
    raise argparse.ArgumentTypeError(
        f"Unknown mode {value!r}. Use ml-agent (default) or zeroshot."
    )


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v))


def _load_flow_payload(path: Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "features" not in data:
        raise ValueError("JSON must contain a 'features' object with CICFlowMeter columns.")
    if not isinstance(data["features"], dict):
        raise ValueError("'features' must be a JSON object (string keys -> numbers).")
    return data


def classify_flow_from_args(
    args: argparse.Namespace, path: Path
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Load flow JSON, run selected tracks, merge. Returns (json_dict, exit_code).
    exit_code 0 on success; 2 on user/input errors.
    """
    mode: ClassifyMode = getattr(args, "mode", "ml_agent")
    skip_llm = bool(getattr(args, "skip_llm", False))

    need_key = mode == "zeroshot" or (mode == "ml_agent" and not skip_llm)
    cfg = load_config(require_api_key=need_key, env_file=args.env_file)
    cfg.classification_mode = mode
    if args.model:
        cfg.default_model = args.model
    if args.ml_backend:
        cfg.ml_classifier = args.ml_backend.lower()  # type: ignore[assignment]
    if args.models_dir:
        cfg.models_dir = Path(args.models_dir)
    if args.prompts_dir:
        cfg.prompts_dir = Path(args.prompts_dir)

    try:
        payload = _load_flow_payload(path)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in {path}:[/red] {e}")
        return None, 2
    features: Dict[str, Any] = {str(k): _to_float(v) for k, v in payload["features"].items()}
    flow_id = payload.get("flow_id")

    artifacts = load_ml_models(Path(cfg.models_dir))
    client: Optional[OpenRouterClient] = None
    if need_key:
        if not cfg.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY required for this track configuration.[/red]")
            return None, 2
        client = OpenRouterClient(cfg)

    zs = None
    tb = None

    if mode == "zeroshot":
        assert client is not None
        zs = classify_flow_zero_shot(
            features,
            artifacts.feature_cols,
            client,
            cfg,
            model=args.model,
            serializer_mode=args.serializer,
        )
    else:
        tb = run_track_b(
            features,
            artifacts,
            client,
            cfg,
            ml_backend=args.ml_backend,
            llm_model=args.model,
            skip_llm=skip_llm,
        )

    merged = merge_results(
        flow_id=str(flow_id) if flow_id is not None else None,
        classification_mode=mode,
        zs=zs,
        tb=tb,
        config=cfg,
        raw_features=features,
    )
    return merged.to_json_dict(), 0


def cmd_classify(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.is_file():
        console.print(
            f"[red]Input file not found:[/red] {path.resolve()}\n"
            "Create a valid template from your trained feature list:\n"
            "  [cyan]python cli.py sample-flow -o flow.json[/cyan]"
        )
        return 2

    out, code = classify_flow_from_args(args, path)
    if code != 0 or out is None:
        return code
    text = json.dumps(out, indent=2)
    if args.pretty:
        console.print_json(data=out)
    else:
        print(text)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    mode: ClassifyMode = args.mode
    need_key = mode == "zeroshot"
    cfg = load_config(require_api_key=need_key, env_file=args.env_file)
    cfg.classification_mode = mode
    if args.model:
        cfg.default_model = args.model
    if args.ml_backend:
        cfg.ml_classifier = args.ml_backend.lower()  # type: ignore[assignment]
    if args.models_dir:
        cfg.models_dir = Path(args.models_dir)

    client: Optional[OpenRouterClient] = (
        OpenRouterClient(cfg) if cfg.openrouter_api_key else None
    )
    if need_key and client is None:
        console.print("[red]OPENROUTER_API_KEY required for zeroshot evaluation.[/red]")
        return 2

    use_llm_explainer = mode == "ml_agent" and not getattr(args, "no_llm_explainer", False)
    summary = evaluate_csv(
        Path(args.dataset),
        cfg,
        client,
        max_rows=args.max_rows,
        random_state=args.seed,
        classification_mode=mode,
        ml_backend=args.ml_backend,
        llm_model=args.model,
        llm_explainer=use_llm_explainer,
        explain_max_rows=int(getattr(args, "explain_max_rows", 25) or 0),
    )
    out_json = Path(args.output)
    out_md = out_json.with_suffix(".md")
    write_eval_artifacts(summary, out_json, out_md)
    console.print(f"[green]Wrote[/green] {out_json} and {out_md}")
    return 0


def cmd_sample_flow(args: argparse.Namespace) -> int:
    """Write a flow.json template with all feature columns from models/feature_cols.joblib."""
    models_dir = Path(args.models_dir or "models")
    fpath = models_dir / "feature_cols.joblib"
    if not fpath.is_file():
        console.print(f"[red]Missing {fpath.resolve()}[/red] (train models first or set --models-dir)")
        return 2
    feature_cols: list[str] = list(joblib.load(fpath))
    features: Dict[str, float] = {str(c): 0.0 for c in feature_cols}

    if args.demo:
        # Non-zero toy values so the ML output is not purely "majority class on zeros"
        for key, val in (
            ("Flow Duration", 5000.0),
            ("Total Fwd Packets", 12.0),
            ("Total Backward Packets", 10.0),
            ("Flow Bytes/s", 1500.0),
            ("Flow Packets/s", 80.0),
            ("SYN Flag Count", 2.0),
        ):
            if key in features:
                features[key] = float(val)

    payload: Dict[str, Any] = {
        "flow_id": args.flow_id or "sample-flow",
        "features": features,
    }
    text = json.dumps(payload, indent=2)
    out = args.output
    if out:
        outp = Path(out)
        outp.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {outp.resolve()} ({len(feature_cols)} features)")
    else:
        print(text)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train RF + XGBoost from the four class CSVs (same pipeline as ml_classifier.ipynb)."""
    from llm_agent.train_ml import train_and_save_artifacts

    data_dir = Path(args.data_dir)
    required = (
        "Benign Traffic.csv",
        "DDoS UDP Flood.csv",
        "DoS TCP Flood.csv",
        "Recon Port Scan.csv",
    )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        console.print(
            f"[red]Missing CSV(s) in {data_dir.resolve()}:[/red]\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nPlace the four CIC IoT CSVs here (see README data layout)."
        )
        return 2
    if args.skip_rf and args.skip_xgb:
        console.print("[red]Cannot pass both --skip-rf and --skip-xgb.[/red]")
        return 2

    try:
        result = train_and_save_artifacts(
            data_dir=data_dir,
            output_dir=Path(args.output_dir),
            random_state=args.seed,
            sgkf_n_splits=args.sgkf_splits,
            dos_tcp_nrows=args.dos_tcp_rows,
            train_rf=not args.skip_rf,
            train_xgb=not args.skip_xgb,
            save_plots=not args.no_plots,
            shap_subsample=args.shap_subsample,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    except Exception:
        console.print("[red]Training failed:[/red]")
        console.print_exception()
        return 1

    console.print("[green]Training complete.[/green]")
    console.print(f"  Train rows: {result.n_train}  Test rows: {result.n_test}")
    if not args.skip_rf:
        console.print(
            f"  RF  — accuracy: {result.rf_accuracy:.4f}  macro-F1: {result.rf_macro_f1:.4f}"
        )
    if not args.skip_xgb:
        console.print(
            f"  XGB — accuracy: {result.xgb_accuracy:.4f}  macro-F1: {result.xgb_macro_f1:.4f}"
        )
    console.print(f"  Artifacts: {result.output_dir}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    cfg = load_config(env_file=args.env_file)
    client = OpenRouterClient(cfg)
    models = client.list_models()
    needle = (args.filter or "").lower()
    for m in models:
        if not needle or needle in m.lower():
            print(m)
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """
    Run an end-to-end chain: optional train, optional single-flow classify, optional CSV evaluate.

    The classify step emits a merged JSON that includes `why`: structured rationale (flow vs packet,
    SHAP evidence, LLM steps) for why the flow received its label(s).
    """
    from llm_agent.train_ml import train_and_save_artifacts

    do_train = bool(getattr(args, "train", False))
    do_classify = bool(getattr(args, "input", None))
    do_eval = bool(getattr(args, "eval_dataset", None))

    if not (do_train or do_classify or do_eval):
        console.print(
            "[red]Nothing to run.[/red] Use at least one of:\n"
            "  [cyan]--train[/cyan]  [cyan]-i flow.json[/cyan]  [cyan]--eval-dataset path.csv[/cyan]"
        )
        return 2

    bundle: Dict[str, Any] = {"pipeline_version": "1", "steps_completed": []}
    n_steps = int(do_train) + int(do_classify) + int(do_eval)
    out_path = getattr(args, "pipeline_output", None)
    if out_path is None and n_steps > 1:
        out_path = "pipeline_report.json"

    if do_train:
        data_dir = Path(args.data_dir)
        required = (
            "Benign Traffic.csv",
            "DDoS UDP Flood.csv",
            "DoS TCP Flood.csv",
            "Recon Port Scan.csv",
        )
        missing = [name for name in required if not (data_dir / name).is_file()]
        if missing:
            console.print(
                f"[red]Train skipped — missing CSV(s) in {data_dir.resolve()}:[/red]\n"
                + "\n".join(f"  - {m}" for m in missing)
            )
            bundle["train"] = {"ok": False, "skipped": True, "missing": missing}
        elif args.skip_rf and args.skip_xgb:
            console.print("[red]Cannot pass both --skip-rf and --skip-xgb.[/red]")
            return 2
        else:
            try:
                result = train_and_save_artifacts(
                    data_dir=data_dir,
                    output_dir=Path(args.output_dir),
                    random_state=args.seed,
                    sgkf_n_splits=args.sgkf_splits,
                    dos_tcp_nrows=args.dos_tcp_rows,
                    train_rf=not args.skip_rf,
                    train_xgb=not args.skip_xgb,
                    save_plots=not args.no_plots,
                    shap_subsample=args.shap_subsample,
                )
            except Exception:
                console.print("[red]Training failed:[/red]")
                console.print_exception()
                return 1
            bundle["steps_completed"].append("train")
            bundle["train"] = {
                "ok": True,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "rf_accuracy": result.rf_accuracy,
                "rf_macro_f1": result.rf_macro_f1,
                "xgb_accuracy": result.xgb_accuracy,
                "xgb_macro_f1": result.xgb_macro_f1,
                "output_dir": result.output_dir,
            }
            console.print("[green]Pipeline:[/green] train finished.")

    if do_classify:
        path = Path(args.input)
        if not path.is_file():
            console.print(
                f"[red]Input file not found:[/red] {path.resolve()}\n"
                "  [cyan]python cli.py sample-flow -o flow.json[/cyan]"
            )
            return 2
        out, code = classify_flow_from_args(args, path)
        if code != 0 or out is None:
            return code
        bundle["steps_completed"].append("classify")
        bundle["classify"] = out
        console.print("[green]Pipeline:[/green] classify finished (see `why` in JSON for explanations).")

    if do_eval:
        mode_ev: ClassifyMode = getattr(args, "mode", "ml_agent")
        need_key = mode_ev == "zeroshot"
        cfg = load_config(require_api_key=need_key, env_file=args.env_file)
        cfg.classification_mode = mode_ev
        if args.model:
            cfg.default_model = args.model
        if args.ml_backend:
            cfg.ml_classifier = args.ml_backend.lower()  # type: ignore[assignment]
        if args.models_dir:
            cfg.models_dir = Path(args.models_dir)

        client = OpenRouterClient(cfg) if cfg.openrouter_api_key else None
        if need_key and client is None:
            console.print("[red]OPENROUTER_API_KEY required for zeroshot evaluation step.[/red]")
            return 2

        use_expl = mode_ev == "ml_agent" and not getattr(args, "no_llm_explainer", False)
        summary = evaluate_csv(
            Path(args.eval_dataset),
            cfg,
            client,
            max_rows=args.eval_max_rows,
            random_state=args.eval_seed,
            classification_mode=mode_ev,
            ml_backend=args.ml_backend,
            llm_model=args.model,
            llm_explainer=use_expl,
            explain_max_rows=int(getattr(args, "explain_max_rows", 25) or 0),
        )
        ev_out = Path(args.eval_output)
        write_eval_artifacts(summary, ev_out, ev_out.with_suffix(".md"))
        bundle["steps_completed"].append("evaluate")
        bundle["evaluate"] = {
            "report_json": str(ev_out.resolve()),
            "report_md": str(ev_out.with_suffix(".md").resolve()),
            "classification_mode": summary.classification_mode,
            "n_rows": summary.n_rows,
            "ml_accuracy": summary.ml_accuracy,
            "ml_macro_f1": summary.ml_macro_f1,
            "zeroshot_accuracy": summary.zeroshot_accuracy,
            "zeroshot_macro_f1": summary.zeroshot_macro_f1,
        }
        console.print(f"[green]Pipeline:[/green] evaluate wrote {ev_out.resolve()}")

    if out_path:
        outp = Path(out_path)
        outp.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote pipeline bundle[/green] {outp.resolve()}")

    if do_classify and n_steps == 1 and not out_path:
        print(json.dumps(bundle["classify"], indent=2))
    elif do_classify and getattr(args, "pretty", False) and bundle.get("classify"):
        console.print_json(data=bundle["classify"])

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-Augmented IoT traffic classification (CSE534 project CLI)"
    )
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional path to a .env file (defaults to auto-discovery via python-dotenv)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("classify", help="Classify a single flow from JSON")
    pc.add_argument("--input", "-i", required=True, help="Path to flow JSON")
    pc.add_argument(
        "--mode",
        type=_classification_mode_type,
        default="ml-agent",
        help="ml-agent (default): ML + SHAP + LLM explainer. zeroshot: LLM-only label + rationale.",
    )
    pc.add_argument(
        "--model",
        "-m",
        default=None,
        help="OpenRouter model id (default: config / OPENROUTER_DEFAULT_MODEL, e.g. tencent/hy3-preview:free)",
    )
    pc.add_argument(
        "--ml-backend",
        choices=("rf", "xgb", "xgboost"),
        default=None,
        help="Random Forest or XGBoost for ml-agent mode (default: env or rf)",
    )
    pc.add_argument(
        "--serializer",
        choices=("verbose", "compact"),
        default="verbose",
        help="Feature serialization detail for zeroshot LLM prompts",
    )
    pc.add_argument(
        "--skip-llm",
        action="store_true",
        help="ml-agent: ML + SHAP + rule-based text only (no OpenRouter explainer call)",
    )
    pc.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory with joblib/json models (default: ./models)",
    )
    pc.add_argument(
        "--prompts-dir",
        type=str,
        default=None,
        help="Directory with prompt templates (default: ./prompts)",
    )
    pc.add_argument(
        "--pretty",
        action="store_true",
        help="Rich-formatted panel output",
    )
    _add_openrouter_log_flags(pc)
    pc.set_defaults(func=cmd_classify)

    pe = sub.add_parser(
        "evaluate",
        help="Evaluate ml-agent (tree + optional LLM explainer) or zeroshot (LLM per row) on a CSV",
        description=(
            "ml-agent: metrics on all rows; by default also runs the OpenRouter SHAP explainer on the "
            "first --explain-max-rows rows (needs OPENROUTER_API_KEY unless you pass --no-llm-explainer). "
            "Per-class row counts when building a CSV: use scripts/make_eval_sample.py --per-class N."
        ),
    )
    pe.add_argument("--dataset", "-d", required=True, help="CSV with Label + all feature columns")
    pe.add_argument("--output", "-o", default="eval_report.json", help="JSON output path")
    pe.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="After loading the CSV, randomly keep at most this many rows (not per-class)",
    )
    pe.add_argument(
        "--no-llm-explainer",
        action="store_true",
        help="ml-agent only: skip OpenRouter SHAP narrative (metrics unchanged; faster / no API key)",
    )
    pe.add_argument(
        "--explain-max-rows",
        type=int,
        default=25,
        help="ml-agent: how many leading rows get an LLM explainer after metrics (default 25)",
    )
    pe.add_argument("--seed", type=int, default=43, help="Subsampling random seed")
    pe.add_argument(
        "--mode",
        type=_classification_mode_type,
        default="ml-agent",
        help="ml-agent (default): ML vs labels + optional LLM explainer on first rows. zeroshot: LLM vs labels (needs API key).",
    )
    pe.add_argument(
        "--model",
        "-m",
        default=None,
        help="OpenRouter model id (zeroshot; also ml-agent LLM explainer when enabled)",
    )
    pe.add_argument(
        "--ml-backend",
        choices=("rf", "xgb", "xgboost"),
        default=None,
        help="ML backend for ml-agent mode",
    )
    pe.add_argument("--models-dir", type=str, default=None)
    _add_openrouter_log_flags(pe)
    pe.set_defaults(func=cmd_evaluate)

    pl = sub.add_parser("list-models", help="List OpenRouter models (requires API key)")
    pl.add_argument("--filter", "-f", default=None, help="Substring filter")
    _add_openrouter_log_flags(pl)
    pl.set_defaults(func=cmd_list_models)

    pt = sub.add_parser(
        "train",
        help="Train RF + XGBoost from data/*.csv (notebook-equivalent; writes models/)",
    )
    pt.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with Benign Traffic.csv, DDoS UDP Flood.csv, DoS TCP Flood.csv, Recon Port Scan.csv",
    )
    pt.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Where to write joblib/json, metrics JSON, and optional plots",
    )
    pt.add_argument(
        "--dos-tcp-rows",
        type=int,
        default=100_000,
        help="Max rows to read from DoS TCP Flood.csv (matches notebook cap)",
    )
    pt.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for StratifiedGroupKFold and estimators",
    )
    pt.add_argument(
        "--sgkf-splits",
        type=int,
        default=4,
        help="StratifiedGroupKFold n_splits (first fold used as held-out test)",
    )
    pt.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figures (confusion matrices, importances, SHAP bar)",
    )
    pt.add_argument(
        "--shap-subsample",
        type=int,
        default=2000,
        help="Max test rows for SHAP summary plot (XGBoost only)",
    )
    pt.add_argument("--skip-rf", action="store_true", help="Train XGBoost only")
    pt.add_argument("--skip-xgb", action="store_true", help="Train Random Forest only")
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser(
        "pipeline",
        help="Full chain: optional train, classify one flow (with `why` explanations), optional CSV eval",
    )
    pp.add_argument(
        "--train",
        action="store_true",
        help="Train RF+XGB from CSVs under --data-dir (skipped if required files missing)",
    )
    pp.add_argument("--data-dir", type=str, default="data", help="Training CSV directory")
    pp.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Model output directory for --train",
    )
    pp.add_argument("--dos-tcp-rows", type=int, default=100_000)
    pp.add_argument("--seed", type=int, default=43)
    pp.add_argument("--sgkf-splits", type=int, default=4)
    pp.add_argument("--no-plots", action="store_true")
    pp.add_argument("--shap-subsample", type=int, default=2000)
    pp.add_argument("--skip-rf", action="store_true")
    pp.add_argument("--skip-xgb", action="store_true")
    pp.add_argument(
        "--input",
        "-i",
        default=None,
        help="flow.json for classification + merged `why` explanation block",
    )
    pp.add_argument(
        "--eval-dataset",
        default=None,
        help="CSV with Label + features for the evaluate step",
    )
    pp.add_argument("--eval-output", default="eval_pipeline.json")
    pp.add_argument("--eval-max-rows", type=int, default=None)
    pp.add_argument("--eval-seed", type=int, default=43)
    pp.add_argument(
        "--no-llm-explainer",
        action="store_true",
        help="evaluate step (ml-agent): skip OpenRouter SHAP narrative",
    )
    pp.add_argument(
        "--explain-max-rows",
        type=int,
        default=25,
        help="evaluate step (ml-agent): max rows for LLM explainer after metrics",
    )
    pp.add_argument(
        "--pipeline-output",
        "-o",
        default=None,
        help="Write full pipeline JSON bundle (default if multiple steps: pipeline_report.json)",
    )
    pp.add_argument(
        "--mode",
        type=_classification_mode_type,
        default="ml-agent",
        help="Classify + evaluate steps: ml-agent (default) or zeroshot",
    )
    pp.add_argument("--model", "-m", default=None)
    pp.add_argument(
        "--ml-backend",
        choices=("rf", "xgb", "xgboost"),
        default=None,
    )
    pp.add_argument("--serializer", choices=("verbose", "compact"), default="verbose")
    pp.add_argument("--skip-llm", action="store_true")
    pp.add_argument("--models-dir", type=str, default=None)
    pp.add_argument("--prompts-dir", type=str, default=None)
    pp.add_argument("--pretty", action="store_true")
    _add_openrouter_log_flags(pp)
    pp.set_defaults(func=cmd_pipeline)

    ps = sub.add_parser(
        "sample-flow",
        help="Generate a flow JSON template (all columns from feature_cols.joblib)",
    )
    ps.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write to this path (default: print JSON to stdout)",
    )
    ps.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing feature_cols.joblib (default: ./models)",
    )
    ps.add_argument(
        "--flow-id",
        type=str,
        default=None,
        help="Optional flow_id field in the JSON",
    )
    ps.add_argument(
        "--demo",
        action="store_true",
        help="Fill a few common CICFlowMeter fields with small non-zero values",
    )
    ps.set_defaults(func=cmd_sample_flow)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_openrouter_logging(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
