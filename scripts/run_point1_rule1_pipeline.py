"""Run the Rule 1 small closed-loop pipeline on a frozen clean/rule1 subset."""

from __future__ import annotations

import json

from common.io.json_io import write_json
from eval.reports.point1_rule1_failures import export_rule1_failures
from point1.pipelines import run_rule1_pipeline
from point1.pipelines.rule1_runner_cli import (
    apply_run_name_defaults as _apply_run_name_defaults,
)
from point1.pipelines.rule1_runner_cli import (
    apply_target_preset as _apply_target_preset,
)
from point1.pipelines.rule1_runner_cli import (
    build_rule1_runner_parser as build_parser,
)
from point1.pipelines.rule1_runner_data import (
    build_rule1_summary as _build_summary,
)
from point1.pipelines.rule1_runner_data import (
    load_target_samples as _load_target_samples,
)
from point1.pipelines.rule1_runner_runtime import (
    build_rule1_runtime as _build_rule1_runtime,
)


def main() -> None:
    """Parse arguments, run the Rule 1 pipeline, and write output plus summary."""
    args = build_parser(description=__doc__).parse_args()
    _apply_target_preset(args)
    _apply_run_name_defaults(args)

    target_samples, summary_context = _load_target_samples(args)
    pipeline, provider_name, model_name, mode = _build_rule1_runtime(args)

    records = run_rule1_pipeline(
        target_samples=target_samples,
        pipeline=pipeline,
        provider_name=provider_name,
        model_name=model_name,
        mode=mode,
        show_progress=True,
        progress_output=args.progress_output,
        checkpoint_output=args.checkpoint_output,
        checkpoint_every=args.checkpoint_every,
        predicate_backend=args.predicate_backend,
        candidate_batch_size=args.candidate_batch_size,
        max_new_tokens=(args.max_new_tokens if args.predicate_backend == "local_qwen" else None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, [record.to_dict() for record in records])
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False, indent=2))

    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary = _build_summary(
        args=args,
        output_path=args.output,
        target_samples=target_samples,
        summary_context=summary_context,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_output, summary)
    print(json.dumps({"summary_output": str(summary_output)}, ensure_ascii=False, indent=2))
    if args.failure_output is not None:
        args.failure_output.parent.mkdir(parents=True, exist_ok=True)
        failure_export = export_rule1_failures(
            output_path=args.output,
            target_samples=target_samples,
        )
        write_json(args.failure_output, failure_export)
        print(
            json.dumps(
                {"failure_output": str(args.failure_output)},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
