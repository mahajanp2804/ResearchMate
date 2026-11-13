"""Run an evaluation from the command line and print the results.

By default it uses a fresh temporary data directory, so your own library is
not touched: the sample papers are installed and indexed with the real
models, then every retrieval method is evaluated on the bundled dataset.

    cd backend
    .venv/bin/python -m scripts.run_evaluation            # retrieval only
    .venv/bin/python -m scripts.run_evaluation --answers  # also LLM-judged answers
    .venv/bin/python -m scripts.run_evaluation --json results.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import EvaluationRun
from app.services.evaluation_data import load_dataset_spec
from app.services.evaluation_service import CUTOFFS

METHOD_LABELS = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank",
}


def print_table(run: EvaluationRun) -> None:
    retrieval = run.metrics["retrieval"]
    columns = [f"P@{k}" for k in (1, 5)] + [f"R@{k}" for k in CUTOFFS] + ["MRR", "ms/query"]
    print(f"\nRetrieval ({retrieval[run.config['methods'][0]]['questions']} answerable questions)")
    print(f"{'Method':<18}" + "".join(f"{column:>9}" for column in columns))
    for method in run.config["methods"]:
        summary = retrieval[method]
        values = (
            [summary[f"precision@{k}"] for k in (1, 5)]
            + [summary[f"recall@{k}"] for k in CUTOFFS]
            + [summary["mrr"]]
        )
        cells = "".join(f"{value:>9.3f}" for value in values)
        print(f"{METHOD_LABELS[method]:<18}{cells}{summary['mean_latency_ms']:>9.1f}")

    print("\nMRR by question type")
    categories = sorted(run.metrics["retrieval_by_category"][run.config["methods"][0]])
    print(f"{'Method':<18}" + "".join(f"{c:>18}" for c in categories))
    for method in run.config["methods"]:
        by_category = run.metrics["retrieval_by_category"][method]
        cells = "".join(
            f"{by_category[c]['mrr']:>12.3f} (n={by_category[c]['questions']})" for c in categories
        )
        print(f"{METHOD_LABELS[method]:<18}{cells}")

    answers = run.metrics.get("answers")
    if answers:
        print("\nAnswer quality (LLM-judged)")
        for name, value in answers["summary"].items():
            print(f"  {name:<28}{value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="sample_papers")
    parser.add_argument("--data-dir", type=Path, help="Defaults to a new temporary directory")
    parser.add_argument("--answers", action="store_true", help="Also judge answers (needs a key)")
    parser.add_argument("--json", type=Path, help="Write the full run to this file")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        data_dir = args.data_dir or Path(temporary) / "data"
        app = create_app(Settings(data_dir=data_dir))
        service = app.state.evaluation
        with TestClient(app):  # runs startup: tables, reconciliation
            with app.state.database.session() as db:
                _, to_index = service.install_sample_papers(db)
            for document_id in to_index:
                app.state.ingestion.index_document(document_id)
            spec = load_dataset_spec(args.dataset)
            with app.state.database.session() as db:
                run_id = service.start_run(db, spec, answer_quality=args.answers).id
            service.execute(run_id)
            with app.state.database.session() as db:
                run = db.get(EvaluationRun, run_id)
                if run.status.value != "completed":
                    print(f"Evaluation {run.status.value}: {run.error_message}", file=sys.stderr)
                print_table(run)
                if args.json:
                    args.json.write_text(
                        json.dumps(
                            {
                                "config": run.config,
                                "metrics": run.metrics,
                                "per_question_results": run.per_question_results,
                            },
                            indent=2,
                        )
                    )
                    print(f"\nFull results written to {args.json}")
                return 0 if run.status.value == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
