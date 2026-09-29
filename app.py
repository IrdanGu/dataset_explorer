from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, request, render_template, send_file

from engine.loader import load_csv, Dataset
from engine.statistics import compute_statistics
from engine.correlations import correlation_matrix
from engine.transforms import (
    log_transform,
    normalize_min_max,
    standardize_zscore,
    one_hot_encode,
    fill_missing,
    drop_missing,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.errorhandler(RuntimeError)
def handle_runtime_error(e):
    return jsonify({"error": str(e)}), 400


UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_current_dataset: Dataset | None = None
_original_dataset: Dataset | None = None


def _ds() -> Dataset:
    if _current_dataset is None:
        raise RuntimeError("No dataset loaded")
    return _current_dataset


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    global _current_dataset, _original_dataset

    f = request.files.get("file")
    if f is None or f.filename == "":
        return jsonify({"error": "No file provided"}), 400

    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files supported"}), 400

    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))

    try:
        _current_dataset = load_csv(save_path, file_name=f.filename)
        _original_dataset = _current_dataset.clone()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_dataset_summary(_current_dataset))


@app.route("/api/summary")
def summary():
    return jsonify(_dataset_summary(_ds()))


@app.route("/api/statistics")
def statistics():
    ds = _ds()
    col_name = request.args.get("column")
    if col_name:
        col = ds.get_column(col_name)
        if col is None:
            return jsonify({"error": f"Column '{col_name}' not found"}), 404
        return jsonify(compute_statistics(col).to_dict())

    stats = [compute_statistics(c).to_dict() for c in ds.columns]
    return jsonify(stats)


@app.route("/api/correlation")
def correlation():
    return jsonify(correlation_matrix(_ds()))


@app.route("/api/distribution")
def distribution():
    ds = _ds()
    col_name = request.args.get("column")
    if not col_name:
        return jsonify({"error": "column parameter required"}), 400
    col = ds.get_column(col_name)
    if col is None:
        return jsonify({"error": f"Column '{col_name}' not found"}), 404
    stats = compute_statistics(col)
    return jsonify(stats.to_dict())


@app.route("/api/preview")
def preview():
    ds = _ds()
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = int(request.args.get("offset", 0))

    rows = []
    end = min(offset + limit, ds.row_count)
    for i in range(offset, end):
        row = {}
        for col in ds.columns:
            row[col.name] = col.values[i]
        rows.append(row)

    return jsonify(
        {
            "columns": ds.column_names(),
            "rows": rows,
            "total": ds.row_count,
            "offset": offset,
            "limit": limit,
        }
    )


@app.route("/api/transform", methods=["POST"])
def transform():
    global _current_dataset

    ds = _ds()
    body = request.get_json(force=True)
    action = body.get("action")
    col = body.get("column", "")

    try:
        if action == "log":
            base = body.get("base", "ln")
            _current_dataset = log_transform(ds, col, base)
        elif action == "normalize":
            _current_dataset = normalize_min_max(ds, col)
        elif action == "zscore":
            _current_dataset = standardize_zscore(ds, col)
        elif action == "onehot":
            _current_dataset = one_hot_encode(ds, col)
        elif action == "fill_missing":
            strategy = body.get("strategy", "mean")
            _current_dataset = fill_missing(ds, col, strategy)
        elif action == "drop_missing":
            target = col if col else None
            _current_dataset = drop_missing(ds, target)
        elif action == "reset":
            if _original_dataset is None:
                return jsonify({"error": "No original dataset to reset to"}), 400
            _current_dataset = _original_dataset.clone()
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(_dataset_summary(_current_dataset))


@app.route("/api/export")
def export():
    ds = _ds()
    export_path = UPLOAD_DIR / f"cleaned_{ds.file_name}"
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        f.write(ds.to_csv_string())
    return send_file(
        str(export_path), as_attachment=True, download_name=f"cleaned_{ds.file_name}"
    )


def _dataset_summary(ds: Dataset) -> dict:
    numeric = ds.numeric_columns()
    categorical = ds.categorical_columns()
    return {
        "file_name": ds.file_name,
        "row_count": ds.row_count,
        "col_count": len(ds.columns),
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "missing": c.missing_count,
                "missing_pct": round(c.missing_count / ds.row_count * 100, 1)
                if ds.row_count > 0
                else 0,
            }
            for c in ds.columns
        ],
        "numeric_columns": [c.name for c in numeric],
        "categorical_columns": [c.name for c in categorical],
        "total_missing": sum(c.missing_count for c in ds.columns),
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
