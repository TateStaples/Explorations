# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
# ]
# ///
import marimo

__generated_with = "0.18.4"
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
    import torch
    import urllib.request
    import zipfile
    from pathlib import Path
    return alt, html, json, mo, np, pd, re, torch, urllib, zipfile, Path


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
    return api_url, app_url, data_dir, readme_url, space_id, space_url, tree_url


@app.cell
def _(api_url, app_url, html, json, readme_url, re, tree_url, urllib):
    def fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def fetch_json(url: str) -> dict | list:
        return json.loads(fetch_text(url))

    def strip_frontmatter(readme_text: str) -> str:
        frontmatter_match = re.match(r"^---\s*\n.*?\n---\s*\n?", readme_text, flags=re.DOTALL)
        if frontmatter_match:
            readme_text = readme_text[frontmatter_match.end() :]
        return readme_text.strip()

    try:
        api_metadata = fetch_json(api_url)
        tree_listing = fetch_json(tree_url)
        readme_text = fetch_text(readme_url)
        app_text = fetch_text(app_url)
        description_text = strip_frontmatter(readme_text)
        description_preview_match = re.search(
            r'<meta property="og:description" content="([^"]+)"',
            fetch_text(f"https://huggingface.co/spaces/{api_metadata['id']}"),
        )
        preview_description = (
            html.unescape(description_preview_match.group(1))
            if description_preview_match
            else "No preview description found."
        )
    except Exception as error:
        api_metadata = {}
        tree_listing = []
        readme_text = ""
        app_text = ""
        description_text = f"Unable to fetch source page in this environment: {error}"
        preview_description = ""
    return api_metadata, app_text, description_text, preview_description, tree_listing


@app.cell(hide_code=True)
def _(api_metadata, description_text, mo, preview_description, space_url):
    like_count = api_metadata.get("likes", "unknown")
    sdk = api_metadata.get("sdk", "unknown")
    last_modified = api_metadata.get("lastModified", "unknown")
    mo.md(
        f"""
        # Dropped a Neural Net: Problem Parsing and Data Exploration

        Source page: [{space_url}]({space_url})

        **Space metadata:** likes={like_count}, sdk={sdk}, last_modified={last_modified}

        ## Puzzle Description (from README)
        {description_text[:2500]}

        ## Hugging Face preview description
        {preview_description}
        """
    )
    return


@app.cell
def _(app_text, data_dir, pd, re, space_url, tree_listing, urllib):
    def discover_zip_urls(space_tree: list[dict], source_text: str) -> list[str]:
        tree_links = [
            f"{space_url}/resolve/main/{item['path']}"
            for item in space_tree
            if item.get("type") == "file" and str(item.get("path", "")).endswith(".zip")
        ]
        text_links = re.findall(r"https?://[^\s\"')]+\.zip", source_text)
        return sorted(set(tree_links + text_links))

    def download_zip(zip_url: str, output_dir: Path) -> Path:
        target_path = output_dir / zip_url.rsplit("/", 1)[-1].split("?", 1)[0]
        if not target_path.exists():
            urllib.request.urlretrieve(zip_url, target_path)
        return target_path

    zip_links = discover_zip_urls(tree_listing, app_text)
    selected_zip = zip_links[0] if zip_links else ""
    archive_path = download_zip(selected_zip, data_dir) if selected_zip else None
    archive_df = (
        pd.DataFrame({"zip_url": zip_links}) if zip_links else pd.DataFrame(columns=["zip_url"])
    )
    return archive_df, archive_path, selected_zip, zip_links


@app.cell(hide_code=True)
def _(archive_df, archive_path, mo, zip_links):
    if archive_path:
        zip_markdown = "\n".join(f"- [{url}]({url})" for url in zip_links)
        mo.md(
            f"### Downloaded archive\n`{archive_path}`\n\n### Zip links discovered\n{zip_markdown}"
        )
    else:
        mo.md("### Downloaded archive\nNo archive downloaded because no .zip link was detected.")
    archive_df
    return


@app.cell
def _(archive_path, pd, zipfile):
    def list_archive_contents(zip_path: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path) as archive:
            file_info = archive.infolist()
            return pd.DataFrame(
                {
                    "filename": [entry.filename for entry in file_info],
                    "size_bytes": [entry.file_size for entry in file_info],
                }
            )

    contents_df = (
        list_archive_contents(str(archive_path))
        if archive_path
        else pd.DataFrame(columns=["filename", "size_bytes"])
    )
    return (contents_df,)


@app.cell
def _(archive_path, contents_df, pd, zipfile):
    def read_historical_data(zip_path: str, files_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        csv_candidates = files_df[files_df["filename"].str.lower().str.endswith(".csv")]
        preferred_name = "historical_data.csv"
        if csv_candidates.empty:
            return "", pd.DataFrame()
        selected_name = (
            preferred_name
            if preferred_name in csv_candidates["filename"].tolist()
            else csv_candidates.iloc[0]["filename"]
        )
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(selected_name) as extracted:
                frame = pd.read_csv(extracted)
        return selected_name, frame

    source_file, historical_df = (
        read_historical_data(str(archive_path), contents_df)
        if archive_path
        else ("", pd.DataFrame())
    )
    return historical_df, source_file


@app.cell
def _(alt, historical_df, np, pd):
    SCATTER_SAMPLE_SIZE = 2500
    RANDOM_SEED = 42
    TOP_CORRELATIONS_COUNT = 10
    preview_df = historical_df.head(20)
    measurement_cols = [
        column for column in historical_df.columns if column.startswith("measurement_")
    ]
    schema_df = (
        pd.DataFrame(
            {
                "column": historical_df.columns,
                "dtype": [str(dtype) for dtype in historical_df.dtypes],
                "missing": historical_df.isna().sum().values,
            }
        )
        if not historical_df.empty
        else pd.DataFrame(columns=["column", "dtype", "missing"])
    )
    model_eval_df = pd.DataFrame()
    feature_corr_df = pd.DataFrame(columns=["feature", "corr_with_true"])
    hypothesis_df = pd.DataFrame(columns=["hypothesis", "evidence"])
    if not historical_df.empty and {"pred", "true"}.issubset(historical_df.columns):
        model_eval_df = historical_df[["pred", "true"]].copy()
        model_eval_df["error"] = model_eval_df["pred"] - model_eval_df["true"]
        model_eval_df["abs_error"] = model_eval_df["error"].abs()
        if measurement_cols:
            corr_series = historical_df[measurement_cols].corrwith(historical_df["true"])
            feature_corr_df = (
                corr_series.abs()
                .nlargest(TOP_CORRELATIONS_COUNT)
                .rename_axis("feature")
                .reset_index(name="corr_with_true")
            )
        mse = float(np.mean(model_eval_df["error"] ** 2))
        true_var = float(np.var(model_eval_df["true"]))
        true_std = float(np.std(model_eval_df["true"]))
        mae = float(model_eval_df["abs_error"].mean())
        pred_true_corr = float(model_eval_df["pred"].corr(model_eval_df["true"]))
        hypothesis_df = pd.DataFrame(
            [
                {
                    "hypothesis": "Provided `pred` approximates a strong baseline model output",
                    "evidence": f"corr(pred,true)={pred_true_corr:.3f}, MAE/σ(true)={mae / true_std:.3f}",
                },
                {
                    "hypothesis": "Residual task signal is mostly linear and captured by measurements",
                    "evidence": f"Top feature |corr| with true reaches {feature_corr_df['corr_with_true'].max():.3f}",
                },
                {
                    "hypothesis": "Error magnitude depends on target scale (heteroskedasticity check)",
                    "evidence": f"corr(|error|,|true|)={model_eval_df['abs_error'].corr(model_eval_df['true'].abs()):.3f}",
                },
                {
                    "hypothesis": "Puzzle architecture likely 48 residual blocks + one final projection",
                    "evidence": "Piece inventory contains 48 (96,48), 48 (48,96), and one (1,48) weight tensors.",
                },
                {
                    "hypothesis": "Model fit remains imperfect, leaving recoverable puzzle signal",
                    "evidence": f"Estimated R²={1 - (mse / true_var):.3f} on historical_data.csv",
                },
            ]
        )
    metrics_df = (
        pd.DataFrame(
            [
                {
                    "metric": "MAE",
                    "value": float(model_eval_df["abs_error"].mean()),
                },
                {
                    "metric": "RMSE",
                    "value": float(np.sqrt(np.mean(model_eval_df["error"] ** 2))),
                },
                {
                    "metric": "corr(pred,true)",
                    "value": float(model_eval_df["pred"].corr(model_eval_df["true"])),
                },
            ]
        )
        if not model_eval_df.empty
        else pd.DataFrame(columns=["metric", "value"])
    )
    missing_chart = (
        alt.Chart(schema_df)
        .mark_bar()
        .encode(
            x=alt.X("column:N", sort="-y", title="Column"),
            y=alt.Y("missing:Q", title="Missing Values"),
            tooltip=["column:N", "dtype:N", "missing:Q"],
        )
        .properties(title="Missing Values by Column")
        .interactive()
        if not schema_df.empty
        else None
    )
    error_hist = (
        alt.Chart(model_eval_df)
        .mark_bar()
        .encode(
            x=alt.X("error:Q", bin=alt.Bin(maxbins=60), title="Prediction Error (pred - true)"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=["count():Q"],
        )
        .properties(title="Error Distribution")
        .interactive()
        if not model_eval_df.empty
        else None
    )
    pred_true_scatter = (
        alt.Chart(
            model_eval_df.sample(
                min(len(model_eval_df), SCATTER_SAMPLE_SIZE),
                random_state=RANDOM_SEED,
            )
        )
        .mark_point(opacity=0.35)
        .encode(
            x=alt.X("true:Q", title="true"),
            y=alt.Y("pred:Q", title="pred"),
            tooltip=["true:Q", "pred:Q", "error:Q"],
        )
        .properties(title="Pred vs True (sample)")
        .interactive()
        if not model_eval_df.empty
        else None
    )
    top_feature_chart = (
        alt.Chart(feature_corr_df)
        .mark_bar()
        .encode(
            x=alt.X("corr_with_true:Q", title="|Correlation| with true"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature:N", "corr_with_true:Q"],
        )
        .properties(title="Top Measurement Correlations with true")
        .interactive()
        if not feature_corr_df.empty
        else None
    )
    return (
        error_hist,
        feature_corr_df,
        hypothesis_df,
        measurement_cols,
        metrics_df,
        missing_chart,
        model_eval_df,
        pred_true_scatter,
        preview_df,
        schema_df,
        top_feature_chart,
    )


@app.cell(hide_code=True)
def _(archive_path, contents_df, historical_df, mo, source_file):
    _piece_file_prefix = "pieces/piece_"
    if historical_df.empty:
        _display_output = mo.md("No tabular files were found in the archive for preliminary analysis.")
    else:
        piece_count = int(contents_df["filename"].str.startswith(_piece_file_prefix).sum())
        _display_output = mo.vstack(
            [
                mo.md(f"## Preliminary Data Analysis\nAnalyzed file: `{source_file}`"),
                mo.md(
                    f"Archive path: `{archive_path}`\n\n"
                    f"Rows: **{historical_df.shape[0]}** | Columns: **{historical_df.shape[1]}** | "
                    f"Piece files: **{piece_count}**"
                ),
                mo.md("### Archive contents"),
                contents_df,
            ]
        )
    _display_output
    return


@app.cell
def _(archive_path, contents_df, pd, re, torch, zipfile):
    _piece_file_prefix = "pieces/piece_"
    if archive_path is None:
        piece_df = pd.DataFrame(columns=["piece_index", "weight_shape", "bias_shape", "file_size"])
    else:
        piece_rows = []
        with zipfile.ZipFile(str(archive_path)) as archive:
            for _, row in contents_df.iterrows():
                filename = row["filename"]
                if not filename.startswith(_piece_file_prefix):
                    continue
                match = re.search(r"piece_(\d+)\.pth$", filename)
                if not match:
                    continue
                with archive.open(filename) as extracted:
                    state = torch.load(
                        extracted,
                        map_location="cpu",
                        weights_only=True,
                    )
                weight = state["weight"]
                bias = state["bias"]
                assert weight.dtype == torch.float32, f"Unexpected weight dtype: {weight.dtype}"
                assert bias.dtype == torch.float32, f"Unexpected bias dtype: {bias.dtype}"
                assert weight.ndim == 2, f"Unexpected weight ndim: {weight.ndim}"
                assert bias.ndim == 1, f"Unexpected bias ndim: {bias.ndim}"
                piece_rows.append(
                    {
                        "piece_index": int(match.group(1)),
                        "weight_shape": str(tuple(weight.shape)),
                        "bias_shape": str(tuple(bias.shape)),
                        "file_size": int(row["size_bytes"]),
                    }
                )
        piece_df = pd.DataFrame(piece_rows).sort_values("piece_index").reset_index(drop=True)
    piece_type_summary = (
        piece_df.groupby(["weight_shape", "bias_shape"], as_index=False)
        .agg(count=("piece_index", "count"))
        .sort_values("count", ascending=False)
        if not piece_df.empty
        else pd.DataFrame(columns=["weight_shape", "bias_shape", "count"])
    )
    return piece_df, piece_type_summary


@app.cell
def _(alt, piece_type_summary):
    piece_type_chart = (
        alt.Chart(piece_type_summary)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("weight_shape:N", sort="-x", title="Weight shape"),
            color=alt.Color("bias_shape:N", title="Bias shape"),
            tooltip=["weight_shape:N", "bias_shape:N", "count:Q"],
        )
        .properties(title="Model Piece Type Counts")
        .interactive()
        if not piece_type_summary.empty
        else None
    )
    return (piece_type_chart,)


@app.cell(hide_code=True)
def _(
    error_hist,
    feature_corr_df,
    hypothesis_df,
    measurement_cols,
    metrics_df,
    missing_chart,
    mo,
    piece_df,
    piece_type_chart,
    piece_type_summary,
    pred_true_scatter,
    preview_df,
    schema_df,
    top_feature_chart,
):
    if preview_df.empty:
        _custom_output = mo.md("Data preview unavailable.")
    else:
        _custom_output = mo.vstack(
            [
                mo.md(
                    f"## Customized exploration for this puzzle data\n"
                    f"- {len(measurement_cols)} measurement features\n"
                    f"- {len(piece_df)} model pieces in archive\n"
                    f"- Piece shape mix is summarized in the inventory section below"
                ),
                mo.md("### Model quality summary from provided `pred` and `true`"),
                metrics_df,
                pred_true_scatter,
                error_hist,
                mo.md("### Top correlated measurements with `true`"),
                feature_corr_df,
                top_feature_chart,
                mo.md("### Working hypotheses from this dataset"),
                hypothesis_df,
                mo.md("### Schema and missingness checks"),
                schema_df,
                missing_chart,
                mo.md("### Piece inventory"),
                piece_type_summary,
                piece_type_chart,
                mo.md("### Piece index preview"),
                piece_df.head(20),
                mo.md(f"### Raw data preview ({len(measurement_cols)} measurement columns)"),
                preview_df,
            ]
        )
    _custom_output
    return


if __name__ == "__main__":
    app.run()
