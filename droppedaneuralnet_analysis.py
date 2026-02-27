import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import html
    import json
    import marimo as mo
    import pandas as pd
    import re
    import urllib.request
    import zipfile
    from pathlib import Path
    return alt, html, json, mo, pd, re, urllib, zipfile, Path


@app.cell
def _(Path):
    space_url = "https://huggingface.co/spaces/jane-street/droppedaneuralnet"
    data_dir = Path("data") / "droppedaneuralnet"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, space_url


@app.cell
def _(html, json, re, space_url, urllib):
    def fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    def extract_description(page_html: str) -> str:
        patterns = [
            r'<meta property="og:description" content="([^"]+)"',
            r'<meta name="description" content="([^"]+)"',
            r'"description":"((?:\\.|[^"])*)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, page_html)
            if match:
                raw_value = match.group(1)
                if raw_value:
                    if "\\u" in raw_value or '\\"' in raw_value:
                        raw_value = json.loads(f'"{raw_value}"')
                    return html.unescape(raw_value).strip()
        return "Unable to automatically extract a description from the page HTML."

    def extract_zip_links(page_html: str) -> list[str]:
        matches = re.findall(r"https?://[^\s\"'>]+\.zip", page_html)
        deduped_matches = sorted(set(matches))
        return deduped_matches

    try:
        page_html = fetch_text(space_url)
        description_text = extract_description(page_html)
        zip_links = extract_zip_links(page_html)
    except Exception as error:
        page_html = ""
        description_text = f"Unable to fetch source page in this environment: {error}"
        zip_links = []
    return description_text, page_html, zip_links


@app.cell(hide_code=True)
def _(description_text, mo, space_url, zip_links):
    zip_markdown = "\n".join(f"- [{url}]({url})" for url in zip_links) or "- No .zip links found in page HTML"
    mo.md(
        f"""
        # Dropped a Neural Net: Problem Parsing and Data Exploration

        Source page: [{space_url}]({space_url})

        ## Parsed Problem Description
        {description_text}

        ## Zip Files Found in Source HTML
        {zip_markdown}
        """
    )
    return


@app.cell
def _(Path, data_dir, pd, zip_links, urllib):
    def download_zip(zip_url: str, output_dir: Path) -> str:
        target_path = output_dir / zip_url.rsplit("/", 1)[-1].split("?", 1)[0]
        if not target_path.exists():
            urllib.request.urlretrieve(zip_url, target_path)
        return str(target_path)

    selected_zip = zip_links[0] if zip_links else ""
    archive_path = download_zip(selected_zip, data_dir) if selected_zip else ""
    archive_df = pd.DataFrame({"zip_url": zip_links}) if zip_links else pd.DataFrame(columns=["zip_url"])
    return archive_df, archive_path, selected_zip


@app.cell(hide_code=True)
def _(archive_df, archive_path, mo):
    if archive_path:
        mo.md(f"### Downloaded archive\n`{archive_path}`")
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

    contents_df = list_archive_contents(archive_path) if archive_path else pd.DataFrame(columns=["filename", "size_bytes"])
    return (contents_df,)


@app.cell
def _(archive_path, contents_df, pd, zipfile):
    def read_first_tabular_file(zip_path: str, files_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        tabular_suffixes = (".csv", ".parquet", ".json")
        candidates = files_df[files_df["filename"].str.lower().str.endswith(tabular_suffixes)]
        if candidates.empty:
            return "", pd.DataFrame()

        first_name = candidates.iloc[0]["filename"]
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(first_name) as extracted:
                if first_name.lower().endswith(".csv"):
                    frame = pd.read_csv(extracted)
                elif first_name.lower().endswith(".json"):
                    frame = pd.read_json(extracted)
                else:
                    frame = pd.read_parquet(extracted)
        return first_name, frame

    source_file, sample_df = read_first_tabular_file(archive_path, contents_df) if archive_path else ("", pd.DataFrame())
    return sample_df, source_file


@app.cell
def _(alt, pd, sample_df):
    preview_df = sample_df.head(20)
    schema_df = (
        pd.DataFrame({"column": sample_df.columns, "dtype": [str(dtype) for dtype in sample_df.dtypes], "missing": sample_df.isna().sum().values})
        if not sample_df.empty
        else pd.DataFrame(columns=["column", "dtype", "missing"])
    )
    numeric_summary = sample_df.describe().T if not sample_df.empty else pd.DataFrame()
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
    return missing_chart, numeric_summary, preview_df, schema_df


@app.cell(hide_code=True)
def _(contents_df, missing_chart, mo, numeric_summary, preview_df, sample_df, schema_df, source_file):
    if sample_df.empty:
        display_output = mo.md("No tabular files were found in the archive for preliminary analysis.")
    else:
        display_output = mo.vstack(
            [
                mo.md(f"## Preliminary Data Analysis\nAnalyzed file: `{source_file}`"),
                mo.md(f"Rows: **{sample_df.shape[0]}** | Columns: **{sample_df.shape[1]}**"),
                mo.md("### Archive contents"),
                contents_df,
                mo.md("### Data preview"),
                preview_df,
                mo.md("### Schema and missingness"),
                schema_df,
                mo.md("### Numeric summary statistics"),
                numeric_summary,
                missing_chart,
            ]
        )
    display_output
    return


if __name__ == "__main__":
    app.run()
