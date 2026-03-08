import json


def create_marimo_notebook(json_path, out_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(out_path, "w") as out:
        out.write("import marimo\n\n")
        out.write('__generated_with = "0.1.0"\n')
        out.write("app = marimo.App()\n\n")

        out.write("@app.cell\n")
        out.write("def imports():\n")
        out.write("    import marimo as mo\n")
        out.write("    import numpy as np\n")
        out.write("    import altair as alt\n")
        out.write("    import pandas as pd\n")
        out.write("    return mo, np, alt, pd\n\n")

        for page in data:
            page_text = page["text"]

            # Escape strings for python multiline string
            safe_text = page_text.replace("\\", "\\\\").replace('"', '\\"')

            # find if there is a Figure mention
            lines = page_text.split("\n")

            text_chunk = []

            for line in lines:
                if line.startswith("Figure "):
                    # write out the accumulated text chunk
                    if text_chunk:
                        chunk_str = (
                            "\\n".join(text_chunk)
                            .replace("\\", "\\\\")
                            .replace('"', '\\"')
                        )
                        out.write("@app.cell\n")
                        out.write("def _(mo):\n")
                        out.write(f'    mo.md("""{chunk_str}""")\n')
                        out.write("    return\n\n")
                        text_chunk = []

                    # write out a placeholder for the figure
                    safe_caption = line.replace("\\", "\\\\").replace('"', '\\"')
                    fig_num = line.split(":")[0]
                    out.write("@app.cell\n")
                    out.write("def _(mo, alt, pd, np):\n")
                    out.write(f'    mo.md("Interactive {fig_num}")\n')
                    out.write(f"    # {safe_caption}\n")
                    out.write("    # TODO: Implement interactive plot\n")
                    out.write("    return\n\n")
                else:
                    text_chunk.append(line)

            if text_chunk:
                chunk_str = (
                    "\\n".join(text_chunk).replace("\\", "\\\\").replace('"', '\\"')
                )
                out.write("@app.cell\n")
                out.write("def _(mo):\n")
                out.write(f'    mo.md("""{chunk_str}""")\n')
                out.write("    return\n\n")

    print(f"Created {out_path}")


create_marimo_notebook("extracted_text.json", "stat_mech.py")
