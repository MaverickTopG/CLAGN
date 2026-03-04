from __future__ import annotations

import pandas as pd


def df_to_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return '(none)'
    cols = [str(c) for c in df.columns]
    out = []
    out.append('| ' + ' | '.join(cols) + ' |')
    out.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append('')
            else:
                vals.append(str(v))
        out.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(out)
