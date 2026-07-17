"""Create an exact unweighted descriptive-statistics table for one census."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(1991, 2000, 2010), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/processed/censo_microdados/characteristics_panels"),
    )
    return parser.parse_args()


def scalar(value: pa.Scalar | pa.Array | pa.ChunkedArray) -> object:
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        value = value[0]
    return value.as_py() if value.is_valid else None


def main() -> None:
    args = parse_args()
    source = args.root / f"persons_{args.year}.parquet"
    parquet = pq.ParquetFile(source)
    rows: list[dict[str, object]] = []

    for field in parquet.schema_arrow:
        data = pq.read_table(source, columns=[field.name]).column(0)
        if pa.types.is_string(field.type):
            substantive = pc.and_(
                pc.is_valid(data), pc.not_equal(pc.utf8_trim_whitespace(data), "")
            )
            substantive = pc.fill_null(substantive, False)
            nonmissing = int(scalar(pc.sum(pc.cast(substantive, pa.int64()))))
        else:
            nonmissing = int(scalar(pc.count(data)))
        result: dict[str, object] = {
            "variable": field.name,
            "type": str(field.type),
            "nonmissing_n": nonmissing,
            "missing_n": parquet.metadata.num_rows - nonmissing,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
        }
        if pa.types.is_boolean(field.type) or pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric = pc.cast(data, pa.float64())
            extrema = scalar(pc.min_max(numeric))
            result.update(
                mean=scalar(pc.mean(numeric)),
                median=scalar(pc.quantile(numeric, q=0.5, interpolation="linear")),
                std_dev=scalar(pc.stddev(numeric, ddof=1)),
                min=extrema["min"] if extrema is not None else None,
                max=extrema["max"] if extrema is not None else None,
            )
        rows.append(result)

    frame = pd.DataFrame(rows)
    csv_path = args.root / f"descriptive_statistics_{args.year}.csv"
    markdown_path = args.root / f"descriptive_statistics_{args.year}.md"
    frame.to_csv(csv_path, index=False, float_format="%.6g")

    display = frame.copy()
    for column in ("mean", "median", "std_dev", "min", "max"):
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:,.6g}"
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    markdown_path.write_text("\n".join([header, separator, *body]) + "\n", encoding="utf-8")
    print(markdown_path)
    print(csv_path)


if __name__ == "__main__":
    main()
