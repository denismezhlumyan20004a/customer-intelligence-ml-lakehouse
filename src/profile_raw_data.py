from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw/aqua/2022")


for file_path in DATA_DIR.glob("*.csv"):
    print("\n" + "=" * 80)
    print(f"ARCHIVO: {file_path.name}")
    print("=" * 80)

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str
    )

    print(f"Filas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")
    print("\nNombres de columnas:")
    print(df.columns.tolist())

    print("\nPrimeras 10 filas:")
    print(df.head(10).to_string())

    print("\nValores nulos por columna:")
    print(df.isna().sum())

    print("\nDuplicados exactos:")
    print(df.duplicated().sum())