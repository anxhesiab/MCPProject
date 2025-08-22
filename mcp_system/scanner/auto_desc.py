def auto_describe(path: str, raw_bytes: bytes) -> str:
    """
    Build a one-line description with no hard-coded domain guesses.
    Example output:
    car_inventory.csv · PK=vin · FK=model_id,dealership_id · in_stock_date 2024-01-01→2025-07-30
    """
    import io, pandas as pd, re
    # ---------- sample ----------
    if path.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw_bytes), nrows=5000)
    else:  # parquet / others
        df = pd.read_parquet(io.BytesIO(raw_bytes))

    cols = list(df.columns)

    # ---------- primary / foreign keys ----------
    pk = next((c for c in cols if re.search(r"(^|_)id$", c, re.I) and df[c].is_unique), cols[0])
    fks = [c for c in cols if re.search(r"(^|_)id$", c, re.I) and c != pk]

    # ---------- date ranges ----------
    date_cols, ranges = [], []
    for c in cols:
        if re.search(r"date|_dt$", c, re.I):
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                ranges.append(f"{c} {s.min().date()}→{s.max().date()}")

    # ---------- compose ----------
    bits = [f"PK={pk}"]
    if fks:
        bits.append("FK=" + ",".join(fks))
    bits.extend(ranges)
    return " · ".join(bits) or "Table"
