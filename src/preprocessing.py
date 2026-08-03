"""
src/preprocessing.py
====================
Future-proof, schema-adaptive data engineering pipeline.
Optimized for 2M+ rows using vectorized Pandas operations.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION LOADER & VALIDATOR
# =============================================================================
def load_and_validate_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        ("data", "raw_data_path"), ("data", "vlookup_path"), ("data", "processed_dir"),
        ("data", "grid_size_deg"), ("data", "lat_min"), ("data", "lon_min"),
        ("data", "hours_per_bin"), ("data", "hist_steps"), ("data", "forecast_steps"),
        ("data", "geolocation_column"), ("data", "description_column"),
        ("data", "date_column"), ("data", "time_column"), ("data", "vlookup_sheet_name"),
        ("data", "vlookup_skip_patterns"), ("data", "vlookup_header_markers"),
        ("data", "fallback_category_name"), ("data", "admin_columns"),
        ("data", "threat_columns"), ("data", "lat_bounds"), ("data", "lon_bounds"),
    ]

    missing = [f"{s}.{k}" for s, k in required_keys if s not in cfg or k not in cfg[s]]
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}:\n{missing}")
    
    logger.info(f"Config loaded and validated: {config_path}")
    return cfg

# =============================================================================
# 2. SAFE PARSERS & LOOKUP BUILDERS
# =============================================================================
def parse_geolocation_safe(geoloc_str, lat_bounds, lon_bounds) -> Tuple[float, float]:
    if pd.isna(geoloc_str) or not isinstance(geoloc_str, str):
        return (0.0, 0.0)
    try:
        coords = ast.literal_eval(geoloc_str.strip())
        if isinstance(coords, (tuple, list)) and len(coords) == 2:
            lat, lon = float(coords[0]), float(coords[1])
            if lat_bounds[0] <= lat <= lat_bounds[1] and lon_bounds[0] <= lon <= lon_bounds[1]:
                return (lat, lon)
    except (ValueError, SyntaxError, TypeError):
        pass
    return (0.0, 0.0)

def build_violation_lookup(cfg: dict) -> Tuple[Dict[str, int], int]:
    data_cfg = cfg["data"]
    xl = pd.ExcelFile(data_cfg["vlookup_path"])
    if data_cfg["vlookup_sheet_name"] not in xl.sheet_names:
        raise ValueError(f"Sheet '{data_cfg['vlookup_sheet_name']}' not found. Available: {xl.sheet_names}")

    df_raw = pd.read_excel(data_cfg["vlookup_path"], sheet_name=data_cfg["vlookup_sheet_name"], header=None)
    
    # Find header row dynamically
    cat_marker = data_cfg["vlookup_header_markers"]["category"].lower()
    desc_marker = data_cfg["vlookup_header_markers"]["description"].lower()
    
    header_idx, cat_col, desc_col = None, None, None
    for idx, row in df_raw.iterrows():
        row_strs = [str(v).lower().strip() for v in row.values]
        if any(cat_marker in v for v in row_strs) and any(desc_marker in v for v in row_strs):
            header_idx = idx
            cat_col = next(i for i, v in enumerate(row_strs) if cat_marker in v)
            desc_col = next(i for i, v in enumerate(row_strs) if desc_marker in v)
            break
            
    if header_idx is None:
        raise ValueError("Could not find VLOOKUP header markers in the Excel file.")

    # Extract valid data rows
    skip_patterns = [p.lower() for p in data_cfg["vlookup_skip_patterns"]]
    lookup = {}
    
    for idx in range(header_idx + 1, len(df_raw)):
        row = df_raw.iloc[idx]
        row_strs = [str(v).lower().strip() for v in row.values]
        if any(p in s for s in row_strs for p in skip_patterns):
            continue
            
        cat_val = row.iloc[cat_col]
        desc_val = row.iloc[desc_col]
        
        if pd.isna(cat_val) or pd.isna(desc_val):
            continue
            
        cat_str = str(cat_val).strip()
        desc_str = str(desc_val).strip().upper()
        
        if cat_str and desc_str and desc_str != "NAN":
            lookup[desc_str] = cat_str

    # Build category index with fallback at the end
    unique_cats = sorted(set(lookup.values()))
    fallback = data_cfg["fallback_category_name"]
    if fallback in unique_cats:
        unique_cats.remove(fallback)
    unique_cats.append(fallback)
    
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    lookup_indexed = {desc: cat_to_idx[cat] for desc, cat in lookup.items()}
    
    logger.info(f"VLOOKUP parsed: {len(lookup_indexed)} mappings -> {len(unique_cats)} categories")
    return lookup_indexed, len(unique_cats)

# =============================================================================
# 3. MAIN PIPELINE (OPTIMIZED FOR 2M+ ROWS)
# =============================================================================
def preprocess(config_path: str) -> None:
    cfg = load_and_validate_config(config_path)
    data_cfg = cfg["data"]
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Step 1: Loading raw data...")
    ext = Path(data_cfg["raw_data_path"]).suffix.lower()
    df = pd.read_csv(data_cfg["raw_data_path"]) if ext == ".csv" else pd.read_excel(data_cfg["raw_data_path"])
    logger.info(f"Loaded {len(df):,} records.")

    # 2. Parse GPS (Vectorized)
    logger.info("Step 2: Parsing geolocation...")
    lat_b = tuple(data_cfg["lat_bounds"])
    lon_b = tuple(data_cfg["lon_bounds"])
    
    # Apply parsing safely
    coords = df[data_cfg["geolocation_column"]].apply(lambda x: parse_geolocation_safe(x, lat_b, lon_b))
    df["lat_clean"] = coords.apply(lambda c: c[0])
    df["lon_clean"] = coords.apply(lambda c: c[1])
    
    valid_mask = (df["lat_clean"] != 0.0) & (df["lon_clean"] != 0.0)
    n_invalid = (~valid_mask).sum()
    logger.info(f"Invalid/out-of-bounds GPS: {n_invalid:,} ({100*n_invalid/len(df):.1f}%) -> Routed to Virtual Node")

    # 3. Spatial Gridding (Vectorized, NO itertuples)
    logger.info("Step 3: Assigning spatial grids...")
    grid_size = data_cfg["grid_size_deg"]
    lat_min, lon_min = data_cfg["lat_min"], data_cfg["lon_min"]

    df["grid_x"] = np.where(valid_mask, np.floor((df["lon_clean"] - lon_min) / grid_size).astype(int), -1)
    df["grid_y"] = np.where(valid_mask, np.floor((df["lat_clean"] - lat_min) / grid_size).astype(int), -1)

    # Get unique physical grids and assign IDs using standard dict mapping
    phys_grids = df.loc[valid_mask, ["grid_x", "grid_y"]].drop_duplicates().reset_index(drop=True)
    phys_grids["node_id"] = range(len(phys_grids))
    
    # Create mapping dictionary safely
    grid_to_node = {(r["grid_x"], r["grid_y"]): r["node_id"] for r in phys_grids.to_dict("records")}
    
    # Map node IDs to main dataframe using vectorized tuple mapping
    df["grid_key"] = list(zip(df["grid_x"], df["grid_y"]))
    df["node_id"] = df["grid_key"].map(grid_to_node).fillna(-1).astype(int)
    
    n_physical = len(phys_grids)
    virtual_id = n_physical
    df.loc[~valid_mask, "node_id"] = virtual_id
    n_nodes = n_physical + 1
    logger.info(f"Physical grids: {n_physical:,} | Total nodes (w/ virtual): {n_nodes:,}")

    # 4. Temporal Binning
    logger.info("Step 4: Binning timestamps...")
    hours = data_cfg["hours_per_bin"]
    df["timestamp"] = pd.to_datetime(df[data_cfg["date_column"]].astype(str) + " " + df[data_cfg["time_column"]].astype(str))
    df["time_bin"] = df["timestamp"].dt.floor(f"{hours}h")
    
    all_bins = sorted(df["time_bin"].unique())
    bin_idx_map = {b: i for i, b in enumerate(all_bins)}
    df["bin_idx"] = df["time_bin"].map(bin_idx_map)
    n_bins = len(all_bins)

    # 5. Map Violations
    logger.info("Step 5: Mapping violation descriptions...")
    lookup, n_channels = build_violation_lookup(cfg)
    other_idx = n_channels - 1
    
    upper_desc = df[data_cfg["description_column"]].astype(str).str.strip().str.upper()
    df["channel_id"] = upper_desc.map(lookup).fillna(other_idx).astype(int)
    
    n_unmapped = (df["channel_id"] == other_idx).sum()
    logger.info(f"Unmapped descriptions -> fallback: {n_unmapped:,} ({100*n_unmapped/len(df):.1f}%)")

        # 6. Aggregate Tensors (Using fast groupby)
    logger.info("Step 6: Aggregating tensors...")
    
    # OPTIMIZATION: Use int16 (2 bytes) instead of float32 (4 bytes). 
    # Max value 32,767 is more than enough for 6-hour counts.
    counts = df.groupby(["bin_idx", "node_id", "channel_id"]).size().reset_index(name="count")
    X = np.zeros((n_bins, n_nodes, n_channels), dtype=np.int16)
    X[counts["bin_idx"], counts["node_id"], counts["channel_id"]] = counts["count"].values
    
    # DELETED: Y_count = X.copy()  <-- This was eating an extra 8GB of RAM!

    # Admin tensor
    # OPTIMIZATION: Use float16 (2 bytes) for proportions.
    n_admin = len(data_cfg["admin_columns"])
    Y_admin = np.zeros((n_bins, n_nodes, n_admin), dtype=np.float16)
    for ci, col in enumerate(data_cfg["admin_columns"]):
        if col in df.columns:
            prop = df.groupby(["bin_idx", "node_id"])[col].apply(lambda x: (x == "Yes").sum() / max(len(x), 1))
            for (b, n), v in prop.items():
                Y_admin[b, n, ci] = v

    # Threat tensor
    # OPTIMIZATION: Use float16 (2 bytes) for proportions.
    n_threat = len(data_cfg["threat_columns"])
    Y_threat = np.zeros((n_bins, n_nodes, n_threat), dtype=np.float16)
    for ci, col in enumerate(data_cfg["threat_columns"]):
        if col in df.columns:
            prop = df.groupby(["bin_idx", "node_id"])[col].apply(lambda x: (x == "Yes").sum() / max(len(x), 1))
            for (b, n), v in prop.items():
                Y_threat[b, n, ci] = v

    # 7. Adjacency Matrix
    logger.info("Step 7: Building adjacency matrix...")
    radius_deg = data_cfg["adjacency_radius_km"] / 111.0
    centers = np.zeros((n_physical, 2))
    for r in phys_grids.to_dict("records"):
        centers[r["node_id"]] = [lat_min + (r["grid_y"] + 0.5) * grid_size, lon_min + (r["grid_x"] + 0.5) * grid_size]

    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    A[:n_physical, :n_physical] = ((dists <= radius_deg) & (dists > 0)).astype(np.float32)
    A[virtual_id, :n_physical] = 1.0
    A[:n_physical, virtual_id] = 1.0
    logger.info(f"Adjacency matrix: {n_nodes} nodes, {int(A.sum()):,} edges")

        # 8. Save
    logger.info("Step 8: Saving processed tensors...")
    
    # Save X as both X.pt and Y_count.pt to avoid keeping a second copy in memory
    torch.save(torch.from_numpy(X), out_dir / "X.pt")
    torch.save(torch.from_numpy(X), out_dir / "Y_count.pt") 
    
    torch.save(torch.from_numpy(Y_admin), out_dir / "Y_admin.pt")
    torch.save(torch.from_numpy(Y_threat), out_dir / "Y_threat.pt")
    torch.save(torch.from_numpy(A), out_dir / "adj_matrix.pt")