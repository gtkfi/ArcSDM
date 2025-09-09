import os
import re
import arcpy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def _rows_with_labels(df: pd.DataFrame, label_cols: list[str], csv_nodata: float) -> pd.Series:
    """Row is labeled if ANY label_col is present (not NaN/empty/sentinel)."""
    if not label_cols:
        # No label columns found → nothing to filter; treat all as unlabeled
        return pd.Series(False, index=df.index)

    masks = []
    for col in label_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            m = s.notna() & (s.astype(float) != float(csv_nodata))
        else:
            # Treat '', 'nan', 'none', 'null', and textual sentinel as missing
            txt = s.astype("string").str.strip()
            m = txt.notna() & (txt != "") \
                & ~txt.str.lower().isin({"nan", "none", "null", str(csv_nodata).lower()})
        masks.append(m)

    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return mask


def _sanitize_features(df: pd.DataFrame, fields: list[str], csv_nodata: float) -> pd.DataFrame:
    """Return a numeric-only view of df[fields] with csv_nodata turned into NaN."""
    # Coerce to numeric (invalid → NaN) and then map sentinel → NaN
    clean = df[fields].apply(pd.to_numeric, errors="coerce").replace(csv_nodata, np.nan)
    return clean


def cosine_similarity(vector_a, vector_b, nodata_value=-9999.0):
    """
    Calculate cosine similarity between two vectors, ignoring NaN and nodata.
    CSI(A,B) = cos θ = (A·B) / (||A|| ||B||)
    """
    # Build mask for valid (both finite, not nodata)
    mask = ~(
        np.isnan(vector_a) | np.isnan(vector_b) |
        (vector_a == nodata_value) | (vector_b == nodata_value)
    )
    if not np.any(mask):
        return nodata_value

    a_clean = vector_a[mask]
    b_clean = vector_b[mask]
    if a_clean.size == 0:
        return nodata_value

    dot_product = np.dot(a_clean, b_clean)
    magnitude_a = np.linalg.norm(a_clean)
    magnitude_b = np.linalg.norm(b_clean)

    if magnitude_a == 0 or magnitude_b == 0:
        return nodata_value
    return dot_product / (magnitude_a * magnitude_b)


def load_labeled_data(labelled_path, label_field_names, feature_field_names):
    """Load labeled data from feature class or table"""
    try:
        if not feature_field_names:
            field_objects = arcpy.ListFields(labelled_path)
            feature_field_names = [
                f.name for f in field_objects
                if f.type in ['Double', 'Float', 'Integer', 'SmallInteger']
            ]
            exclude_fields = ['OBJECTID', 'OID', 'FID', 'Shape_Length', 'Shape_Area', 'SHAPE']
            feature_field_names = [f for f in feature_field_names if f not in exclude_fields]

        fields = feature_field_names.copy()
        if label_field_names:
            fields.extend([f for f in label_field_names if f not in fields])

        desc = arcpy.Describe(labelled_path)
        has_geometry = hasattr(desc, 'shapeType')
        if has_geometry:
            fields.append('SHAPE@XY')

        data = []
        with arcpy.da.SearchCursor(labelled_path, fields) as cursor:
            for row in cursor:
                data.append(row)

        df = pd.DataFrame(data, columns=fields)
        # Normalize "missing" values
        df.replace({None: np.nan}, inplace=True)

        arcpy.AddMessage(f"Loaded {len(df)} labeled points with {len(feature_field_names)} features")
        return df, feature_field_names, has_geometry

    except Exception as e:
        arcpy.AddError(f"Error loading labeled data: {e}")
        return None, None, False


def load_raster_data(rasters_list):
    """Load raster data"""
    try:
        raster_data = []
        for raster_path in rasters_list:
            if arcpy.Exists(raster_path):
                raster_data.append(raster_path)
            else:
                arcpy.AddWarning(f"Raster does not exist: {raster_path}")
        
        arcpy.AddMessage(f"Loaded {len(raster_data)} raster layers")
        return raster_data
        
    except Exception as e:
        arcpy.AddError(f"Error loading raster data: {e}")
        return []

def extract_raster_valuessss(raster_path, points_df, has_geometry):
    """Extract raster values at point locations"""
    try:
        if has_geometry:
            # Use SHAPE@XY coordinates
            coord_field = 'SHAPE@XY'
            point_values = []
            
            for idx, row in points_df.iterrows():
                x, y = row[coord_field]
                try:
                    # Extract value at point
                    result = arcpy.GetCellValue_management(raster_path, f"{x} {y}")
                    value = float(result.getOutput(0))
                    point_values.append(value)
                except:
                    point_values.append(np.nan)
            
            return np.array(point_values)
        else:
            points_values = arcpy.RasterToNumPyArray(raster_path)
            return np.array(points_values)
            
    except Exception as e:
        arcpy.AddError(f"Error extracting raster values: {e}")
        return np.array([])

def calculate_pairwise_csi(labeled_df, feature_fields, csv_nodata, block_size=None):
    """
    Fast pairwise CSI between all labeled points.
    - Ignores NaN and csv_nodata (treated as missing).
    - Sets CSI=csv_nodata where vectors have no overlapping valid dimensions
      or where denominators are zero.
    - Diagonal is forced to 1.0.

    Args:
        labeled_df: DataFrame containing features (and possibly other cols).
        feature_fields: list[str] columns to use.
        csv_nodata: float sentinel for "no data".
        block_size: if None, compute full matrix in one go.
                    If set (e.g., 2000), compute in blocks to reduce peak memory.

    Returns:
        np.ndarray (n_points, n_points) of CSI values.
    """
    # 1) Sanitize to numeric and replace sentinel -> NaN
    F = _sanitize_features(labeled_df, feature_fields, csv_nodata)  # (n,d), NaNs for missing
    n, d = F.shape
    arcpy.AddMessage(f"Calculating pairwise CSI for {n} points (d={d}). block_size={block_size or 'all'}")

    # 2) Validity mask & zero-filled copy for fast algebra
    M = ~F.isna().to_numpy()                       # (n,d) bool: True where valid
    X = np.nan_to_num(F.to_numpy(dtype=float), nan=0.0)  # (n,d) float: NaN -> 0

    # 3) Row norms over valid entries only (zeros elsewhere don’t contribute)
    norms = np.linalg.norm(X, axis=1)              # (n,)

    # Prepare output
    out = np.full((n, n), csv_nodata, dtype=float)

    # Full vectorized (fastest, more memory)
    if block_size is None:
        # Numerator = X @ X^T only counts overlapping valid dims because invalids are 0
        numer = X @ X.T                            # (n,n)

        # Denominator per pair = ||xi|| * ||xj||
        denom = norms[:, None] * norms[None, :]    # (n,n)

        # Overlap count (how many dims valid in both) to detect empty intersections
        overlap = (M @ M.T).astype(np.int32)       # (n,n)

        with np.errstate(divide='ignore', invalid='ignore'):
            cos = numer / denom

        # Where denom==0 or no overlap -> csv_nodata (leave as initialized)
        valid_pairs = (denom > 0) & (overlap > 0)
        out[valid_pairs] = cos[valid_pairs]

        # Diagonal = 1.0 as per original behavior
        np.fill_diagonal(out, 1.0)
        return out

    # Block processing (lower memory; still very fast)
    bs = int(block_size)
    arcpy.AddMessage("Using block processing…")
    for i0 in range(0, n, bs):
        i1 = min(i0 + bs, n)

        Xi = X[i0:i1]                     # (bi,d)
        Mi = M[i0:i1]                     # (bi,d)
        norm_i = norms[i0:i1]             # (bi,)

        # Numerator block: Xi @ X^T -> (bi,n)
        numer_blk = Xi @ X.T

        # Denominator block: outer(norm_i, norms) -> (bi,n)
        denom_blk = norm_i[:, None] * norms[None, :]

        # Overlap block: Mi @ M^T -> (bi,n)
        overlap_blk = (Mi @ M.T).astype(np.int32)

        with np.errstate(divide='ignore', invalid='ignore'):
            cos_blk = numer_blk / denom_blk

        valid_pairs = (denom_blk > 0) & (overlap_blk > 0)
        out[i0:i1][valid_pairs] = cos_blk[valid_pairs]

        # Progress every couple blocks
        if (i0 // bs) % 2 == 0:
            arcpy.AddMessage(f"  processed rows {i0+1}–{i1} / {n}")

    # Diagonal = 1.0
    np.fill_diagonal(out, 1.0)
    return out


def calculate_centroid_matrix(labeled_df, feature_fields, csv_nodata):
    """CSI between each labeled point and the centroid of all points."""
    features_clean_df = _sanitize_features(labeled_df, feature_fields, csv_nodata)
    centroid = features_clean_df.mean(skipna=True).to_numpy(dtype=float)

    F = features_clean_df.to_numpy(dtype=float)  # (n,d)
    M = ~np.isnan(F) & ~np.isnan(centroid)       # (n,d) valid overlap mask

    # Zero out invalids
    F_masked = np.where(M, F, 0.0)
    centroid_masked = np.where(M, centroid, 0.0)

    # Dot products per row
    numer = np.sum(F_masked * centroid_masked, axis=1)

    # Norms per row and centroid
    row_norms = np.linalg.norm(F_masked, axis=1)
    centroid_norms = np.linalg.norm(centroid_masked, axis=1)  # (n,d) → shape (n,)

    denom = row_norms * centroid_norms
    csi = np.full(F.shape[0], csv_nodata, dtype=float)

    valid = (denom > 0)
    csi[valid] = numer[valid] / denom[valid]

    return csi, centroid


def _detect_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find X/Y columns with very tolerant matching."""
    norm = {c.replace(" ", "").lstrip("\ufeff").strip().lower(): c for c in df.columns}
    pairs = [
        ("x","y"), ("lon","lat"), ("longitude","latitude"),
        ("easting","northing"), ("xcoord","ycoord"),
        ("projx","projy"), ("utm_x","utm_y"), ("x_","y_"),
    ]
    for a, b in pairs:
        arcpy.AddMessage(f" {a} and {b}")
        if a in norm and b in norm:
            return norm[a], norm[b]
    # last-ditch: exact X/Y uppercase
    if "X" in df.columns and "Y" in df.columns:
        return "X", "Y"
    return None, None

# --- raster sampling --------------------------------------------------------

def extract_raster_values(raster_path: str, points_df: pd.DataFrame, has_geometry: bool) -> np.ndarray:
    """
    Return a 1-D float array of length len(points_df) with the raster value at each row's location.
    Uses SHAPE@XY (if present) or tolerant XY-column detection. NoData => np.nan.
    """
    ras = arcpy.Raster(raster_path)
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")

    ex = ras.extent
    cw, ch = ras.meanCellWidth, ras.meanCellHeight
    nrows, ncols = arr.shape
    out = np.full(len(points_df), np.nan, dtype="float32")

    # Get XYs
    xs = ys = None
    if has_geometry and "SHAPE@XY" in points_df.columns:
        xy = points_df["SHAPE@XY"].to_numpy()
        xs = np.array([t[0] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
        ys = np.array([t[1] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
    else:
        xcol, ycol = _detect_xy_columns(points_df)
        if xcol and ycol:
            xs = pd.to_numeric(points_df[xcol], errors="coerce").to_numpy(dtype="float32")
            ys = pd.to_numeric(points_df[ycol], errors="coerce").to_numpy(dtype="float32")

    if xs is None or ys is None:
        raise ValueError("Could not find coordinates: need SHAPE@XY or X/Y columns.")

    # Vector index into the raster array
    cols = np.floor((xs - ex.XMin) / cw).astype("int64", copy=False)
    rows = np.floor((ex.YMax - ys) / ch).astype("int64", copy=False)
    valid = (
        np.isfinite(xs) & np.isfinite(ys) &
        (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
    )
    out[valid] = arr[rows[valid], cols[valid]]
    return out

# --- CSI calculation --------------------------------------------------------

def calculate_evidence_csi(labeled_df, feature_fields, rasters_list,
                           evidence_vectors_file, has_geometry, csv_nodata):
    """CSI between labeled points and evidence, ignoring NaN/csv_nodata consistently."""
    evidence_results = {}

    # features_clean must be numeric with NaNs where missing
    features_clean_df = _sanitize_features(labeled_df, feature_fields, csv_nodata)
    features_clean = features_clean_df.to_numpy(dtype=float)

    if rasters_list:
        arcpy.AddMessage("Processing raster evidence...")
        for i, raster_path in enumerate(rasters_list):
            arcpy.AddMessage(f"Processing raster {i+1}/{len(rasters_list)}: {os.path.basename(raster_path)}")
            raster_values = extract_raster_values(raster_path, labeled_df, has_geometry)  # -> (N,)

            N = len(labeled_df)
            raster_csi = np.full(N, csv_nodata, dtype="float32")

            for j in range(N):
                rv = raster_values[j]
                # Skip invalid raster value
                if (isinstance(rv, float) and np.isnan(rv)) or (csv_nodata is not None and rv == csv_nodata):
                    continue

                # Pick first valid feature value in this row
                row_feat = features_clean[j]
                valid_mask = ~np.isnan(row_feat)
                if not np.any(valid_mask):
                    continue
                feat_val = row_feat[valid_mask][0]

                # Cosine similarity for scalar inputs; ensure scalar output
                csi_value = cosine_similarity(np.array([feat_val]), np.array([rv]), csv_nodata)
                csi_value = float(np.asarray(csi_value).squeeze())
                raster_csi[j] = csi_value

            valid_count = int(np.sum(raster_csi != csv_nodata))
            arcpy.AddMessage(f"  Calculated {valid_count} valid CSI values for raster {i+1}")
            evidence_results[f"raster_{i+1}_{os.path.basename(raster_path)}"] = raster_csi

    # --- Vector evidence (multi-dim) ---
    if evidence_vectors_file and arcpy.Exists(evidence_vectors_file):
        arcpy.AddMessage("Processing vector evidence...")
        evidence_fields = [
            f.name for f in arcpy.ListFields(evidence_vectors_file)
            if f.type in ['Double', 'Float', 'Integer', 'SmallInteger']
        ]

        evidence_data = []
        with arcpy.da.SearchCursor(evidence_vectors_file, evidence_fields) as cursor:
            for row in cursor:
                # Normalize None → NaN
                evidence_data.append([np.nan if val is None else val for val in row])

        evidence_df = pd.DataFrame(evidence_data, columns=evidence_fields)
        evidence_clean = evidence_df.apply(pd.to_numeric, errors="coerce").replace(csv_nodata, np.nan)

        for ev_idx in range(len(evidence_clean)):
            ev_vec = evidence_clean.iloc[ev_idx].to_numpy(dtype=float)

            vector_csi = np.full(len(labeled_df), csv_nodata)
            for label_idx in range(len(labeled_df)):
                lab_vec = features_clean[label_idx]
                # Align by valid indices in both vectors
                valid = ~(np.isnan(lab_vec) | np.isnan(ev_vec))
                if not np.any(valid):
                    continue
                csi_value = cosine_similarity(lab_vec[valid], ev_vec[valid], csv_nodata)
                vector_csi[label_idx] = csi_value

            valid_count = np.sum(vector_csi != csv_nodata)
            arcpy.AddMessage(f"  Calculated {valid_count} valid CSI values for evidence vector {ev_idx+1}")
            evidence_results[f"evidence_vector_{ev_idx+1}"] = vector_csi

    return evidence_results


def create_output_rasters(
    labeled_df,
    evidence_results,
    out_raster_folder,
    csv_nodata,
    has_geometry,
    source_fc=None,
    template_raster=None,
    cell_size=None,
    min_points=3,
):
    """
    Create rasters from CSI results by IDW.
    - Uses template_raster (preferred) to set extent/snap/cell size.
    - Otherwise derives SR from source_fc or env; falls back to WGS84.
    - Skips layers with fewer than min_points valid points.
    """
 
    if not has_geometry:
        arcpy.AddWarning("Cannot create rasters without point geometry.")
        return

    # Make sure output folder exists
    os.makedirs(out_raster_folder, exist_ok=True)

    # Resolve spatial reference
    sr = None
    try:
        if template_raster and arcpy.Exists(template_raster):
            sr = arcpy.Describe(template_raster).spatialReference
        elif source_fc and arcpy.Exists(source_fc):
            sr = arcpy.Describe(source_fc).spatialReference
        elif arcpy.env.outputCoordinateSystem:
            sr = arcpy.env.outputCoordinateSystem
    except Exception:
        sr = None

    if sr is None or sr.name in (None, "", "Unknown"):
        arcpy.AddWarning("No spatial reference found; defaulting to WGS 1984.")
        sr = arcpy.SpatialReference(4326)

    # Set raster environment from template if available
    if template_raster and arcpy.Exists(template_raster):
        tdesc = arcpy.Describe(template_raster)
        arcpy.env.snapRaster = template_raster
        arcpy.env.extent = tdesc.extent
        if cell_size is None:
            try:
                cell_size = float(getattr(tdesc, "meanCellWidth", None) or getattr(tdesc, "children", [])[0].meanCellWidth)
            except Exception:
                cell_size = None

    if cell_size is None:
        cell_size = arcpy.env.cellSize  # Default to environment cell size
        arcpy.AddWarning(f"No cell size provided or derivable; using default {cell_size}.")

    try:
        for evidence_name, csi_values in evidence_results.items():
            # Sanitize output name
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(evidence_name))[:60]
            out_raster_path = os.path.join(out_raster_folder, f"csi_{safe_name}.tif")

            arcpy.AddMessage(f"Creating raster for {evidence_name} → {out_raster_path}")

            # Build temp point FC with SR
            temp_fc = arcpy.management.CreateFeatureclass(
                "in_memory", f"tmp_pts_{safe_name}", "POINT", spatial_reference=sr
            )
            arcpy.management.AddField(temp_fc, "CSI_VALUE", "DOUBLE")

            # Insert points (skip nodata/NaN)
            inserted = 0
            with arcpy.da.InsertCursor(temp_fc, ["SHAPE@", "CSI_VALUE"]) as icur:
                for idx, csi_val in enumerate(csi_values):
                    if idx >= len(labeled_df):
                        break
                    if csi_val == csv_nodata or not np.isfinite(csi_val):
                        continue
                    try:
                        x, y = labeled_df["SHAPE@XY"].iloc[idx]
                    except Exception:
                        # If DF lacks XY, bail for this layer
                        x = y = None
                    if x is None or y is None:
                        continue
                    geom = arcpy.PointGeometry(arcpy.Point(float(x), float(y)), sr)
                    icur.insertRow([geom, float(csi_val)])
                    inserted += 1

            if inserted < min_points:
                arcpy.AddWarning(
                    f"Only {inserted} valid points for {evidence_name}; "
                    f"need ≥{min_points} for IDW. Skipping."
                )
                arcpy.management.Delete(temp_fc)
                continue

            # Run IDW
            try:
                idw_raster = arcpy.sa.Idw(
                    in_point_features=temp_fc,
                    z_field="CSI_VALUE",
                    cell_size=cell_size,  # uses env.snapRaster/extents if set
                )
                idw_raster.save(out_raster_path)
                arcpy.AddMessage(f"Saved raster: {out_raster_path}")
            except Exception as e:
                arcpy.AddError(f"IDW failed for {evidence_name}: {e}")
            finally:
                arcpy.management.Delete(temp_fc)

    except Exception as e:
        arcpy.AddError(f"Error creating output rasters: {e}")



def save_csv_results(pairwise_matrix, centroid_csi, evidence_results, 
                    out_labelled_pairwise_csv, out_centroid_matrix_csv, 
                    out_evidence_table_csv, csv_nodata):
    """Save all results to CSV files"""
    try:
        # Save pairwise CSI matrix
        pairwise_df = pd.DataFrame(pairwise_matrix)
        pairwise_df.index = [f"Point_{i+1}" for i in range(len(pairwise_matrix))]
        pairwise_df.columns = [f"Point_{i+1}" for i in range(len(pairwise_matrix))]
        pairwise_df.to_csv(out_labelled_pairwise_csv)
        arcpy.AddMessage(f"Saved pairwise CSI matrix: {out_labelled_pairwise_csv}")
        
        # Save centroid CSI
        centroid_df = pd.DataFrame({
            'Point_ID': [f"Point_{i+1}" for i in range(len(centroid_csi))],
            'Centroid_CSI': centroid_csi
        })
        centroid_df.to_csv(out_centroid_matrix_csv, index=False)
        arcpy.AddMessage(f"Saved centroid CSI: {out_centroid_matrix_csv}")
        
        # Save evidence CSI table
        if evidence_results and out_evidence_table_csv:
            evidence_df = pd.DataFrame(evidence_results)
            evidence_df.index = [f"Point_{i+1}" for i in range(len(evidence_df))]
            evidence_df.to_csv(out_evidence_table_csv)
            arcpy.AddMessage(f"Saved evidence CSI table: {out_evidence_table_csv}")
            
    except Exception as e:
        arcpy.AddError(f"Error saving CSV results: {e}")

def execute(self, parameters, messages):
    """Execute the CSI calculation"""
    try:
        # Get parameters
        labelled_path = parameters[0].valueAsText
        csv_nodata = float(parameters[1].value) if parameters[1].value else -9999.0
        label_field_names = parameters[2].valueAsText.split(';') if parameters[2].valueAsText else []
        feature_field_names = parameters[3].valueAsText.split(';') if parameters[3].valueAsText else None
        evidence_type = parameters[4].valueAsText if parameters[4].value else None
        rasters_list = parameters[5].valueAsText.split(';') if parameters[5].valueAsText else []
        evidence_vectors_file = parameters[6].valueAsText if parameters[6].value else None
        out_labelled_pairwise_csv = parameters[7].valueAsText
        out_centroid_matrix_csv = parameters[8].valueAsText
        out_evidence_table_csv = parameters[9].valueAsText if parameters[9].value else None
        out_raster_folder = parameters[10].valueAsText if parameters[10].value else None
        
        arcpy.AddMessage("Starting CSI Analysis...")
        arcpy.AddMessage("=" * 50)
        
        # Load labeled data
        labeled_df, feature_fields, has_geometry = load_labeled_data(
            labelled_path, label_field_names, feature_field_names
        )
        
        if labeled_df is None:
            return
        
        label_mask = _rows_with_labels(labeled_df, label_field_names, csv_nodata)
        labeled_only = labeled_df.loc[label_mask].reset_index(drop=True)

        arcpy.AddMessage(
            f"Detected label columns: {label_field_names or 'None'}; "
            f"using {len(labeled_only)} labeled points out of {len(labeled_df)} total."
        )

        if len(labeled_only) == 0:
            arcpy.AddWarning("No labeled rows found — falling back to all rows.")
            labeled_only = labeled_df.copy()
        
        # Calculate pairwise CSI matrix
        pairwise_matrix = calculate_pairwise_csi(labeled_only, feature_fields, csv_nodata)
        
        # Calculate centroid CSI
        centroid_csi, centroid = calculate_centroid_matrix(labeled_only, feature_fields, csv_nodata)
        
        # Calculate evidence CSI if requested
        evidence_results = {}
        if evidence_type in ["Raster"] and rasters_list:
            raster_data = load_raster_data(rasters_list)
            if raster_data:
                evidence_results.update(
                    calculate_evidence_csi(
                        labeled_df, feature_fields, raster_data, None, 
                        has_geometry, csv_nodata
                    )
                )
        
        if evidence_type in ["Vector"] and evidence_vectors_file:
            evidence_results.update(
                calculate_evidence_csi(
                    labeled_df, feature_fields, [], evidence_vectors_file, 
                    has_geometry, csv_nodata
                )
            )
        
        # Save CSV results
        save_csv_results(
            pairwise_matrix, centroid_csi, evidence_results,
            out_labelled_pairwise_csv, out_centroid_matrix_csv, 
            out_evidence_table_csv, csv_nodata
        )
        
        # Create output rasters if requested
        if out_raster_folder and evidence_results:
            create_output_rasters(
                labeled_df, evidence_results, out_raster_folder,
                csv_nodata, has_geometry, template_raster=rasters_list
            )
        
        arcpy.AddMessage("\nCSI Analysis completed successfully!")
        
    except Exception as e:
        arcpy.AddError(f"Error in CSI calculation: {e}")
        import traceback
        arcpy.AddError(traceback.format_exc())