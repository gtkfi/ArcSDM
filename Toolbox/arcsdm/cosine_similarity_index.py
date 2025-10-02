import os
import re
import arcpy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

def _rows_with_labels(
    df: pd.DataFrame,
    label_cols: List[str],
    csv_nodata: float) -> pd.Series: 
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


def _sanitize_features(
    df: pd.DataFrame,
    label_cols: List[str],
    csv_nodata: float
) -> pd.DataFrame:
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


def _sanitize_features(
    df: pd.DataFrame,
    fields: List[str],
    csv_nodata: float
) -> pd.DataFrame:
    """Return a numeric-only view of df[fields] with csv_nodata turned into NaN."""
    # Coerce to numeric (invalid → NaN) and then map sentinel → NaN
    clean = df[fields].apply(pd.to_numeric, errors="coerce").replace(csv_nodata, np.nan)
    return clean


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    nodata_value: float = -9999.0
) -> float:
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


def load_labeled_data(
    labelled_path: str,
    label_field_names: Optional[List[str]],
    feature_field_names: Optional[List[str]]
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]], bool]:

    """Load labeled data from feature class or table"""
    try:
        
        if not feature_field_names:
            field_objects = arcpy.ListFields(labelled_path)
            all_numeric_fields = [
                f.name for f in field_objects
                if f.type in ['Double', 'Float', 'Integer', 'SmallInteger']
            ]
            exclude_fields = ['OBJECTID', 'OID', 'FID', 'Shape_Length', 'Shape_Area', 'SHAPE']
            all_numeric_fields = [f for f in all_numeric_fields if f not in exclude_fields]
            
            # Add any remaining fields if needed
            remaining_fields = [f for f in all_numeric_fields if f not in feature_field_names]
            feature_field_names.extend(remaining_fields)
            
            arcpy.AddMessage(f"Auto-detected feature fields: {feature_field_names}")

        fields = feature_field_names.copy()
        if label_field_names:
            fields.extend([f for f in label_field_names if f not in fields])

        desc = arcpy.Describe(labelled_path)
        has_geometry = hasattr(desc, 'shapeType')
        
        # For non-geometry tables, try to detect coordinate fields
        if not has_geometry:
            sample_fields = [f.name for f in arcpy.ListFields(labelled_path)]
            temp_data = []
            with arcpy.da.SearchCursor(labelled_path, sample_fields[:10]) as cursor:
                for i, row in enumerate(cursor):
                    if i >= 5:
                        break
                    temp_data.append(row)
            
            if temp_data:
                temp_df = pd.DataFrame(temp_data, columns=sample_fields[:10])
                xcol, ycol = _detect_xy_columns(temp_df)
                if xcol and ycol and xcol not in fields:
                    fields.append(xcol)
                if xcol and ycol and ycol not in fields:
                    fields.append(ycol)
        else:
            fields.append('SHAPE@XY')

        data = []
        with arcpy.da.SearchCursor(labelled_path, fields) as cursor:
            for row in cursor:
                data.append(row)

        df = pd.DataFrame(data, columns=fields)
        df.replace({None: np.nan}, inplace=True)

        arcpy.AddMessage(f"Loaded {len(df)} labeled points with {len(feature_field_names)} features")
        return df, feature_field_names, has_geometry

    except Exception as e:
        arcpy.AddError(f"Error loading labeled data: {e}")
        return None, None, False


def load_raster_data(
    rasters_list: List[str]
) -> List[str]:
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


def calculate_corner_csi(
    labeled_df: pd.DataFrame,
    feature_fields: List[str],
    csv_nodata: float
) -> np.ndarray:
    """
    Calculate CSI between corner vectors A and B.
    Returns only upper diagonal of the matrix.
    """
    # Focus only on labeled points
    F = _sanitize_features(labeled_df, feature_fields, csv_nodata)
    n = len(F)
    
    arcpy.AddMessage(f"Calculating corner CSI for {n} labeled points with vectors A and B")
    
    # Initialize results matrix (upper triangular only)
    corner_csi = np.full((n, n), None, dtype=float)
    
    # Convert to numpy for faster computation
    features_array = F.to_numpy(dtype=float)
    
    # Calculate only upper triangular matrix
    for i in range(n):
        for j in range(i, n):  # Only upper diagonal (i <= j)
            if i == j:
                corner_csi[i, j] = 1.0  # Self-similarity
            else:
                # Calculate CSI between vector A (row i) and vector B (row j)
                vector_a = features_array[i]
                vector_b = features_array[j]
                
                # Calculate cosine similarity
                csi_value = cosine_similarity(vector_a, vector_b, csv_nodata)
                corner_csi[i, j] = csi_value
                
        # Progress reporting
        if (i + 1) % 50 == 0 or i == n - 1:
            arcpy.AddMessage(f"  Processed corner {i + 1}/{n}")
    
    return corner_csi


def _detect_xy_columns(
    df: pd.DataFrame
) -> Tuple[Optional[str], Optional[str]]:

    """Find X/Y columns with tolerant matching."""
    
    norm = {c.replace(" ", "").lstrip("\ufeff").strip().lower(): c for c in df.columns}
    
    pairs = [
        ("x","y"), ("lon","lat"), ("longitude","latitude"),
        ("easting","northing"), ("xcoord","ycoord"),
        ("projx","projy"), ("utm_x","utm_y"), ("x_","y_"),
        ("point_x", "point_y"), ("coord_x", "coord_y"),
        ("x_coord", "y_coord"), ("xcoordinate", "ycoordinate"),
    ]
    
    for a, b in pairs:
        if a in norm and b in norm:
            return norm[a], norm[b]
    
    if "X" in df.columns and "Y" in df.columns:
        return "X", "Y"
    
    arcpy.AddMessage("No coordinate columns found")
    return None, None


def extract_raster_values(
    raster_path: str,
    points_df: pd.DataFrame,
    has_geometry: bool
) -> np.ndarray:
    """
    Return a 1-D float array of length len(points_df) with the raster value at each row's location.
    """
    try:
        ras = arcpy.Raster(raster_path)
        arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")

        ex = ras.extent
        cw, ch = ras.meanCellWidth, ras.meanCellHeight
        nrows, ncols = arr.shape
        out = np.full(len(points_df), np.nan, dtype="float32")

        # Get XYs
        xs = ys = None
        if has_geometry and "SHAPE@XY" in points_df.columns:
            arcpy.AddMessage("Using SHAPE@XY coordinates")
            xy = points_df["SHAPE@XY"].to_numpy()
            xs = np.array([t[0] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
            ys = np.array([t[1] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
        else:
            xcol, ycol = _detect_xy_columns(points_df)
            if xcol and ycol:
                arcpy.AddMessage(f"Using coordinate columns: {xcol}, {ycol}")
                xs = pd.to_numeric(points_df[xcol], errors="coerce").to_numpy(dtype="float32")
                ys = pd.to_numeric(points_df[ycol], errors="coerce").to_numpy(dtype="float32")

        if xs is None or ys is None:
            error_msg = (
                f"Could not find coordinates. Available columns: {list(points_df.columns)}\n"
                "For geometry data: ensure SHAPE@XY column exists\n"
                "For tabular data: ensure coordinate columns exist"
            )
            raise ValueError(error_msg)

        # Vector index into the raster array
        cols = np.floor((xs - ex.XMin) / cw).astype("int64", copy=False)
        rows = np.floor((ex.YMax - ys) / ch).astype("int64", copy=False)
        valid = (
            np.isfinite(xs) & np.isfinite(ys) &
            (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
        )
        
        valid_count = np.sum(valid)
        arcpy.AddMessage(f"Extracting values for {valid_count} valid points out of {len(points_df)} total")
        
        out[valid] = arr[rows[valid], cols[valid]]
        return out
        
    except Exception as e:
        arcpy.AddError(f"Error in extract_raster_values: {e}")
        raise


def calculate_evidence_matrix(
    labeled_df: pd.DataFrame,
    feature_fields: List[str],
    rasters_list: List[str],
    has_geometry: bool,
    csv_nodata: float
) -> Dict[str, np.ndarray]:
    """
    Calculate evidence matrix: labeled points vs raster values.
    Returns matrix of shape (n_labeled_points, n_rasters) for 3x40 format.
    """
    if not rasters_list:
        return {}
    
    # Focus only on labeled points
    n_points = len(labeled_df)
    n_rasters = len(rasters_list)
    
    arcpy.AddMessage(f"Calculating evidence matrix: {n_points} labeled points × {n_rasters} rasters")
    
    # Initialize evidence matrix
    evidence_matrix = np.full((n_points, n_rasters), csv_nodata, dtype=float)
    evidence_results = {}
    
    # Get feature vectors for labeled points
    features_clean_df = _sanitize_features(labeled_df, feature_fields, csv_nodata)
    features_clean = features_clean_df.to_numpy(dtype=float)
    
    for raster_idx, raster_path in enumerate(rasters_list):
        arcpy.AddMessage(f"Processing raster {raster_idx + 1}/{n_rasters}: {os.path.basename(raster_path)}")
        
        # Extract raster values at labeled point locations
        raster_values = extract_raster_values(raster_path, labeled_df, has_geometry)
        
        # Calculate CSI between each labeled point and raster values
        for point_idx in range(n_points):
            raster_val = raster_values[point_idx]
            
            # Skip invalid raster values
            if np.isnan(raster_val) or raster_val == csv_nodata:
                continue
                
            # Get feature vector for this point
            point_features = features_clean[point_idx]
            
            # Skip if no valid features
            valid_features = ~np.isnan(point_features)
            if not np.any(valid_features):
                continue
            
            # For now, use first valid feature for CSI calculation with raster
            # Could be enhanced to use all features or specific combinations
            first_valid_feature = point_features[valid_features][0]
            
            # Calculate CSI between point feature and raster value
            csi_value = cosine_similarity(
                np.array([first_valid_feature]), 
                np.array([raster_val]), 
                csv_nodata
            )
            
            evidence_matrix[point_idx, raster_idx] = float(csi_value)
        
        # Store individual raster results
        raster_name = f"raster_{raster_idx + 1}_{os.path.basename(raster_path)}"
        evidence_results[raster_name] = evidence_matrix[:, raster_idx].copy()
        
        valid_count = np.sum(evidence_matrix[:, raster_idx] != csv_nodata)
        arcpy.AddMessage(f"  Calculated {valid_count} valid CSI values")
    
    # Store the full evidence matrix
    evidence_results['evidence_matrix'] = evidence_matrix
    
    arcpy.AddMessage(f"Evidence matrix shape: {evidence_matrix.shape}")
    return evidence_results


def create_label_to_data_rasters(
    labeled_df: pd.DataFrame,
    all_data_df: pd.DataFrame,
    feature_fields: List[str],
    out_raster_folder: str,
    csv_nodata: float,
    has_geometry: bool,
    template_raster: Optional[Union[str, List[str]]] = None,
    cell_size: Optional[float] = None,
    min_points: int = 3
) -> None:
    """
    Create rasters showing CSI between each labeled point and all data points.
    """
    os.makedirs(out_raster_folder, exist_ok=True)
    
    # Resolve spatial reference
    sr = _get_spatial_reference(template_raster, labeled_df, has_geometry)
    _set_raster_environment(template_raster, cell_size)
    
    # Get features for labeled and all data
    labeled_features = _sanitize_features(labeled_df, feature_fields, csv_nodata).to_numpy(dtype=float)
    all_features = _sanitize_features(all_data_df, feature_fields, csv_nodata).to_numpy(dtype=float)
    
    n_labeled = len(labeled_df)
    
    for label_idx in range(n_labeled):
        arcpy.AddMessage(f"Creating raster for labeled point {label_idx + 1}/{n_labeled}")
        
        # Calculate CSI between this labeled point and all data points
        label_vector = labeled_features[label_idx]
        csi_values = []
        
        for data_idx in range(len(all_data_df)):
            data_vector = all_features[data_idx]
            csi_val = cosine_similarity(label_vector, data_vector, csv_nodata)
            csi_values.append(csi_val)
        
        # Create raster from CSI values
        _create_csi_raster(
            all_data_df, 
            np.array(csi_values), 
            f"label_{label_idx + 1}_to_data",
            out_raster_folder,
            sr,
            has_geometry,
            csv_nodata,
            min_points
        )


def _get_spatial_reference(
    template_raster: Optional[Union[str, List[str]]],
) -> Optional[arcpy.SpatialReference]:
    """Get spatial reference from template or data"""
    sr = None
    try:
        if template_raster:
            template_path = template_raster[0] if isinstance(template_raster, list) else template_raster
            if arcpy.Exists(template_path):
                sr = arcpy.Describe(template_path).spatialReference
        if sr is None and arcpy.env.outputCoordinateSystem:
            sr = arcpy.env.outputCoordinateSystem
    except Exception:
        pass

    if sr is None or getattr(sr, "name", "") in ("", None, "Unknown"):
        arcpy.AddWarning("No spatial reference found; defaulting to WGS 1984 (WKID 4326).")
        sr = arcpy.SpatialReference(4326)
    
    return sr


def _set_raster_environment(
    template_raster: Optional[Union[str, List[str]]],
    cell_size: Optional[float]
) -> None:
    """Set raster processing environment"""
    if template_raster:
        template_path = template_raster[0] if isinstance(template_raster, list) else template_raster
        if arcpy.Exists(template_path):
            tdesc = arcpy.Describe(template_path)
            arcpy.env.snapRaster = template_path
            arcpy.env.extent = tdesc.extent
            arcpy.env.mask = template_path
            arcpy.AddMessage(f"Set raster environment from template: {template_path}")
            if cell_size is None:
                try:
                    cell_size = float(getattr(tdesc, "meanCellWidth", None) or 
                                    (getattr(tdesc, "children", [None])[0].meanCellWidth if getattr(tdesc, "children", None) else None))
                except Exception:
                    pass

    if cell_size is None:
        cell_size = arcpy.env.cellSize
        arcpy.AddWarning(f"No cell size provided; using environment cellSize={cell_size}")


def _create_csi_raster(
    data_df: pd.DataFrame,
    csi_values: np.ndarray,
    name: str,
    out_folder: str,
    sr: Any,
    has_geometry: bool,
    csv_nodata: float,
    min_points: int
) -> None:
    """Create individual CSI raster using IDW"""
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name))[:60]
    out_path = os.path.join(out_folder, f"csi_{safe_name}.tif")
    
    temp_fc = arcpy.management.CreateFeatureclass("in_memory", f"tmp_{safe_name}", "POINT", spatial_reference=sr)
    arcpy.management.AddField(temp_fc, "CSI_VALUE", "DOUBLE")
    
    inserted = 0
    with arcpy.da.InsertCursor(temp_fc, ["SHAPE@", "CSI_VALUE"]) as icur:
        for idx, csi_val in enumerate(csi_values):
            if idx >= len(data_df):
                break
            if csi_val == csv_nodata or not np.isfinite(csi_val):
                continue
                
            x, y = _get_point_coordinates(data_df, idx, has_geometry)
            if x is None or y is None:
                continue
                
            geom = arcpy.PointGeometry(arcpy.Point(x, y), sr)
            icur.insertRow([geom, float(csi_val)])
            inserted += 1
    
    if inserted >= min_points:
        try:
            idw_raster = arcpy.sa.Idw(temp_fc, "CSI_VALUE")
            idw_raster.save(out_path)
            arcpy.AddMessage(f"Saved raster: {out_path}")
        except Exception as e:
            arcpy.AddError(f"IDW failed for {name}: {e}")
    else:
        arcpy.AddWarning(f"Only {inserted} valid points for {name}; need ≥{min_points}")
    
    arcpy.management.Delete(temp_fc)


def _get_point_coordinates(
    df: pd.DataFrame,
    idx: int,
    has_geometry: bool
) -> Tuple[Optional[float], Optional[float]]:
    """Get X,Y coordinates for a point"""
    if has_geometry and "SHAPE@XY" in df.columns:
        v = df.iloc[idx]["SHAPE@XY"]
        if isinstance(v, tuple) and len(v) == 2:
            return v[0], v[1]
    else:
        xcol, ycol = _detect_xy_columns(df)
        if xcol and ycol:
            x = pd.to_numeric(df.iloc[idx][xcol], errors="coerce")
            y = pd.to_numeric(df.iloc[idx][ycol], errors="coerce")
            if np.isfinite(x) and np.isfinite(y):
                return float(x), float(y)
    return None, None


def save_csv_results(
    corner_matrix: np.ndarray,
    evidence_results: Dict[str, np.ndarray],
    out_labelled_pairwise_csv: str,
    out_evidence_table_csv: Optional[str]
) -> None:
    """Save results to CSV files"""
    try:
        # Save corner CSI matrix (upper triangular only)
        corner_df = pd.DataFrame(corner_matrix)
        corner_df.index = [f"Point_{i+1}" for i in range(len(corner_matrix))]
        corner_df.columns = [f"Point_{i+1}" for i in range(len(corner_matrix))]
        corner_df.to_csv(out_labelled_pairwise_csv)
        arcpy.AddMessage(f"Saved corner CSI matrix: {out_labelled_pairwise_csv}")
        
        # Save evidence matrix and individual results
        if evidence_results and out_evidence_table_csv:
            # Save the full evidence matrix if it exists
            if 'evidence_matrix' in evidence_results:
                evidence_matrix = evidence_results['evidence_matrix']
                evidence_df = pd.DataFrame(evidence_matrix)
                evidence_df.index = [f"Point_{i+1}" for i in range(len(evidence_df))]
                evidence_df.columns = [f"Raster_{i+1}" for i in range(evidence_df.shape[1])]
                evidence_df.to_csv(out_evidence_table_csv)
                arcpy.AddMessage(f"Saved evidence matrix ({evidence_df.shape}): {out_evidence_table_csv}")
            
            # Save individual raster results
            individual_results = {k: v for k, v in evidence_results.items() if k != 'evidence_matrix'}
            if individual_results:
                individual_df = pd.DataFrame(individual_results)
                individual_df.index = [f"Point_{i+1}" for i in range(len(individual_df))]
                individual_csv = out_evidence_table_csv.replace('.csv', '_individual.csv')
                individual_df.to_csv(individual_csv)
                arcpy.AddMessage(f"Saved individual evidence results: {individual_csv}")
            
    except Exception as e:
        arcpy.AddError(f"Error saving CSV results: {e}")


def validate_feature_rasters(
    rasters_list: List[str],
    feature_fields: List[str]
) -> Optional[Dict[str, str]]:
    """
    Validate that rasters correspond to feature space variables.
    Returns mapping of feature field to raster path.
    """
    if len(rasters_list) != len(feature_fields):
        arcpy.AddError(f"Number of rasters ({len(rasters_list)}) must match number of feature fields ({len(feature_fields)})")
        return None
    
    feature_raster_map = {}
    for i, (field, raster_path) in enumerate(zip(feature_fields, rasters_list)):
        if not arcpy.Exists(raster_path):
            arcpy.AddError(f"Raster does not exist: {raster_path}")
            return None
        
        feature_raster_map[field] = raster_path
        arcpy.AddMessage(f"Feature '{field}' mapped to raster: {os.path.basename(raster_path)}")
    
    return feature_raster_map


def get_raster_properties(
    raster_path: str
) -> Optional[Dict[str, Any]]:
    """Get raster spatial properties for processing"""
    try:
        raster = arcpy.Raster(raster_path)
        extent = raster.extent
        cell_width = raster.meanCellWidth
        cell_height = raster.meanCellHeight
        spatial_ref = raster.spatialReference
        
        # Convert to numpy array
        array = arcpy.RasterToNumPyArray(raster, nodata_to_value=np.nan).astype("float32")
        
        return {
            'array': array,
            'extent': extent,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'spatial_ref': spatial_ref,
            'nrows': array.shape[0],
            'ncols': array.shape[1]
        }
    except Exception as e:
        arcpy.AddError(f"Error reading raster {raster_path}: {e}")
        return None


def create_pixel_vectors(
    feature_raster_map: Dict[str, str],
    feature_fields: List[str]
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Create pixel vectors from feature rasters.
    Returns 3D array: (nrows, ncols, n_features)
    """
    arcpy.AddMessage("Creating pixel vectors from feature rasters...")
    
    # Load all raster properties
    raster_props = {}
    reference_props = None
    
    for field in feature_fields:
        raster_path = feature_raster_map[field]
        props = get_raster_properties(raster_path)
        if props is None:
            return None, None
        
        raster_props[field] = props
        
        # Use first raster as reference for dimensions and extent
        if reference_props is None:
            reference_props = props
    
    # Validate all rasters have same dimensions
    nrows, ncols = reference_props['nrows'], reference_props['ncols']
    for field, props in raster_props.items():
        if props['nrows'] != nrows or props['ncols'] != ncols:
            arcpy.AddError(f"Raster dimension mismatch for {field}: "
                          f"Expected {nrows}x{ncols}, got {props['nrows']}x{props['ncols']}")
            return None, None
    
    # Create 3D array for pixel vectors
    pixel_vectors = np.full((nrows, ncols, len(feature_fields)), np.nan, dtype='float32')
    
    for i, field in enumerate(feature_fields):
        pixel_vectors[:, :, i] = raster_props[field]['array']
        arcpy.AddMessage(f"Loaded feature '{field}' into pixel vector layer {i+1}")
    
    return pixel_vectors, reference_props


def calculate_pixel_to_label_csi(
    pixel_vectors: np.ndarray,
    labeled_features: np.ndarray,
    csv_nodata: float,
    progress_step: int = 1000
) -> List[np.ndarray]:
    """
    Calculate CSI between each pixel vector and each labeled point vector.
    Returns list of 2D arrays, one per labeled point.
    """
    nrows, ncols, n_features = pixel_vectors.shape
    n_labeled = len(labeled_features)
    
    arcpy.AddMessage(f"Calculating pixel-to-label CSI for {nrows}x{ncols} pixels vs {n_labeled} labeled points")
    
    # Initialize output arrays - one per labeled point
    csi_arrays = []
    for i in range(n_labeled):
        csi_arrays.append(np.full((nrows, ncols), csv_nodata))
    
    total_pixels = nrows * ncols
    processed_pixels = 0
    
    # Process each pixel
    for row in range(nrows):
        for col in range(ncols):
            # Get pixel vector (values from all feature rasters at this location)
            pixel_vector = pixel_vectors[row, col, :]
            
            # Skip pixels with any invalid values
            if np.any(np.isnan(pixel_vector)) or np.any(pixel_vector == csv_nodata):
                processed_pixels += 1
                continue
            
            # Calculate CSI between this pixel and each labeled point
            for label_idx in range(n_labeled):
                labeled_vector = labeled_features[label_idx]
                
                # Calculate CSI between pixel vector and labeled vector
                csi_value = cosine_similarity(pixel_vector, labeled_vector, csv_nodata)
                
                # Store result
                if csi_value != csv_nodata:
                    csi_arrays[label_idx][row, col] = csi_value
            
            processed_pixels += 1
            
            # Progress reporting
            if processed_pixels % progress_step == 0 or processed_pixels == total_pixels:
                pct = (processed_pixels / total_pixels) * 100
                arcpy.AddMessage(f"  Processed {processed_pixels}/{total_pixels} pixels ({pct:.1f}%)")
    
    return csi_arrays


def save_csi_rasters(
    csi_arrays: List[np.ndarray],
    labeled_df: pd.DataFrame,
    reference_props: Dict[str, Any],
    out_raster_folder: str,
    label_field_names: List[str],
    csv_nodata: float
) -> None:
    """
    Save CSI arrays as raster files - one per labeled point.
    """
    os.makedirs(out_raster_folder, exist_ok=True)
    
    n_labeled = len(csi_arrays)
    arcpy.AddMessage(f"Saving {n_labeled} CSI rasters to: {out_raster_folder}")
    
    # Set up raster environment
    extent = reference_props['extent']
    cell_width = reference_props['cell_width']
    cell_height = reference_props['cell_height']
    spatial_ref = reference_props['spatial_ref']
    
    for label_idx, csi_array in enumerate(csi_arrays):
        try:
            # Create label identifier
            label_id = f"label_{label_idx + 1}"
            
            # Try to get a meaningful label from the data if available
            if label_field_names and len(label_field_names) > 0:
                label_col = label_field_names[0]
                if label_col in labeled_df.columns:
                    label_value = labeled_df.iloc[label_idx][label_col]
                    if pd.notna(label_value) and str(label_value).strip():
                        # Sanitize label value for filename
                        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", str(label_value))[:20]
                        label_id = f"label_{label_idx + 1}_{safe_label}"
            
            # Create output filename
            output_filename = f"csi_{label_id}.tif"
            output_path = os.path.join(out_raster_folder, output_filename)
            
            # Convert numpy array to raster
            # Replace csv_nodata with np.nan for proper NoData handling
            clean_array = np.where(csi_array == csv_nodata, np.nan, csi_array)
            
            # Create raster from array
            lower_left = arcpy.Point(extent.XMin, extent.YMin)
            raster = arcpy.NumPyArrayToRaster(clean_array, lower_left, cell_width, cell_height)
            
            # Set spatial reference
            arcpy.management.DefineProjection(raster, spatial_ref)
            
            # Set NoData value
            arcpy.management.SetRasterProperties(raster, nodata="1 " + str(csv_nodata))
            
            # Save raster
            raster.save(output_path)
            
            # Calculate statistics
            valid_pixels = np.sum(~np.isnan(clean_array))
            min_csi = np.nanmin(clean_array) if valid_pixels > 0 else csv_nodata
            max_csi = np.nanmax(clean_array) if valid_pixels > 0 else csv_nodata
            mean_csi = np.nanmean(clean_array) if valid_pixels > 0 else csv_nodata
            
            arcpy.AddMessage(f"Saved {output_filename}: {valid_pixels} valid pixels, "
                           f"CSI range [{min_csi:.4f}, {max_csi:.4f}], mean {mean_csi:.4f}")
            
        except Exception as e:
            arcpy.AddError(f"Error saving raster for label {label_idx + 1}: {e}")


def pixel_to_label_csi(
    labeled_df: pd.DataFrame,
    feature_fields: List[str],
    rasters_list: List[str],
    out_raster_folder: str,
    label_field_names: List[str],
    csv_nodata: float
) -> None:
    """
    - Validate feature rasters match feature space
    - Create pixel vectors from rasters
    - Calculate CSI between each pixel and each labeled point
    - Output p rasters (one per labeled point)
    """
    arcpy.AddMessage("\n" + "="*60)
    arcpy.AddMessage("PART 2: Pixel-to-Label CSI Analysis")
    arcpy.AddMessage("="*60)
    
    # Step 1: Validate feature rasters
    arcpy.AddMessage("Step 1: Validating feature rasters...")
    feature_raster_map = validate_feature_rasters(rasters_list, feature_fields)
    if feature_raster_map is None:
        arcpy.AddError("Feature raster validation failed")
        return False
    
    # Step 2: Create pixel vectors
    arcpy.AddMessage("Step 2: Creating pixel vectors...")
    pixel_vectors, reference_props = create_pixel_vectors(feature_raster_map, feature_fields)
    if pixel_vectors is None:
        arcpy.AddError("Failed to create pixel vectors")
        return False
    
    # Step 3: Prepare labeled feature vectors
    arcpy.AddMessage("Step 3: Preparing labeled feature vectors...")
    labeled_features_df = _sanitize_features(labeled_df, feature_fields, csv_nodata)
    labeled_features = labeled_features_df.to_numpy(dtype='float32')
    
    # Validate labeled features
    valid_labels = []
    for i, label_vector in enumerate(labeled_features):
        if not np.any(np.isnan(label_vector)) and not np.any(label_vector == csv_nodata):
            valid_labels.append(i)
    
    if len(valid_labels) == 0:
        arcpy.AddError("No valid labeled feature vectors found")
        return False
    
    arcpy.AddMessage(f"Using {len(valid_labels)} valid labeled points out of {len(labeled_features)}")
    
    # Filter to valid labels only
    labeled_features = labeled_features[valid_labels]
    labeled_df_filtered = labeled_df.iloc[valid_labels].reset_index(drop=True)
    
    # Step 4: Calculate pixel-to-label CSI
    arcpy.AddMessage("Step 4: Calculating pixel-to-label CSI...")
    csi_arrays = calculate_pixel_to_label_csi(
        pixel_vectors, labeled_features, reference_props, csv_nodata
    )
    
    # Step 5: Save output rasters
    arcpy.AddMessage("Step 5: Saving CSI rasters...")
    save_csi_rasters(
        csi_arrays, labeled_df_filtered, reference_props, out_raster_folder,
        label_field_names, csv_nodata
    )
    
    arcpy.AddMessage(f"\nPart 2 completed successfully!")
    arcpy.AddMessage(f"Created {len(csi_arrays)} CSI rasters in: {out_raster_folder}")
    
    return True


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
        out_labelled_pairwise_csv = parameters[7].valueAsText
        out_evidence_table_csv = parameters[9].valueAsText if parameters[9].value else None
        out_raster_folder = parameters[10].valueAsText if parameters[10].value else None
        
        # Filter labeled points by field value
        selected_label_field = parameters[11].valueAsText if len(parameters) > 11 and parameters[11].value else None
        
        arcpy.AddMessage("Starting CSI Analysis...")
        arcpy.AddMessage("="*60)
        
        # Load labeled data
        all_df, feature_fields, _ = load_labeled_data(
            labelled_path, label_field_names, feature_field_names
        )
        
        if all_df is None:
            return
        
        # Filter to only labeled points
        label_mask = _rows_with_labels(all_df, label_field_names, csv_nodata)
        labeled_df = all_df.loc[label_mask].reset_index(drop=True)
        
        # Optional: Further filter by selected label field
        if selected_label_field and selected_label_field in labeled_df.columns:
            selected_mask = labeled_df[selected_label_field].notna()
            labeled_df = labeled_df.loc[selected_mask].reset_index(drop=True)
            arcpy.AddMessage(f"Filtered to {len(labeled_df)} points with valid {selected_label_field}")
        
        arcpy.AddMessage(f"Using {len(labeled_df)} labeled points for analysis")

        if len(labeled_df) == 0:
            arcpy.AddError("No labeled rows found - cannot proceed with analysis.")
            return
        
        # Calculate corner CSI matrix
        arcpy.AddMessage("\nCorner CSI Matrix Calculation")
        corner_matrix = calculate_corner_csi(labeled_df, feature_fields, csv_nodata)
        
        # Pixel-to-Label CSI (if rasters provided and output folder specified)
        if evidence_type == "Raster" and rasters_list and out_raster_folder:
            
            coord_1, coord_2 = _detect_xy_columns(labeled_df)
            feature_fields_only = [f for f in feature_fields if f not in (coord_1, coord_2)]
            success = pixel_to_label_csi(
                labeled_df, feature_fields_only, rasters_list, out_raster_folder,
                label_field_names, csv_nodata
            )
            if not success:
                arcpy.AddWarning("Part 2 workflow failed, continuing with Part 1 results only")
        
        # Save CSV results
        arcpy.AddMessage("\nSaving CSV results...")
        evidence_results = {}  # Empty for Part 2 implementation
        save_csv_results(
            corner_matrix, evidence_results,
            out_labelled_pairwise_csv, out_evidence_table_csv
        )
        
        arcpy.AddMessage(f"\nCSI Analysis completed successfully!")
        arcpy.AddMessage(f"Corner matrix shape: {corner_matrix.shape}")
        if evidence_type == "Raster" and rasters_list and out_raster_folder:
            arcpy.AddMessage(f"Created {len(labeled_df)} pixel-to-label CSI rasters")
        
    except Exception as e:
        arcpy.AddError(f"Error in CSI calculation: {e}")
        import traceback
        arcpy.AddError(traceback.format_exc())