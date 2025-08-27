import os
import math
import csv
import arcpy
import numpy as np
from collections import defaultdict



def _normalize_rows(X):
    # L2-normalize rows; protect against zero vectors
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = np.nan  # zero vectors -> NaN rows
    Xn = X / norms
    return Xn


def _cosine_similarity(A, B=None):
    """
    Row-wise cosine similarity between rows of A and rows of B.
    If B is None: pairwise within A (NxN).
    Assumes numeric arrays; handles NaNs by propagating them.
    """
    if B is None:
        An = _normalize_rows(A)
        # Use nan_to_num to avoid NaNs producing NaN dot-products where possible
        sim = np.nan_to_num(An @ An.T)
        return sim
    else:
        An = _normalize_rows(A)
        Bn = _normalize_rows(B)
        sim = np.nan_to_num(An @ Bn.T)
        return sim


def _read_csv_vectors(path, label_fields, feature_fields_list=None, csv_nodata=None):
    """
    Read labelled vectors from CSV/TXT.
    label_fields: list[str]
    feature_fields_list: list[str] or None (auto-detect features = all non-label columns)
    Returns: (labels: np.ndarray[object], X: np.ndarray[float64], feat_names: list[str])
    """
    import csv, math
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # validate labels
        for lf in (label_fields or []):
            if lf not in headers:
                raise ValueError(f"Label field '{lf}' not found in CSV. Available: {headers}")

        # decide features
        if feature_fields_list is None:
            feature_fields = [h for h in headers if h not in set(label_fields or [])]
        else:
            feature_fields = list(feature_fields_list)
            for fn in feature_fields:
                if fn not in headers:
                    raise ValueError(f"Feature column '{fn}' not found in CSV. Available: {headers}")

        labels = []
        rows = []
        for row in reader:
            if label_fields:
                lab_tuple = tuple(str(row[lf]) for lf in label_fields)
                labels.append(lab_tuple if len(lab_tuple) > 1 else lab_tuple[0])
            else:
                labels.append(None)  # e.g., evidence CSV

            vec = []
            for h in feature_fields:
                v = row.get(h, "")
                try:
                    x = float(v)
                except:
                    x = np.nan
                if (csv_nodata is not None) and (not math.isnan(csv_nodata)) and (x == csv_nodata):
                    x = np.nan
                vec.append(x)
            rows.append(vec)

    X = np.array(rows, dtype="float64")
    return np.array(labels, dtype=object), X, feature_fields


def _read_feature_class_vectors(fc_path, label_fields, feature_fields_list=None):
    """
    Read labelled vectors from a Feature Class/Shapefile.
    label_fields: list[str]
    feature_fields_list: list[str] or None (auto-detect numeric fields except labels)
    Returns: (labels: np.ndarray[object], X: np.ndarray[float64], feat_names: list[str])
    """
    fields = arcpy.ListFields(fc_path)
    field_names = [f.name for f in fields]

    # validate labels
    for lf in (label_fields or []):
        if lf not in field_names:
            raise ValueError(f"Label field '{lf}' not found in feature class. Available: {field_names}")

    # decide features
    if feature_fields_list is None:
        numeric_types = {"Double", "Single", "Integer", "SmallInteger", "Float"}
        feature_fields = [
            f.name for f in fields
            if (f.name not in set(label_fields or [])) and (f.type in numeric_types)
        ]
    else:
        feature_fields = list(feature_fields_list)
        for fn in feature_fields:
            if fn not in field_names:
                raise ValueError(f"Feature field '{fn}' not found in feature class. Available: {field_names}")

    use_fields = (label_fields or []) + feature_fields

    labels = []
    rows = []
    with arcpy.da.SearchCursor(fc_path, use_fields) as cur:
        for rec in cur:
            # first len(label_fields) are labels
            if label_fields:
                lab_vals = tuple(str(rec[i]) for i in range(len(label_fields)))
                labels.append(lab_vals if len(lab_vals) > 1 else lab_vals[0])
                feat_vals = rec[len(label_fields):]
            else:
                labels.append(None)
                feat_vals = rec

            rows.append([float(v) if v is not None else np.nan for v in feat_vals])

    X = np.array(rows, dtype="float64")
    return np.array(labels, dtype=object), X, feature_fields


def _centroids_by_label(labels, X):
    # compute mean vector per unique label (ignoring NaNs)
    label_to_rows = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_rows[lab].append(i)
    uniq = list(label_to_rows.keys())
    centroids = []
    for lab in uniq:
        idx = label_to_rows[lab]
        sub = X[idx, :]
        # nanmean along rows; if all-NaN in a column, fill 0
        c = np.nanmean(sub, axis=0)
        c = np.where(np.isnan(c), 0.0, c)
        centroids.append(c)
    C = np.vstack(centroids)
    return np.array(uniq, dtype=object), C


def _write_csv_matrix(path, row_labels, col_labels, matrix):
    """
    Write a similarity matrix to CSV quickly using vectorized NumPy string ops.
    Handles NaN/inf as blanks.
    """
    arcpy.AddMessage(f"Writing CSV matrix to: {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    m = np.asarray(matrix, dtype="float64")
    mask = np.isfinite(m)

    # Preallocate string array
    out = np.full(m.shape, "", dtype=object)

    # Vectorized formatting of finite entries
    out[mask] = np.char.mod("%.6f", m[mask])

    # Convert to Python list-of-lists for writerows
    out_rows = out.tolist()

    # Prepend row labels to each row
    final_rows = []
    for rlab, row in zip(row_labels, out_rows):
        final_rows.append([rlab] + row)

    # Use big buffer to reduce syscalls
    total_rows = len(final_rows)
    with open(path, "w", newline="", encoding="utf-8", buffering=1024*1024) as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([""] + list(col_labels))
        for i, row in enumerate(final_rows):
            writer.writerow(row)
            if i == total_rows // 4:
                arcpy.AddMessage("25% of rows written...")
            elif i == total_rows // 2:
                arcpy.AddMessage("50% of rows written...")
            elif i == 3 * total_rows // 4:
                arcpy.AddMessage("75% of rows written...")
            arcpy.SetProgressorPosition(i + 1)

    arcpy.AddMessage(f"Finished writing CSV matrix with shape {m.shape}.")


def _stack_rasters(raster_paths):
    """
    raster_paths: list[str]
    Returns: stack(H,W,B), ref_raster, ref_desc
    """
    arrays = []
    ref_desc = None
    ref_raster = None

    for i, rp in enumerate([p for p in raster_paths if p]):
        ras = arcpy.Raster(rp)
        dsc = arcpy.Describe(ras)
        arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float64")
        arrays.append(arr)
        if i == 0:
            ref_desc = dsc
            ref_raster = ras
        else:
            if arr.shape != arrays[0].shape:
                raise ValueError(f"All rasters must have the same shape. '{rp}' differs.")

    if not arrays:
        raise ValueError("No valid rasters provided.")
    stack = np.stack(arrays, axis=2)  # (H,W,B)
    return stack, ref_raster, ref_desc



def _save_similarity_raster(sim_arr, ref_desc, out_path):
    # sim_arr orientation matches RasterToNumPyArray (upper-left origin).
    # NumPyArrayToRaster expects lower-left origin, so flipud before writing.
    ll = arcpy.Point(ref_desc.Extent.XMin, ref_desc.Extent.YMin)
    cellsize = ref_desc.meanCellWidth
    arr_out = np.flipud(sim_arr.astype("float32"))
    out_ras = arcpy.NumPyArrayToRaster(arr_out, ll, cellsize, cellsize, -9999)
    arcpy.management.CopyRaster(out_ras, out_path, pixel_type="32_BIT_FLOAT", nodata_value=-9999)
    return out_path

def _sanitize_name(name):
    # Allow tuple labels: join with underscore
    if isinstance(name, (list, tuple)):
        s = "_".join(str(x) for x in name)
    else:
        s = str(name)
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)
    return safe.strip("_") or "class"


    # ---------------- Execute ----------------
def execute(self, parameters, messages):
    arcpy.AddMessage("Starting Cosine Similarity Index (CSI) computation...")

    labelled_path = parameters[0].valueAsText
    label_field_names = parameters[1].valueAsText.split(';') if parameters[1].valueAsText else []
    # Convert comma-separated string to list (or None)
    feature_field_names = parameters[2].valueAsText.split(';') if parameters[2].valueAsText else None

    evidence_type = parameters[3].valueAsText
    rasters_list = parameters[4].valueAsText.split(';') if parameters[4].valueAsText else []
    evidence_vectors_file = parameters[5].valueAsText if parameters[5].value else None

    out_labelled_pairwise_csv = parameters[6].valueAsText
    out_centroid_matrix_csv = parameters[7].valueAsText
    out_evidence_table_csv = parameters[8].valueAsText if parameters[8].value else None
    out_raster_folder = parameters[9].valueAsText if parameters[9].value else None
    csv_nodata = float(parameters[10].value) if parameters[10].value else None

    # --- Echo inputs ---
    arcpy.AddMessage(f"Labelled data: {labelled_path}")
    arcpy.AddMessage(f"Label fields: {label_field_names if label_field_names else '[none]'}")
    arcpy.AddMessage(f"Feature fields: {feature_field_names if feature_field_names else 'Auto-detect'}")
    arcpy.AddMessage(f"Evidence type: {evidence_type}")
    if evidence_type == "Raster layers":
        arcpy.AddMessage(f"Evidence rasters ({len(rasters_list)}): {rasters_list}")
    else:
        arcpy.AddMessage(f"Evidence vectors file: {evidence_vectors_file}")
    if csv_nodata is not None:
        arcpy.AddMessage(f"CSV NoData sentinel: {csv_nodata}")

    # --- Detect input type (CSV/TXT vs Feature Class/Shapefile) ---
    ext = os.path.splitext(labelled_path)[1].lower()
    if ext in [".csv", ".txt"]:
        labelled_is_fc = False
        arcpy.AddMessage("Detected labelled data as CSV/TXT.")
    elif arcpy.Exists(labelled_path):
        desc = arcpy.Describe(labelled_path)
        if hasattr(desc, "datasetType") and desc.datasetType == "FeatureClass":
            labelled_is_fc = True
            arcpy.AddMessage("Detected labelled data as Feature Class.")
        elif ext == ".shp":
            labelled_is_fc = True
            arcpy.AddMessage("Detected labelled data as Shapefile.")
        else:
            raise ValueError("Input is not recognized as CSV/TXT or a feature class/shapefile.")
    else:
        raise ValueError(f"Input not found: {labelled_path}")

    # -------- Read labelled data --------
    arcpy.AddMessage("Reading labelled data...")
    if labelled_is_fc:
        labels, X, feat_names = _read_feature_class_vectors(labelled_path, label_field_names, feature_field_names)
    else:
        labels, X, feat_names = _read_csv_vectors(labelled_path, label_field_names, feature_field_names, csv_nodata)


    if X.size == 0 or X.shape[1] == 0:
        raise ValueError("No feature columns found for labelled data.")

    arcpy.AddMessage(f"Labelled samples: {len(labels)} | features: {len(feat_names)}")

    # -------- Pairwise CSI for labelled samples --------
    arcpy.AddMessage("Computing pairwise CSI for labelled samples...")
    pairwise = _cosine_similarity(X)
    _write_csv_matrix(
        out_labelled_pairwise_csv,
        [f"row{i}" for i in range(len(labels))],
        [f"row{i}" for i in range(len(labels))],
        pairwise
    )
    arcpy.AddMessage(f"Wrote pairwise CSI matrix: {out_labelled_pairwise_csv}")

    # -------- Class centroids & their CSI matrix --------
    arcpy.AddMessage("Computing class centroids and centroid CSI matrix...")
    class_names, C = _centroids_by_label(labels, X)
    centroid_sim = _cosine_similarity(C)
    _write_csv_matrix(out_centroid_matrix_csv, class_names, class_names, centroid_sim)
    arcpy.AddMessage(f"Wrote centroid CSI matrix: {out_centroid_matrix_csv}")

    # -------- Evidence processing --------
    if evidence_type == "CSV/TXT vectors":
        if not evidence_vectors_file or not out_evidence_table_csv:
            raise ValueError("Provide an evidence vectors file and an output CSV path.")
        arcpy.AddMessage("Reading evidence vectors from CSV/TXT...")
        # Unlabeled evidence rows; enforce same feature order as labelled data
        _ev_labels, EV, _ = _read_csv_vectors(
            evidence_vectors_file,
            label_fields=[],                    # no labels expected for evidence
            feature_fields_list=feat_names,     # enforce same order
            csv_nodata=csv_nodata
        )

        arcpy.AddMessage("Computing CSI of evidence vectors vs class centroids...")
        sim_ev = _cosine_similarity(EV, C)  # (N_evidence x N_classes)

        os.makedirs(os.path.dirname(out_evidence_table_csv) or ".", exist_ok=True)
        with open(out_evidence_table_csv, "w", newline="", encoding="utf-8", buffering=1024*1024) as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["row_index"] + [str(cn) for cn in class_names])
            # Vectorized formatting per row
            mask = np.isfinite(sim_ev)
            out = np.full(sim_ev.shape, "", dtype=object)
            out[mask] = np.char.mod("%.6f", sim_ev[mask])
            for i, row in enumerate(out.tolist()):
                w.writerow([i] + row)
        arcpy.AddMessage(f"Wrote evidence CSI table: {out_evidence_table_csv}")

    elif evidence_type == "Raster layers":
        if (not rasters_list) or (not out_raster_folder):
            raise ValueError("Provide evidence rasters and an output folder for CSI rasters.")
        arcpy.AddMessage(f"Stacking evidence rasters ({len(rasters_list)} bands)...")
        stack, ref_raster, ref_desc = _stack_rasters(rasters_list)  # (H,W,B)

        H, W, B = stack.shape
        if B != len(feat_names):
            arcpy.AddWarning(
                f"Number of rasters ({B}) != number of feature columns in labelled data ({len(feat_names)}). "
                "Assuming the same order/name correspondence."
            )

        # Mask & normalize per-pixel vectors
        mask = np.any(np.isnan(stack), axis=2)
        norms = np.linalg.norm(stack, axis=2, keepdims=True)
        norms[norms == 0] = np.nan
        Rnorm = stack / norms  # (H,W,B), NaNs where invalid

        # Normalize class centroids
        Cn = _normalize_rows(C)  # (K,B)

        # Compute similarity per class and save
        os.makedirs(out_raster_folder, exist_ok=True)
        for ci, cname in enumerate(class_names):
            vec = Cn[ci, :]  # (B,)
            sim = np.nansum(Rnorm * vec.reshape(1, 1, -1), axis=2)  # (H,W)
            sim[mask] = np.nan
            out_name = f"CSI_{_sanitize_name(cname)}"
            out_path = os.path.join(out_raster_folder, out_name)
            _save_similarity_raster(sim, ref_desc, out_path)
            arcpy.AddMessage(f"Wrote CSI raster for class '{cname}': {out_path}")
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}")

    arcpy.AddMessage("CSI computation finished.")


def execute_prototype(self, parameters, messages):
    arcpy.AddMessage("Starting Cosine Similarity Index (CSI) computation...")

    labelled_path = parameters[0].valueAsText
    label_field_names = parameters[1].valueAsText.split(';') if parameters[1].valueAsText else []
    feature_field_names = parameters[2].valueAsText.split(';') if parameters[2].valueAsText else None

    evidence_type = parameters[3].valueAsText
    rasters_list = parameters[4].valueAsText.split(';') if parameters[4].valueAsText else []
    evidence_vectors_file = parameters[5].valueAsText if parameters[5].valueAsText else None

    out_labelled_pairwise_csv = parameters[6].valueAsText
    out_centroid_matrix_csv = parameters[7].valueAsText
    out_evidence_table_csv = parameters[8].valueAsText if parameters[8].value else None
    out_raster_folder = parameters[9].valueAsText if parameters[9].value else None
    csv_nodata = float(parameters[10].value) if parameters[10].value else None


    # --- Detect input type (CSV/TXT vs Feature Class/Shapefile) ---
    ext = os.path.splitext(labelled_path)[1].lower()
    if ext in [".csv", ".txt"]:
        labelled_is_fc = False
    elif arcpy.Exists(labelled_path):
        desc = arcpy.Describe(labelled_path)
        if hasattr(desc, "datasetType") and desc.datasetType == "FeatureClass":
            labelled_is_fc = True
        elif ext == ".shp":
            labelled_is_fc = True
        else:
            raise ValueError("Input is not recognized as CSV/TXT or a feature class/shapefile.")
    else:
        raise ValueError(f"Input not found: {labelled_path}")

    # -------- Read labelled data --------
    if labelled_is_fc:
        labels, X, feat_names = _read_feature_class_vectors(labelled_path, label_field_names, feature_field_names)
    else:
        labels, X, feat_names = _read_csv_vectors(labelled_path, label_field_names, feature_field_names, csv_nodata)

    if X.size == 0 or X.shape[1] == 0:
        raise ValueError("No feature columns found for labelled data.")

    arcpy.AddMessage(f"Labelled samples: {len(labels)} | features: {len(feat_names)}")

    # -------- Pairwise CSI for labelled samples --------
    arcpy.AddMessage("Computing pairwise CSI for labelled samples...")
    pairwise = _cosine_similarity(X)
    _write_csv_matrix(out_labelled_pairwise_csv,
                            [f"row{i}" for i in range(len(labels))],
                            [f"row{i}" for i in range(len(labels))],
                            pairwise)
    arcpy.AddMessage(f"Wrote pairwise CSI matrix: {out_labelled_pairwise_csv}")

    # -------- Class centroids & their CSI matrix --------
    arcpy.AddMessage("Computing class centroids and centroid CSI matrix...")
    class_names, C = _centroids_by_label(labels, X)
    centroid_sim = _cosine_similarity(C)
    _write_csv_matrix(out_centroid_matrix_csv, class_names, class_names, centroid_sim)
    arcpy.AddMessage(f"Wrote centroid CSI matrix: {out_centroid_matrix_csv}")

    # -------- Evidence processing --------
    if evidence_type == "CSV/TXT vectors":
        if not evidence_vectors_file or not out_evidence_table_csv:
            raise ValueError("Provide an evidence vectors file and an output CSV path.")
        arcpy.AddMessage("Reading evidence vectors from CSV/TXT...")
        # Reuse feature column names; if header differs, we still consume all numeric columns except label_field (if present)
        # For evidence CSV we ignore any label column; treat every row as an unlabeled vector
        ev_labels, EV, ev_cols = _read_csv_vectors(
            evidence_vectors_file,
            label_field=label_field_names[0] if label_field_names else None,
            feature_fields=",".join(feat_names),  # enforce same feature order as labelled data
            csv_nodata=csv_nodata,
        )
        # Note: _read_csv_vectors requires a label column name; to avoid coupling, we do a lightweight CSV read here:
        # But to keep things consistent and robust, implement a simple reader inline:
        EV = []
        used_cols = None
        with open(evidence_vectors_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            hdrs = reader.fieldnames or []
            for h in feat_names:
                if h not in hdrs:
                    raise ValueError(f"Evidence CSV missing feature '{h}' expected from labelled data.")
            used_cols = feat_names
            for row in reader:
                vec = []
                for h in used_cols:
                    v = row.get(h, "")
                    try:
                        x = float(v)
                    except:
                        x = np.nan
                    if csv_nodata is not None and not math.isnan(csv_nodata) and x == csv_nodata:
                        x = np.nan
                    vec.append(x)
                EV.append(vec)
        EV = np.array(EV, dtype="float64")

        arcpy.AddMessage("Computing CSI of evidence vectors vs class centroids...")
        sim_ev = _cosine_similarity(EV, C)  # (N_evidence x N_classes)
        os.makedirs(os.path.dirname(out_evidence_table_csv) or ".", exist_ok=True)
        with open(out_evidence_table_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row_index"] + [str(cn) for cn in class_names])
            for i, row in enumerate(sim_ev):
                w.writerow([i] + ["{:.6f}".format(float(x)) if np.isfinite(x) else "" for x in row])
        arcpy.AddMessage(f"Wrote evidence CSI table: {out_evidence_table_csv}")

    elif evidence_type == "Raster layers":
        if (not rasters_list) or (not out_raster_folder):
            raise ValueError("Provide evidence rasters and an output folder for CSI rasters.")
        raster_paths = [raster_path for raster_path in rasters_list]
        arcpy.AddMessage(f"Stacking evidence rasters ({len(raster_paths)} bands)...")
        stack, ref_raster, ref_desc = _stack_rasters(rasters_list)  # (H,W,B)

        H, W, B = stack.shape
        if B != len(feat_names):
            arcpy.AddWarning(f"Number of rasters ({B}) != number of feature columns in labelled data ({len(feat_names)}). "
                              "Assuming the same order/name correspondence.")

        # Build a mask where any band is NaN
        mask = np.any(np.isnan(stack), axis=2)

        # Normalize per-pixel vectors
        norms = np.linalg.norm(stack, axis=2, keepdims=True)
        norms[norms == 0] = np.nan
        Rnorm = stack / norms  # (H,W,B), NaNs where invalid

        # Normalize class centroids
        Cn = _normalize_rows(C)  # (K,B)

        # Compute similarity per class: sum over bands of Rnorm * centroid_b
        os.makedirs(out_raster_folder, exist_ok=True)
        for ci, cname in enumerate(class_names):
            vec = Cn[ci, :]  # (B,)
            # Broadcast and compute dot product across band axis
            sim = np.nansum(Rnorm * vec.reshape(1, 1, -1), axis=2)
            # Set masked pixels to NoData
            sim[mask] = np.nan
            out_name = "CSI_{}".format(_sanitize_name(cname))
            out_path = os.path.join(out_raster_folder, out_name)
            _save_similarity_raster(sim, ref_desc, out_path)
            arcpy.AddMessage(f"Wrote CSI raster for class '{cname}': {out_path}")
    else:
        raise ValueError(f"Unknown evidence type: {evidence_type}")

    arcpy.AddMessage("CSI computation finished.")