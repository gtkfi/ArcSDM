import copy
import json
import os
from typing import Optional, Sequence, Tuple

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import arcsdm.common
import arcsdm.machine_learning.general
import arcsdm.machine_learning.pytorch_utils
import arcsdm.smote

from arcsdm.mlp.classification.model import MLPClassifierModel
from arcsdm.mlp.classification.metrics import classification_metric_value
from arcsdm.mlp.classification.types import HiddenLayerSpec, LastLayerConfig
from utils.arcpy_callback import ArcPyLoggingCallback


NUMERIC_FIELD_TYPES = ["SmallInteger", "Integer", "Single", "Double"]
CLASSIFIER_LAST_LAYER_ACTIVATIONS = ["linear", "sigmoid", "softmax"]


def _source_label_mapping(target_labels: Sequence[str], target_labels_attr: Optional[str]) -> Optional[dict]:
    if len(target_labels) > 1:
        return {str(i): str(target_labels[i]) for i in range(len(target_labels))}

    if target_labels_attr is None:
        return None

    target_desc = arcpy.Describe(target_labels[0]).dataType
    if target_desc not in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
        return None

    fields = arcpy.ListFields(target_labels[0], target_labels_attr)
    if not fields or fields[0].type in ["Integer", "SmallInteger", "Float", "Double"]:
        return None

    values = [row[0] for row in arcpy.da.SearchCursor(target_labels[0], [target_labels_attr])]
    encoder = LabelEncoder()
    encoder.fit(values)
    return {
        str(int(code)): str(label)
        for code, label in zip(encoder.transform(encoder.classes_), encoder.classes_)
    }


def _metadata_label_mapping(unique_labels: np.ndarray, label_to_index: dict, source_mapping: Optional[dict]) -> dict:
    metadata_mapping = {}
    for label in unique_labels.tolist():
        model_index = label_to_index[float(label)]
        label_key = str(int(label)) if float(label).is_integer() else str(float(label))
        metadata_mapping[str(model_index)] = str(source_mapping.get(label_key, label_key)) if source_mapping else label_key
    return metadata_mapping


def _validation_label_to_index(label, label_to_index: dict, source_label_to_index: dict) -> int:
    try:
        return label_to_index[float(label)]
    except (KeyError, TypeError, ValueError):
        label_key = str(label)
        if label_key in source_label_to_index:
            return source_label_to_index[label_key]
        raise KeyError(label)


def _validate_classifier_last_layer(last_layer_activation: Optional[str], class_count: int) -> str:
    activation = str(last_layer_activation or "linear").lower().strip()
    if activation not in CLASSIFIER_LAST_LAYER_ACTIVATIONS:
        msg = f"Unsupported classifier last-layer activation: {last_layer_activation}."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    if class_count > 2 and activation == "sigmoid":
        msg = "Sigmoid output is only supported for binary classification. Use Linear or Softmax for multiclass classification."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    if class_count > 1 and activation == "softmax":
        arcpy.AddWarning("PyTorch CrossEntropyLoss expects raw logits; Linear is recommended over Softmax for multiclass output.")

    return activation


@arcsdm.common.gp_tool
def train_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    target_labels: Sequence[str],
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    hidden_layers: Sequence[HiddenLayerSpec],
    last_layer: LastLayerConfig,
    epochs: int,
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    is_early_stopping: bool,
    early_stopping_patience: Optional[int],
    validation_split: Optional[float],
    validation_data: Optional[str],
    validation_metrics: Optional[str],
    random_state: Optional[int],
    apply_smote: bool,
    smote_params: Optional[Tuple[Optional[int], int, int]],
    output_model_file: str
) -> None:
    """Train a multi-layer perceptron (MLP) classifier using PyTorch.
    
    Parameters:
        input_rasters: List of file paths to input feature rasters.
        X_nodata_value: NoData value to apply to input features, or None to use existing NoData.
        standardize: Whether to standardize features to zero mean and unit variance.
        target_labels: List of file paths to target label rasters or vectors, or a single path for binary classification.
        target_labels_attr: If target_labels contains vector data, the attribute field to use for labels
        y_nodata_value: NoData value to apply to target labels, or None to use existing NoData.
        hidden_layers: Specification of hidden layers (units and activation).
        last_layer: Specification of last layer (units, activation, dropout).
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        optimizer: Optimizer to use (e.g. "adam", "sgd").
        learning_rate: Learning rate for the optimizer.
        is_early_stopping: Whether to use early stopping.
        early_stopping_patience: Number of epochs with no improvement to wait before stopping.
        validation_split: Fraction of training data to use for validation.
        validation_data: Path to validation data.
        validation_metrics: Metrics to evaluate on validation data.
        random_state: Random seed for reproducibility.
        apply_smote: Whether to apply SMOTE for imbalanced data.
        smote_params: Parameters for SMOTE (k_neighbors, sampling_strategy, random_state).
        output_model_file: Path to save the trained model.
    """
    arcpy.AddMessage("Starting MLP classifier training...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    grids = [arcsdm.machine_learning.general.describe_raster_grid(p) for p in input_rasters]
    if not arcsdm.machine_learning.general.check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    ref_raster = grids[0]["path"]
    source_mapping = _source_label_mapping(target_labels, target_labels_attr)

    if len(target_labels) > 1:
        mapping = dict()
        label_arrays = []

        for i in range(len(target_labels)):
            label_array = arcsdm.machine_learning.general.rasterize_vector_to_array(
                vector_path=target_labels[i],
                ref_path=ref_raster,
                value_field=None,
                const=i + 1
            )

            mapping[str(int(i))] = str(target_labels[i])
            label_arrays.append(label_array)

        y = arcsdm.machine_learning.general.pick_value(label_arrays, prefer="first")

        unique_json = arcpy.CreateUniqueName("mapping.json", arcpy.env.scratchFolder)
        with open(unique_json, "w") as f:
            json.dump(mapping, f, indent=2)
            json_str = json.dumps(mapping, indent=2)

            arcpy.AddMessage(f"Encoded label features and saved mapping to {unique_json}. Mapping: {json_str}")

        del label_arrays
    else:
        if arcpy.Describe(target_labels[0]).dataType in ["FeatureLayer", "FeatureClass", "ShapeFile"]:
            y = arcsdm.machine_learning.general.rasterize_vector_to_array(
                vector_path=target_labels[0],
                ref_path=ref_raster,
                value_field=target_labels_attr,
                const=1
            )
        elif arcpy.Describe(target_labels[0]).dataType in ["RasterLayer", "RasterDataset", "RasterBand"]:
            label_bands = arcsdm.machine_learning.general.raster_to_band_arrays(target_labels[0])
            if len(label_bands) != 1:
                msg = "Target label raster must have exactly one band."
                arcpy.AddError(msg)
                raise arcsdm.machine_learning.general.MLPInputError(msg)

            y = label_bands[0]
            if y_nodata_value is not None and not arcsdm.machine_learning.general.is_nan_like(y_nodata_value):
                arcsdm.machine_learning.general.apply_explicit_nodata_inplace(y, y_nodata_value, np.float32)

    raster_arrays = arcsdm.machine_learning.general.read_raster_bands(
        raster_files=input_rasters,
        nodata_value=X_nodata_value
    )

    mask_2D = arcsdm.machine_learning.general.get_nodata_mask(raster_arrays + [y])
    valid = ~mask_2D.ravel()

    X = np.column_stack([arr.ravel() for arr in raster_arrays])
    y = y.ravel()

    X = X[valid]
    y = y[valid]

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        msg = "At least two classes are required for classification."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    label_to_index = {float(label): idx for idx, label in enumerate(np.sort(unique_labels).tolist())}
    label_mapping = _metadata_label_mapping(unique_labels, label_to_index, source_mapping)
    source_label_to_index = {label: int(idx) for idx, label in label_mapping.items()}
    y = np.asarray([label_to_index[float(label)] for label in y], dtype=np.int64)

    if validation_data is not None:
        usable_fields = [f for f in arcpy.ListFields(validation_data) if f.type in NUMERIC_FIELD_TYPES + ["String"]]
        fields = [f.name for f in usable_fields]
        numeric_fields = [f.name for f in usable_fields if f.type in NUMERIC_FIELD_TYPES]
        if len(fields) < 2:
            msg = "Validation table must contain at least one feature field and one target field."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        y_field = None
        for candidate in ["label", "labels", "target", "class", "y"]:
            if candidate in [f.lower() for f in fields]:
                y_field = fields[[f.lower() for f in fields].index(candidate)]
                break
        if y_field is None:
            y_field = fields[-1]

        x_fields = [f for f in numeric_fields if f != y_field]
        if len(x_fields) != X.shape[1]:
            msg = f"Validation table must contain {X.shape[1]} feature fields, found {len(x_fields)}."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        table_arr = arcpy.da.TableToNumPyArray(validation_data, x_fields + [y_field], skip_nulls=True)
        if len(table_arr) == 0:
            msg = "Validation table has no usable rows."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        X_test = np.column_stack([table_arr[field] for field in x_fields]).astype(np.float32)
        y_test_raw = np.asarray(table_arr[y_field])
        try:
            y_test = np.asarray([
                _validation_label_to_index(label, label_to_index, source_label_to_index)
                for label in y_test_raw
            ], dtype=np.int64)
        except KeyError as exc:
            msg = f"Validation labels contain unseen class: {exc}."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        X_train = X
        y_train = y
    elif validation_split and validation_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=random_state, shuffle=True)
    else:
        arcpy.AddWarning("Validation split was not provided; using default validation_split=0.2")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)

    if apply_smote:
        n_synthetic, minority_class_label, k_neighbors = smote_params if smote_params else (None, 1, 5)
        mapped_minority = label_to_index.get(float(minority_class_label), int(minority_class_label))
        X_train, y_train = arcsdm.smote.smote(
            X_train,
            y_train,
            n_synthetic=n_synthetic,
            minority_class=mapped_minority,
            k_neighbors=int(k_neighbors),
            random_state=random_state
        )
        arcpy.AddMessage("Applied SMOTE to training data.")

    scaler = None
    if standardize:
        X_train, scaler = arcsdm.machine_learning.general.standardize(X_train)
        X_test, _ = arcsdm.machine_learning.general.standardize(X_test, scaler=scaler)
        arcpy.AddMessage("Data was standardized.")

    training_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    testing_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    training_loader = DataLoader(training_dataset, batch_size=batch_size)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size)

    last_layer_activation, last_layer_dropout = last_layer
    last_layer_activation = _validate_classifier_last_layer(last_layer_activation, len(unique_labels))

    if (len(unique_labels) == 2) and (last_layer_activation == "sigmoid"):
        target_label_count = 1
    else:
        target_label_count = len(unique_labels)

    last_layer = (target_label_count, last_layer_activation, last_layer_dropout)

    model = MLPClassifierModel(
        input_dims=X_train.shape[1],
        hidden_layers=hidden_layers,
        last_layer=last_layer
    )
    model.to(device)

    optimizer = arcsdm.machine_learning.pytorch_utils.get_pytorch_optimizer(optimizer, model.parameters(), learning_rate)

    if target_label_count == 1:
        criterion = nn.BCELoss()
        target_dtype = torch.float32
    else:
        criterion = nn.CrossEntropyLoss()
        target_dtype = torch.long

    best_val_loss = None
    best_model_wts = None
    train_loss_dict = {}
    val_loss_dict = {}
    trained_epochs = 0
    patience = max(1, int(early_stopping_patience) if early_stopping_patience is not None else 5)
    stale_epochs = 0
    callback = ArcPyLoggingCallback(epochs)

    callback.on_train_begin()
    try:
        for epoch in range(epochs):
            train_ret = arcsdm.machine_learning.pytorch_utils.train_classifier_epoch(
                device,
                training_loader,
                model,
                criterion,
                optimizer,
                target_dtype=target_dtype,
                binary_classifier=target_label_count == 1
            )
            val_ret = arcsdm.machine_learning.pytorch_utils.evaluate_classifier_epoch(
                device,
                testing_loader,
                model,
                criterion,
                target_dtype=target_dtype,
                binary_classifier=target_label_count == 1
            )

            current_loss = train_ret['loss'].item()
            current_val_loss = val_ret['loss']

            train_loss_dict[epoch + 1] = current_loss
            val_loss_dict[epoch + 1] = current_val_loss
            trained_epochs = epoch + 1

            if (best_val_loss is None) or (current_val_loss < best_val_loss):
                best_val_loss = current_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            callback.on_epoch_end(
                epoch,
                {
                    "train_loss": f"{current_loss:.6f}",
                    "train_accuracy": f"{train_ret['accuracy']:.2%}",
                    "val_loss": f"{current_val_loss:.6f}",
                    "val_accuracy": f"{val_ret['accuracy']:.2%}",
                }
            )

            if is_early_stopping and stale_epochs >= patience:
                arcpy.AddMessage(f"Early stopping at epoch {epoch + 1}.")
                break
    finally:
        callback.on_train_end({"epochs_ran": trained_epochs})

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    if validation_metrics:
        metric = validation_metrics.strip().lower()
        y_true_all = []
        y_pred_all = []
        model.eval()
        with torch.no_grad():
            for data, target in testing_loader:
                data = data.to(device).to(torch.float32)
                output = model(data)
                if target_label_count == 1:
                    pred = output.reshape(-1).round().cpu().numpy().astype(np.int64)
                    true = target.reshape(-1).cpu().numpy().astype(np.int64)
                else:
                    pred = output.argmax(dim=1).cpu().numpy().astype(np.int64)
                    true = target.cpu().numpy().astype(np.int64)
                y_true_all.append(true)
                y_pred_all.append(pred)

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        val_metric = classification_metric_value(metric, y_true, y_pred)
        metric_labels = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "F1",
        }
        if val_metric is not None:
            arcpy.AddMessage(f"Validation {metric_labels[metric]}: {val_metric:.4f}")

    output_dir = arcpy.mp.ArcGISProject("CURRENT").homeFolder
    if output_dir and output_dir.lower().endswith(".gdb"):
        output_dir = os.path.dirname(output_dir)

    if not output_dir:
        raise arcsdm.machine_learning.general.MLPError("Could not determine output folder for saving plot output.")

    png_path = arcpy.CreateUniqueName("training_vs_validation_loss.png", output_dir)
    fig_width = max(8.0, min(16.0, 8.0 + trained_epochs / 25.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    arcsdm.machine_learning.general.plot_loss_curves(
        ax=ax,
        epochs=trained_epochs,
        train_loss_dict=train_loss_dict,
        val_loss_dict=val_loss_dict
    )
    fig.savefig(png_path)
    arcpy.AddMessage(f"Loss curve saved to {png_path}")

    epoch_with_min_loss = min(train_loss_dict, key=train_loss_dict.get)

    arcpy.AddMessage(f"Epoch with smallest loss: {epoch_with_min_loss}")
    arcpy.AddMessage("Saving best model...")

    output_dirname = os.path.dirname(output_model_file)
    if output_dirname and not os.path.exists(output_dirname):
        os.makedirs(output_dirname, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "model_type": "mlp_classifier",
        "input_dims": int(X_train.shape[1]),
        "hidden_layers": hidden_layers,
        "last_layer": last_layer,
        "target_label_count": int(target_label_count),
        "unique_labels": [float(x) for x in unique_labels.tolist()],
        "label_mapping": label_mapping,
        "batch_size": int(batch_size),
        "standardize": bool(standardize),
        "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
        "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "trained_epochs": int(trained_epochs),
    }

    torch.save(model.state_dict(), output_model_file)

    metadata_file = f"{os.path.splitext(output_model_file)[0]}.meta.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    arcpy.AddMessage(f"Saved model weights to {output_model_file}")
    arcpy.AddMessage(f"Saved model metadata to {metadata_file}")
