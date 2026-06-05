import copy
import json
import os
from typing import Optional, Sequence, Tuple

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import arcsdm.common
import arcsdm.machine_learning.general
import arcsdm.machine_learning.pytorch_utils
import arcsdm.smote

from arcsdm.mlp.regression.data import read_regressor_target_array, validate_regressor_input_rasters
from arcsdm.mlp.regression.metrics import log_regression_validation_metric
from arcsdm.mlp.regression.model import MLPRegressorModel
from arcsdm.mlp.regression.types import HiddenLayerSpec, LastLayerConfig
from utils.arcpy_callback import ArcPyLoggingCallback


@arcsdm.common.gp_tool
def train_MLP_regressor(
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
    loss_function: str,
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
    """Train a Multilayer Perceptron (MLP) regression model using PyTorch.
    
    Parameters:
        input_rasters: List of file paths to input feature rasters.
        X_nodata_value: NoData value to apply to input features, or None to use existing NoData.
        standardize: Whether to standardize features to zero mean and unit variance.
        target_labels: List of file paths to target label rasters or vectors, or a single path for binary classification.
        target_labels_attr: If target_labels contains vector data, the attribute field to use for labels
        y_nodata_value: NoData value to apply to target labels, or None to use existing NoData.
        hidden_layers: Specification of hidden layers (units, activation, dropout).
        last_layer: Activation function of the last layer.
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

    arcpy.AddMessage("Starting MLP regressor training...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    grids = validate_regressor_input_rasters(input_rasters)
    ref_raster = grids[0]["path"]

    y = read_regressor_target_array(
        target_label=target_labels[0],
        target_labels_attr=target_labels_attr,
        y_nodata_value=y_nodata_value,
        ref_raster_path=ref_raster
    )

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

    if validation_data is not None:
        fields = [f.name for f in arcpy.ListFields(validation_data) if f.type in ["SmallInteger", "Integer", "Single", "Double"]]
        if len(fields) < 2:
            msg = "Validation table must contain at least one feature field and one target field."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        y_field = None
        lower_fields = [f.lower() for f in fields]
        for candidate in ["label", "labels", "target", "value", "y"]:
            if candidate in lower_fields:
                y_field = fields[lower_fields.index(candidate)]
                break
        if y_field is None:
            y_field = fields[-1]

        x_fields = [f for f in fields if f != y_field]
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
        y_test = np.asarray(table_arr[y_field], dtype=np.float32)

        X_train = X
        y_train = y
    elif validation_split and validation_split > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=random_state, shuffle=True)
    else:
        arcpy.AddWarning("Validation split was not provided; using default validation_split=0.2")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)

    if apply_smote:
        n_synthetic, minority_class_label, k_neighbors = smote_params if smote_params else (None, 1, 5)
        X_train, y_train = arcsdm.smote.smote(
            X_train,
            y_train,
            n_synthetic=n_synthetic,
            minority_class=minority_class_label,
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

    last_layer_activation = last_layer[0] if isinstance(last_layer, (list, tuple)) else last_layer
    last_layer = (1, last_layer_activation, None)

    model = MLPRegressorModel(
        input_dims=X_train.shape[1],
        hidden_layers=hidden_layers,
        last_layer=last_layer
    )
    model.to(device)

    pytorch_optimizer = arcsdm.machine_learning.pytorch_utils.get_pytorch_optimizer(optimizer, model.parameters(), learning_rate)
    try:
        criterion = arcsdm.machine_learning.pytorch_utils.get_pytorch_regression_loss(loss_function)
    except arcpy.ExecuteError:
        msg = f"Unsupported loss function: {loss_function}"
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    arcpy.AddMessage(f"Using loss function: {loss_function}")

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
            train_loss = arcsdm.machine_learning.pytorch_utils.train_regression_epoch(device, training_loader, model, criterion, pytorch_optimizer)
            val_loss = arcsdm.machine_learning.pytorch_utils.evaluate_regression_epoch(device, testing_loader, model, criterion)

            train_loss_dict[epoch + 1] = train_loss
            val_loss_dict[epoch + 1] = val_loss
            trained_epochs = epoch + 1

            if (best_val_loss is None) or (val_loss < best_val_loss):
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            callback.on_epoch_end(
                epoch,
                {
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                }
            )

            if is_early_stopping and stale_epochs >= patience:
                arcpy.AddMessage(f"Early stopping at epoch {epoch + 1}.")
                break
    finally:
        callback.on_train_end({"epochs_ran": trained_epochs})

    if validation_metrics:
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for data, target in testing_loader:
                data = data.to(device).to(torch.float32)
                pred = model(data).reshape(-1).cpu().numpy()
                y_pred.append(pred)
                y_true.append(target.cpu().numpy().reshape(-1))

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        log_regression_validation_metric(validation_metrics, y_true, y_pred)

    output_dir = arcpy.mp.ArcGISProject("CURRENT").homeFolder
    if output_dir and output_dir.lower().endswith(".gdb"):
        output_dir = os.path.dirname(output_dir)

    if not output_dir:
        raise arcsdm.machine_learning.general.MLPError("Could not determine output folder for saving plot output.")

    png_path = arcpy.CreateUniqueName("training_vs_validation_loss_regressor.png", output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    arcsdm.machine_learning.general.plot_loss_curves(
        ax=ax,
        epochs=trained_epochs,
        train_loss_dict=train_loss_dict,
        val_loss_dict=val_loss_dict
    )
    ax.set_title(f"Training vs Validation Loss ({loss_function})")
    fig.savefig(png_path)
    arcpy.AddMessage(f"Loss curve saved to {png_path}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    output_dirname = os.path.dirname(output_model_file)
    if output_dirname and not os.path.exists(output_dirname):
        os.makedirs(output_dirname, exist_ok=True)

    metadata = {
        "schema_version": 1,
        "model_type": "mlp_regressor",
        "input_dims": int(X_train.shape[1]),
        "hidden_layers": hidden_layers,
        "last_layer": last_layer,
        "loss_function": loss_function,
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

    return None
