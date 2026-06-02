import arcpy
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn


from collections import OrderedDict
from typing import Optional, Sequence, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


import arcsdm.common
import arcsdm.machine_learning.general
import arcsdm.machine_learning.pytorch_utils
import arcsdm.smote

from utils.arcpy_callback import ArcPyLoggingCallback


HiddenLayerSpec = Tuple[int, Optional[str], Optional[float]]
LastLayerConfig = Tuple[Optional[str], Optional[float]]


class MLPClassifierModel(nn.Module):
    def __init__(self, input_dims: int, hidden_layers: Sequence[HiddenLayerSpec], last_layer: HiddenLayerSpec) -> None:
        super(MLPClassifierModel, self).__init__()

        all_layers = hidden_layers + [last_layer]

        layers = []
        layers.append(("lin_input", nn.Linear(in_features=input_dims, out_features=all_layers[0][0])))
        for i in range(len(all_layers) - 1):
            neurons, activation_func, dropout_rate = tuple(all_layers[i])
            next_layer_neurons, _, _ = tuple(all_layers[i + 1])

            if (activation_func is not None) and (self.get_activation_function(activation_func) is not None):
                layers.append((f"a_{i}", self.get_activation_function(activation_func)))
            if (dropout_rate is not None) and (dropout_rate is not 0):
                layers.append((f"do_{i}", nn.Dropout(dropout_rate)))

            layers.append((f"l_{i}", nn.Linear(in_features=neurons, out_features=next_layer_neurons)))

        # Last activation & dropout
        idx = len(all_layers)
        neurons, activation_func, dropout_rate = tuple(all_layers[-1])
        if (activation_func is not None) and (self.get_activation_function(activation_func) is not None):
            layers.append((f"a_{idx}", self.get_activation_function(activation_func)))
        if (dropout_rate is not None) and (dropout_rate is not 0):
            layers.append((f"do_{idx}", nn.Dropout(dropout_rate)))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)#.squeeze()

    def get_activation_function(self, name: str) -> Optional[nn.Module]:
        name = name.lower().strip()
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softmax":
            return nn.Softmax(dim=1)
        else:
            return None


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
    arcpy.AddMessage("Starting MLP classifier training...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    # Validate input rasters
    grids = [arcsdm.machine_learning.general.describe_raster_grid(p) for p in input_rasters]
    if not arcsdm.machine_learning.general.check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    ref_raster = grids[0]["path"]

    # Read label data

    # If more than one vector, multilabel
    if len(target_labels) > 1:
        # Can assume files are vectors, since it's checked in the tool UI
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

        # Combine the encoded arrays into one of size reference raster
        y = arcsdm.machine_learning.general.pick_value(label_arrays, prefer="first")

        # Save label mapping in case user wants to
        unique_json = arcpy.CreateUniqueName("mapping.json", arcpy.env.scratchFolder)
        with open(unique_json, "w") as f:
            json.dump(mapping, f, indent=2)
            json_str = json.dumps(mapping, indent=2)

            arcpy.AddMessage(f"Encoded label features and saved mapping to {unique_json}. Mapping: {json_str}")

        del label_arrays
    else:
        # If the label data is a vector, rasterize it and read as array
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

    # Read input feature rasters
    raster_arrays = arcsdm.machine_learning.general.read_raster_bands(
        raster_files=input_rasters,
        nodata_value=X_nodata_value
    )

    mask_2D = arcsdm.machine_learning.general.get_nodata_mask(raster_arrays + [y])
    valid = ~mask_2D.ravel()

    # Form X data from the rasters
    X = np.column_stack([arr.ravel() for arr in raster_arrays])
    y = y.ravel()

    # Apply mask - drop nan values
    X = X[valid]
    y = y[valid]

    # Remap labels to contiguous integers [0..K-1].
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        msg = "At least two classes are required for classification."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    label_to_index = {float(label): idx for idx, label in enumerate(np.sort(unique_labels).tolist())}
    y = np.asarray([label_to_index[float(label)] for label in y], dtype=np.int64)

    # Prepare validation data and final training dataset
    if validation_data is not None:
        fields = [f.name for f in arcpy.ListFields(validation_data) if f.type in ["SmallInteger", "Integer", "Single", "Double"]]
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
        y_test_raw = np.asarray(table_arr[y_field], dtype=np.float64)
        try:
            y_test = np.asarray([label_to_index[float(label)] for label in y_test_raw], dtype=np.int64)
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

    # Create last layer: get number of output classes
    if (len(unique_labels) == 2) and (last_layer_activation == "sigmoid"):
        # Binary classifier
        target_label_count = 1
    else:
        target_label_count = len(unique_labels)

    if (last_layer_activation == "sigmoid") and (target_label_count > 1):
        arcpy.AddWarning("Sigmoid is recommended for binary classification. Consider Softmax for multiclass.")

    last_layer = (target_label_count, last_layer_activation, last_layer_dropout)

    model = MLPClassifierModel(
        input_dims=X_train.shape[1],
        hidden_layers=hidden_layers,
        last_layer=last_layer
    )
    model.to(device)

    optimizer = arcsdm.machine_learning.pytorch_utils.get_pytorch_optimizer(optimizer, model.parameters(), learning_rate)

    if target_label_count == 1:
        # Binary crossentropy for binary classification
        # criterion = nn.BCEWithLogitsLoss() # This combines a Sigmoid layer and BCELoss - no need to have a Sigmoid final layer in the model
        # (according to docs, it would be preferable)
        criterion = nn.BCELoss()
        target_dtype = torch.float32
    else:
        criterion = nn.CrossEntropyLoss()
        target_dtype = torch.long

    # Training and validation

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
        if metric == "accuracy":
            val_metric = float((y_true == y_pred).mean())
            arcpy.AddMessage(f"Validation accuracy: {val_metric:.4f}")
        elif metric == "precision":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true != 1))
            val_metric = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            arcpy.AddMessage(f"Validation precision: {val_metric:.4f}")
        elif metric == "recall":
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fn = np.sum((y_pred != 1) & (y_true == 1))
            val_metric = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            arcpy.AddMessage(f"Validation recall: {val_metric:.4f}")

    # Plot loss curve & save to file
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

    # Save best model weights and metadata

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


@arcsdm.common.arcsdm.common.gp_tool
def test_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    target_labels: Sequence[str],
    target_labels_attr: Optional[str],
    y_nodata_value: Optional[float],
    model_file: str,
    classification_threshold: float,
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str],
    test_metrics: Optional[str]
) -> None:
    arcpy.AddMessage("Starting MLP classifier test...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")
    return None


@arcsdm.common.arcsdm.common.gp_tool
def predict_with_MLP_classifier(
    input_rasters: Sequence[str],
    X_nodata_value: Optional[float],
    standardize: bool,
    model_file: str,
    classification_threshold: float,
    output_raster_prob: Optional[str],
    output_raster_classified: Optional[str]
) -> None:
    arcpy.AddMessage("Starting prediction with classifier...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")

    # Validate input rasters
    grids = [arcsdm.machine_learning.general.describe_raster_grid(p) for p in input_rasters]
    if not arcsdm.machine_learning.general.check_raster_grids(grids, same_extent=True):
        msg = "Input feature rasters should have same grid properties."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    ref_raster_path = grids[0]["path"]

    metadata_file = f"{os.path.splitext(model_file)[0]}.meta.json"
    if not os.path.exists(metadata_file):
        msg = f"Model metadata file not found: {metadata_file}"
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    input_dims = int(metadata["input_dims"])
    hidden_layers = metadata["hidden_layers"]
    last_layer = metadata["last_layer"]
    target_label_count = int(metadata["target_label_count"])
    model_standardize = bool(metadata.get("standardize", False))

    if bool(standardize) != model_standardize:
        arcpy.AddWarning(
            "Prediction standardize parameter differs from training metadata; "
            "using training metadata settings."
        )

    model = MLPClassifierModel(
        input_dims=input_dims,
        hidden_layers=hidden_layers,
        last_layer=last_layer
    )

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    raster_arrays = arcsdm.machine_learning.general.read_raster_bands(
        raster_files=input_rasters,
        nodata_value=X_nodata_value
    )

    mask_2D = arcsdm.machine_learning.general.get_nodata_mask(raster_arrays)
    valid = ~mask_2D.ravel()

    # Form X data from the rasters
    X = np.column_stack([arr.ravel() for arr in raster_arrays])

    # Apply mask - drop nan values
    X = X[valid]

    if model_standardize:
        scaler_mean = metadata.get("scaler_mean")
        scaler_scale = metadata.get("scaler_scale")

        if (scaler_mean is None) or (scaler_scale is None):
            msg = "Model metadata indicates standardization, but scaler statistics are missing."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
        scaler_scale = np.asarray(scaler_scale, dtype=np.float32)

        if X.shape[1] != scaler_mean.shape[0] or X.shape[1] != scaler_scale.shape[0]:
            msg = "Input feature count does not match scaler statistics in model metadata."
            arcpy.AddError(msg)
            raise arcsdm.machine_learning.general.MLPInputError(msg)

        X = (X - scaler_mean) / scaler_scale
        arcpy.AddMessage("Data was standardized using saved training scaler metadata.")

    if X.shape[1] != input_dims:
        msg = f"Input feature count mismatch. Model expects {input_dims}, got {X.shape[1]}."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    dummy_labels = torch.zeros(X.shape[0], 1) # Not used, but required by DataLoader

    pred_dataset = TensorDataset(torch.from_numpy(X), dummy_labels)
    pred_batch_size = int(metadata.get("batch_size", 1024))
    if pred_batch_size < 1:
        pred_batch_size = 1024
    pred_loader = DataLoader(pred_dataset, batch_size=pred_batch_size)

    predicted = arcsdm.machine_learning.pytorch_utils.predict(device, pred_loader, model)
    predicted_raw = torch.cat(predicted)

    last_layer_activation = str(last_layer[1]).lower().strip() if last_layer and len(last_layer) > 1 and last_layer[1] is not None else None

    if target_label_count == 1:
        if last_layer_activation == "sigmoid":
            predicted_probs = predicted_raw.reshape(-1).cpu().numpy()
        else:
            predicted_probs = torch.sigmoid(predicted_raw).reshape(-1).cpu().numpy()

        class_labels = (predicted_probs >= classification_threshold).astype(np.uint8)
    else:
        if last_layer_activation == "softmax":
            class_prob_matrix = predicted_raw
        else:
            class_prob_matrix = torch.softmax(predicted_raw, dim=1)

        class_prob_np = class_prob_matrix.cpu().numpy()
        predicted_probs = class_prob_np.max(axis=1)
        class_labels = class_prob_np.argmax(axis=1).astype(np.int32)

    height = int(grids[0]["rows"])
    width = int(grids[0]["cols"])

    prob_raster_array = arcsdm.machine_learning.general.reshape_predictions(
        predictions=predicted_probs,
        height=height,
        width=width,
        nodata_mask=mask_2D
    )
    class_raster_array = arcsdm.machine_learning.general.reshape_predictions(
        predictions=class_labels,
        height=height,
        width=width,
        nodata_mask=mask_2D
    )

    desc = arcpy.Describe(ref_raster_path)
    lower_left = arcpy.Point(desc.extent.XMin, desc.extent.YMin)
    x_cell_size = desc.meanCellWidth
    y_cell_size = desc.meanCellHeight

    if output_raster_prob:
        out_prob = arcpy.NumPyArrayToRaster(prob_raster_array, lower_left, x_cell_size, y_cell_size)
        out_prob.save(output_raster_prob)
        arcpy.AddMessage(f"Saved probability raster to {output_raster_prob}")

    if output_raster_classified:
        out_cls = arcpy.NumPyArrayToRaster(class_raster_array, lower_left, x_cell_size, y_cell_size)
        out_cls.save(output_raster_classified)
        arcpy.AddMessage(f"Saved classified raster to {output_raster_classified}")

    return None
