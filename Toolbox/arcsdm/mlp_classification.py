import arcpy
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn


from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


import arcsdm.common
import arcsdm.machine_learning.general
import arcsdm.machine_learning.pytorch_utils


class MLPClassifierModel(nn.Module):
    def __init__(self, input_dims, hidden_layers, last_layer):
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

    def forward(self, x):
        return self.layers(x)#.squeeze()

    def get_activation_function(self, name):
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
    input_rasters,
    X_nodata_value,
    standardize,
    target_labels,
    target_labels_attr,
    y_nodata_value,
    hidden_layers,
    last_layer,
    epochs,
    batch_size,
    optimizer,
    learning_rate,
    is_early_stopping,  # TODO: implement early stopping
    early_stopping_patience,
    validation_split,
    validation_data,  # TODO: use validation_data as alternative to validation_split
    validation_metrics,  # TODO: utilize validation metrics selection
    random_state,
    apply_smote,  # TODO: apply smote
    smote_params,
    output_model_file
):
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
        y = arcsdm.machine_learning.generalpick_value(label_arrays, prefer="first")

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
            # If label file is raster, check the nodata value
            # TODO: handle raster label data
            arcpy.AddMessage("Raster data not yet supported.")

    y_mask = np.isnan(y)

    # Check label counts
    unique_labels = np.unique(y[~y_mask])
    if (len(unique_labels) < 2):
        msg = "At least two classes are required for classification."
        arcpy.AddError(msg)
        raise arcsdm.machine_learning.general.MLPInputError(msg)

    # TODO: Verify that target labels are contiguous from 0, ..., K-1
    # (otherwise CrossEntropyLoss can fail)

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

    # Prepare validation data and final training dataset
    if validation_split != 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=random_state, shuffle=True)
    elif validation_data is not None:
        # TODO: handle reading X_test and y_test from validation data and remove message below
        X_test = X_train.copy()  # TODO: remove once validation data is read from file
        y_test = y_test.copy()  # TODO: remove once validation data is read from file
        arcpy.AddMessage("Validation data is not supported")
        X_train = X
        y_train = y

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

    # TODO: add warnings if sigmoid used with more classes than what's allowed

    last_layer = (target_label_count, last_layer_activation, last_layer_dropout)

    model = MLPClassifierModel(
        input_dims=X_train.shape[1],
        hidden_layers=hidden_layers,
        last_layer=last_layer
    )
    model.to(device)

    # TODO: remove
    arcpy.AddMessage(f"Initialized MLP classifier model: {model}")

    optimizer = arcsdm.machine_learning.pytorch_utils.get_pytorch_optimizer(optimizer, model.parameters(), learning_rate)

    # TODO: consider adding better check for binary classification.
    # (currently just tried to keep the number of various states that are checked minimal)

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

    # TODO: implement early stopping

    best_val_loss = None
    best_model_wts = None
    train_loss_dict = {}
    val_loss_dict = {}

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

        if (best_val_loss is None) or (current_val_loss < best_val_loss):
            best_val_loss = current_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}: "
            f"train loss: {train_ret['loss']:.6f}, "
            f"train accuracy: {train_ret['accuracy']:.2%}, "
            f"val loss: {val_ret['loss']:.6f}, "
            f"val accuracy: {val_ret['accuracy']:.2%}")

    # TODO: Add metrics selected by user to messages

    # Plot loss curve & save to file
    output_dir = arcpy.mp.ArcGISProject("CURRENT").homeFolder
    if output_dir and output_dir.lower().endswith(".gdb"):
        output_dir = os.path.dirname(output_dir)

    if not output_dir:
        raise arcsdm.machine_learning.general.MLPError("Could not determine output folder for saving plot output.")

    png_path = arcpy.CreateUniqueName("training_vs_validation_loss.png", output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))  # TODO: calculate a good ratio for figsize
    arcsdm.machine_learning.general.plot_loss_curves(
        ax=ax,
        epochs=epochs,  # TODO: remember to use updated value for epochs once early stopping is implemented
        train_loss_dict=train_loss_dict,
        val_loss_dict=val_loss_dict
    )
    fig.savefig(png_path)
    arcpy.AddMessage(f"Loss curve saved to {png_path}")

    epoch_with_min_loss = min(train_loss_dict, key=train_loss_dict.get)

    arcpy.AddMessage(f"Epoch with smallest loss: {epoch_with_min_loss}")

    # Save best model weights and metadata

    arcpy.AddMessage("Saving best model...")
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

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
        "standardize": bool(standardize),
        "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
        "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
    }

    torch.save(model.state_dict(), output_model_file)

    metadata_file = f"{os.path.splitext(output_model_file)[0]}.meta.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    arcpy.AddMessage(f"Saved model weights to {output_model_file}")
    arcpy.AddMessage(f"Saved model metadata to {metadata_file}")


@arcsdm.common.gp_tool
def test_MLP_classifier(
    input_rasters,
    X_nodata_value,
    standardize,
    target_labels,
    target_labels_attr,
    y_nodata_value,
    model_file,
    classification_threshold,
    output_raster_prob,
    output_raster_classified,
    test_metrics
):
    arcpy.AddMessage("Starting MLP classifier test...")
    device = arcsdm.machine_learning.pytorch_utils.get_device()
    arcpy.AddMessage(f"Device is: {device}")
    return None


@arcsdm.common.gp_tool
def predict_with_MLP_classifier(
    input_rasters,
    X_nodata_value,
    standardize,
    model_file,
    classification_threshold,
    output_raster_prob,
    output_raster_classified
):
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
    # TODO: consider adding batch_size as UI parameter, default batch size is 1
    # pred_loader = DataLoader(pred_dataset, batch_size=batch_size)
    pred_loader = DataLoader(pred_dataset, batch_size=1024)

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
