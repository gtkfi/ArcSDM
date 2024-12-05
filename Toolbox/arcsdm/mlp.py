from numbers import Number

import sys
import arcpy
import numpy as np
from typing import Literal, Optional, Sequence, Tuple
from tensorflow import keras
from keras.metrics import CategoricalCrossentropy, MeanAbsoluteError, MeanSquaredError, Precision, Recall
from keras.layers import Flatten
from keras.optimizers import SGD, Adagrad, Adam, RMSprop

from arcsdm.machine_learning.general import prepare_data_for_ml, save_model
from utils.arcpy_callback import ArcPyLoggingCallback


def _keras_optimizer(optimizer: str, **kwargs):
    """Create a Keras optimizer from given name and parameters."""
    if optimizer == "adam":
        return Adam(**kwargs)
    elif optimizer == "adagrad":
        return Adagrad(**kwargs)
    elif optimizer == "rmsprop":
        return RMSprop(**kwargs)
    elif optimizer == "sgd":
        return SGD(**kwargs)
    else:
        arcpy.AddError(f"Unidentified optimizer: {optimizer}")
        raise arcpy.ExecuteError


def _keras_metric(metric_name: str):
    if metric_name.lower() == "accuracy":
        return "accuracy"
    elif metric_name.lower() == "precision":
        return Precision(name="precision")
    elif metric_name.lower() == "recall":
        return Recall(name="recall")
    elif metric_name.lower() == "categorical_crossentropy":
        return CategoricalCrossentropy(name="categorical_crossentropy")
    elif metric_name.lower() == "mse":
        return MeanSquaredError(name="mse")
    elif metric_name.lower() == "mae":
        return MeanAbsoluteError(name="mae")
    else:
        arcpy.AddError(f"Unsupported metric for Keras model: {metric_name}")
        raise arcpy.ExecuteError


def _check_MLP_inputs(
    neurons: Sequence[int],
    validation_split: Optional[float],
    learning_rate: float,
    dropout_rate: Optional[Number],
    es_patience: int,
    batch_size: int,
    epochs: int,
    output_neurons: int,
    loss_function: str,
) -> None:
    """Check parameters for Keras MLP training."""
    if len(neurons) == 0:
        arcpy.AddError("Neurons parameter must be a non-empty list.")
        raise arcpy.ExecuteError

    if any(neuron < 1 for neuron in neurons):
        arcpy.AddError("Each neuron in neurons list must be at least 1.")
        raise arcpy.ExecuteError

    if validation_split and not (0.0 < validation_split < 1.0):
        arcpy.AddError("Validation split must be a value between 0 and 1, exclusive.")
        raise arcpy.ExecuteError

    if learning_rate <= 0.0:
        arcpy.AddError("Learning rate must be greater than 0.")
        raise arcpy.ExecuteError

    if dropout_rate and not (0.0 <= dropout_rate <= 1.0):
        arcpy.AddError("Dropout rate must be between 0 and 1, inclusive.")
        raise arcpy.ExecuteError

    if es_patience <= 0:
        arcpy.AddError("Early stopping patience must be greater than 0.")
        raise arcpy.ExecuteError

    if batch_size <= 0:
        arcpy.AddError("Batch size must be greater than 0.")
        raise arcpy.ExecuteError

    if epochs <= 0:
        arcpy.AddError("Number of epochs must be greater than 0.")
        raise arcpy.ExecuteError

    if output_neurons <= 0:
        arcpy.AddError("Number of output neurons must be greater than 0.")
        raise arcpy.ExecuteError

    if output_neurons > 1 and loss_function == "binary_crossentropy":
        arcpy.AddError("Number of output neurons must be 1 when used loss function is binary crossentropy.")
        raise arcpy.ExecuteError

    if output_neurons <= 2 and loss_function == "categorical_crossentropy":
        arcpy.AddError("Number of output neurons must be greater than 2 when used loss function is categorical crossentropy.")
        raise arcpy.ExecuteError


def _check_ML_model_data_input(X: np.ndarray, y: np.ndarray):
    """Check if the input data for the ML model is in the correct shape."""
    if X.ndim != 2:
        arcpy.AddError(f"X must be a 2-dimensional array, but is an array with shape {X.shape}.")
        raise arcpy.ExecuteError

    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0]

    if n_samples_X != n_samples_y:
        arcpy.AddError(f"The number of samples in X and y must be equal, but got {n_samples_X} in X and {n_samples_y} in y.")
        raise arcpy.ExecuteError


def train_MLP_classifier(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    activation: Literal["relu", "linear", "sigmoid", "tanh"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["sigmoid", "softmax"] = "sigmoid",
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam", "adagrad", "rmsprop", "sdg"] = "adam",
    learning_rate: Number = 0.001,
    loss_function: Literal["binary_crossentropy", "categorical_crossentropy"] = "binary_crossentropy",
    dropout_rate: Optional[Number] = None,
    early_stopping: Optional[bool] = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["accuracy", "precision", "recall"]]] = ["accuracy"],
    random_state: Optional[int] = None,
) -> Tuple[keras.Model, dict]:
    """
    Train MLP (Multilayer Perceptron) classifier using Keras.

    Creates a Sequential model with Dense NN layers. For each element in `neurons`, Dense layer with corresponding
    dimensionality/neurons is created with the specified activation function (`activation`). If `dropout_rate` is
    specified, a Dropout layer is added after each Dense layer.

    Parameters default to a binary classification model using sigmoid as last activation, binary crossentropy as loss
    function and 1 output neuron/unit.

    For more information about Keras models, read the documentation here: https://keras.io/.

    Args:
        X: Input data. Should be a 2-dimensional array where each row represents a sample and each column a
            feature. Features should ideally be normalized or standardized.
        y: Target labels. For binary classification, y should be a 1-dimensional array of binary labels (0 or 1).
            For multi-class classification, y should be a 2D array with one-hot encoded labels. The number of columns
            should match the number of classes.
        neurons: Number of neurons in each hidden layer.
        validation_split: Fraction of data used for validation during training. Value must be > 0 and < 1 or None.
            Defaults to 0.2.
        validation_data: Separate dataset used for validation during training. Overrides validation_split if
            provided. Expected data form is (X_valid, y_valid). Defaults to None.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer. Defaults to 1.
        last_activation: Activation function used in the output layer. Defaults to 'sigmoid'.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Value must be > 0. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'binary_crossentropy'.
        dropout_rate: Fraction of the input units to drop. Value must be >= 0 and <= 1. Defaults to None.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['accuracy'].
        random_state: Seed for random number generation. Sets Python, Numpy and Tensorflow seeds to make
            program deterministic. Defaults to None (random state / seed).

    Returns:
        Trained MLP model and training history.

    Raises:
        Error: Some of the numeric parameters have invalid values.
        Error: Shape of X or y is invalid.
    """
    # 1. Check input data
    arcpy.AddMessage("Checking input data...")
    _check_ML_model_data_input(X=X, y=y)
    _check_MLP_inputs(
        neurons=neurons,
        validation_split=validation_split,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        es_patience=es_patience,
        batch_size=batch_size,
        epochs=epochs,
        output_neurons=output_neurons,
        loss_function=loss_function,
    )

    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    # 2. Create and compile a sequential model
    arcpy.AddMessage("Creating and compiling a sequential model...")
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(X.shape[1],)))

    arcpy.AddMessage("Adding hidden layers...")
    for neuron in neurons:
        model.add(keras.layers.Dense(units=neuron, activation=activation))

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(units=output_neurons, activation=last_activation))

    model.add(Flatten())

    arcpy.AddMessage("Compiling the model...")
    model.compile(
        optimizer=_keras_optimizer(optimizer, learning_rate=learning_rate),
        loss=loss_function,
        metrics=[_keras_metric(metric) for metric in metrics],
    )
    
    # 3. Train the model
    # Early stopping callback
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []
    callbacks.append(ArcPyLoggingCallback(epochs))
    
    arcpy.AddMessage("Training the model...")
    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=validation_split if validation_split else 0.0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history


def Execute_MLP_classifier(self, parameters, messages):
    
    try:
        input_rasters = parameters[0].valueAsText.split(';')
        target_labels = parameters[1].valueAsText
        nodata_value = parameters[2].value
        neurons = [int(n) for n in parameters[3].valueAsText.split(',')]
        validation_split = float(parameters[4].value) if parameters[4].value else 0.2
        validation_data = parameters[5].valueAsText if parameters[5].valueAsText else None
        activation = parameters[6].valueAsText
        output_neurons = parameters[7].value
        last_activation = parameters[8].valueAsText
        epochs = parameters[9].value
        batch_size = int(parameters[10].value)
        optimizer = parameters[11].valueAsText
        learning_rate = float(parameters[12].value)
        loss_function = parameters[13].valueAsText
        dropout_rate = float(parameters[14].value) if parameters[14].value else None
        early_stopping = parameters[15].value
        es_patience = int(parameters[16].value)
        metrics = parameters[17].valueAsText.split(',')
        random_state = int(parameters[18].value) if parameters[18].value else None
        output_file = parameters[19].valueAsText
        
        arcpy.AddMessage("Starting MLP classifier training...")
        
        X, y, _ = prepare_data_for_ml(input_rasters, target_labels, nodata_value)

        arcpy.AddMessage("Data preparation completed.")

        model, history = train_MLP_classifier(
            X=X,
            y=y,
            neurons=neurons,
            validation_split=validation_split,
            validation_data=validation_data,
            activation=activation,
            output_neurons=output_neurons,
            last_activation=last_activation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss_function=loss_function,
            dropout_rate=dropout_rate,
            early_stopping=early_stopping,
            es_patience=es_patience,
            metrics=metrics,
            random_state=random_state,
        )
        
        arcpy.AddMessage("="*5 + "Model training completed." + "="*5)
        arcpy.AddMessage(f"Saving model to {output_file}.joblib")
        arcpy.AddMessage(f"Model training history:")
        arcpy.AddMessage(f"{history.history}")
        
        save_model(model, output_file)

    # Return geoprocessing specific errors
    except arcpy.ExecuteError:    
        arcpy.AddError(arcpy.GetMessages(2))    

    # Return any other type of error
    except:
        # By default any other errors will be caught here
        e = sys.exc_info()[1]
        print(e.args[0])


def Execute_MLP_regressor(self, parameters, messages):
    
    try:
        input_rasters = parameters[0].valueAsText.split(';')
        target_labels = parameters[1].valueAsText
        nodata_value = parameters[2].value
        neurons = [int(n) for n in parameters[3].valueAsText.split(',')]
        validation_split = float(parameters[4].value) if parameters[4].value else 0.2
        validation_data = parameters[5].valueAsText if parameters[5].valueAsText else None
        activation = parameters[6].valueAsText
        output_neurons = parameters[7].value
        last_activation = parameters[8].valueAsText
        epochs = parameters[9].value
        batch_size = int(parameters[10].value)
        optimizer = parameters[11].valueAsText
        learning_rate = float(parameters[12].value)
        loss_function = parameters[13].valueAsText
        dropout_rate = float(parameters[14].value) if parameters[14].value else None
        early_stopping = parameters[15].value
        es_patience = int(parameters[16].value)
        metrics = parameters[17].valueAsText.split(',')
        random_state = int(parameters[18].value) if parameters[18].value else None
        output_file = parameters[19].valueAsText
        
        arcpy.AddMessage("Starting MLP classifier training...")
        
        X, y, _ = prepare_data_for_ml(input_rasters, target_labels, nodata_value)

        arcpy.AddMessage("Data preparation completed.")

        model, history = train_MLP_regressor(
            X=X,
            y=y,
            neurons=neurons,
            validation_split=validation_split,
            validation_data=validation_data,
            activation=activation,
            output_neurons=output_neurons,
            last_activation=last_activation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss_function=loss_function,
            dropout_rate=dropout_rate,
            early_stopping=early_stopping,
            es_patience=es_patience,
            metrics=metrics,
            random_state=random_state,
        )
        
        arcpy.AddMessage("="*5 + "Model training completed." + "="*5)
        arcpy.AddMessage(f"Saving model to {output_file}.joblib")
        arcpy.AddMessage(f"Model training history:")
        arcpy.AddMessage(f"{history.history}")
        
        save_model(model, output_file)

    # Return geoprocessing specific errors
    except arcpy.ExecuteError:    
        arcpy.AddError(arcpy.GetMessages(2))    

    # Return any other type of error
    except:
        # By default any other errors will be caught here
        e = sys.exc_info()[1]
        print(e.args[0])


def train_MLP_regressor(
    X: np.ndarray,
    y: np.ndarray,
    neurons: Sequence[int],
    validation_split: Optional[float] = 0.2,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    activation: Literal["relu", "linear", "sigmoid", "tanh"] = "relu",
    output_neurons: int = 1,
    last_activation: Literal["linear"] = "linear",
    epochs: int = 50,
    batch_size: int = 32,
    optimizer: Literal["adam", "adagrad", "rmsprop", "sdg"] = "adam",
    learning_rate: Number = 0.001,
    loss_function: Literal["mse", "mae", "hinge", "huber"] = "mse",
    dropout_rate: Optional[Number] = None,
    early_stopping: bool = True,
    es_patience: int = 5,
    metrics: Optional[Sequence[Literal["mse", "rmse", "mae", "r2"]]] = ["mse"],
    random_state: Optional[int] = None,
) -> Tuple[keras.Model, dict]:
    """
    Train MLP (Multilayer Perceptron) regressor using Keras.

    Creates a Sequential model with Dense NN layers. For each element in `neurons`, Dense layer with corresponding
    dimensionality/neurons is created with the specified activation function (`activation`). If `dropout_rate` is
    specified, a Dropout layer is added after each Dense layer.

    For more information about Keras models, read the documentation here: https://keras.io/.

    Args:
        X: Input data. Should be a 2-dimensional array where each row represents a sample and each column a
            feature. Features should ideally be normalized or standardized.
        y: Target labels. Should be a 1-dimensional array where each entry corresponds to the continuous
            target value for the respective sample in X.
        neurons: Number of neurons in each hidden layer.
        validation_split: Fraction of data used for validation during training. Value must be > 0 and < 1 or None.
            Defaults to 0.2.
        validation_data: Separate dataset used for validation during training. Overrides validation_split if
            provided. Expected data form is (X_valid, y_valid). Defaults to None.
        activation: Activation function used in each hidden layer. Defaults to 'relu'.
        output_neurons: Number of neurons in the output layer. Defaults to 1.
        last_activation: Activation function used in the output layer. Defaults to 'linear'.
        epochs: Number of epochs to train the model. Defaults to 50.
        batch_size: Number of samples per gradient update. Defaults to 32.
        optimizer: Optimizer to be used. Defaults to 'adam'.
        learning_rate: Learning rate to be used in training. Value must be > 0. Defalts to 0.001.
        loss_function: Loss function to be used. Defaults to 'mse'.
        dropout_rate: Fraction of the input units to drop. Value must be >= 0 and <= 1. Defaults to None.
        early_stopping: Whether or not to use early stopping in training. Defaults to True.
        es_patience: Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        metrics: Metrics to be evaluated by the model during training and testing. Defaults to ['mse'].
        random_state: Seed for random number generation. Sets Python, Numpy and Tensorflow seeds to make
            program deterministic. Defaults to None (random state / seed).

    Returns:
        Trained MLP model and training history.

    Raises:
        arcpy.AddError: Some of the numeric parameters have invalid values.
        arcpy.AddError: Shape of X or y is invalid.
    """
    # 1. Check input data
    _check_ML_model_data_input(X=X, y=y)
    _check_MLP_inputs(
        neurons=neurons,
        validation_split=validation_split,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        es_patience=es_patience,
        batch_size=batch_size,
        epochs=epochs,
        output_neurons=output_neurons,
        loss_function=loss_function,
    )

    if random_state is not None:
        keras.utils.set_random_seed(random_state)

    # 2. Create and compile a sequential model
    arcpy.AddMessage("Creating and compiling a sequential model...")
    model = keras.Sequential()

    model.add(keras.layers.Input(shape=(X.shape[1],)))

    arcpy.AddMessage("Adding hidden layers...")
    for neuron in neurons:
        model.add(keras.layers.Dense(units=neuron, activation=activation))

        if dropout_rate is not None:
            model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(units=output_neurons, activation=last_activation))

    arcpy.AddMessage("Compiling the model...")
    model.compile(
        optimizer=_keras_optimizer(optimizer, learning_rate=learning_rate),
        loss=loss_function,
        metrics=[_keras_metric(metric) for metric in metrics],
    )
    
    # 3. Train the model
    # Early stopping callback
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=es_patience)] if early_stopping else []
    callbacks.append(ArcPyLoggingCallback(epochs))
    
    arcpy.AddMessage("Training the model...")

    history = model.fit(
        X,
        y,
        epochs=epochs,
        validation_split=validation_split if validation_split else 0.0,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, history