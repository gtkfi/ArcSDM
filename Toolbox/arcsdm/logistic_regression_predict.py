"""
Original implementation is included in EIS Toolkit (https://github.com/GispoCoding/eis_toolkit).
"""
import arcpy
import numpy as np
import pandas as pd
from typing import Literal, Optional, Sequence, Tuple, Union
from sklearn.linear_model import LogisticRegression

from arcsdm.machine_learning.general import prepare_data_for_ml, train_and_validate_sklearn_model, save_model


def Execute(self, parameters, messages):
    """The source code of the tool."""
    #try:
    input_data = parameters[0].valueAsText.split(';')
    target = parameters[1].valueAsText
    nodata_value = parameters[2].valueAsText
    validation_method = parameters[3].valueAsText
    metrics = parameters[4].valueAsText.split(';')
    split_size = parameters[5].value
    cv_folds = parameters[6].value
    penalty = parameters[7].valueAsText if parameters[7].valueAsText != "none" else None
    max_iter = parameters[8].value
    solver = parameters[9].valueAsText
    verbose = parameters[10].value
    random_state = parameters[11].value
    output_file = parameters[12].valueAsText

    X, y = prepare_data_for_ml(input_data, target, nodata_value)

    # Perform Logistic Regression
    model, metrics = logistic_regression_train(
        X=X,
        y=y,
        validation_method=validation_method,
        metrics=metrics,
        split_size=split_size,
        cv_folds=cv_folds,
        penalty=penalty,
        max_iter=max_iter,
        solver=solver,
        verbose=verbose,
        random_state=random_state,
    )
    
    arcpy.AddMessage("="*5 + "Model training completed." + "="*5)
    arcpy.AddMessage(f"Saving model to {output_file}.joblib")
    
    save_model(model, output_file)

    for metric, value in metrics.items():
        arcpy.AddMessage(f"{metric}: {value}")

    '''

    # Return geoprocessing specific errors
    except arcpy.ExecuteError:    
        arcpy.AddError(arcpy.GetMessages(2))    

    # Return any other type of error
    except:
        # By default any other errors will be caught here
        e = sys.exc_info()[1]
        print(e.args[0])
    '''
def logistic_regression_train(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    validation_method: Literal["split", "kfold_cv", "skfold_cv", "loo_cv", "none"] = "split",
    metrics: Sequence[Literal["accuracy", "precision", "recall", "f1", "auc"]] = ["accuracy"],
    split_size: float = 0.2,
    cv_folds: int = 5,
    penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
    max_iter: int = 100,
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
    verbose: int = 0,
    random_state: Optional[int] = None,
) -> Tuple[LogisticRegression, dict]:
    """
    Train and optionally validate a Logistic Regression classifier model using Sklearn.

    Various options and configurations for model performance evaluation are available. No validation,
    split to train and validation parts, and cross-validation can be chosen. If validation is performed,
    metric(s) to calculate can be defined and validation process configured (cross-validation method,
    number of folds, size of the split). Depending on the details of the validation process,
    the output metrics dictionary can be empty, one-dimensional or nested.

    The choice of the algorithm depends on the penalty chosen. Supported penalties by solver:
    'lbfgs' - ['l2', None]
    'liblinear' - ['l1', 'l2']
    'newton-cg' - ['l2', None]
    'newton-cholesky' - ['l2', None]
    'sag' - ['l2', None]
    'saga' - ['elasticnet', 'l1', 'l2', None]

    For more information about Sklearn Logistic Regression, read the documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

    Args:
        X: Training data.
        y: Target labels.
        validation_method: Validation method to use. "split" divides data into two parts, "kfold_cv"
            performs k-fold cross-validation, "skfold_cv" performs stratified k-fold cross-validation,
            "loo_cv" performs leave-one-out cross-validation and "none" will not validate model at all
            (in this case, all X and y will be used solely for training).
        metrics: Metrics to use for scoring the model. Defaults to "accuracy".
        split_size: Fraction of the dataset to be used as validation data (rest is used for training).
            Used only when validation_method is "split". Defaults to 0.2.
        cv_folds: Number of folds used in cross-validation. Used only when validation_method is "kfold_cv"
            or "skfold_cv". Defaults to 5.
        penalty: Specifies the norm of the penalty. Defaults to 'l2'.
        max_iter: Maximum number of iterations taken for the solvers to converge. Defaults to 100.
        solver: Algorithm to use in the optimization problem. Defaults to 'lbfgs'.
        verbose: Specifies if modeling progress and performance should be printed. 0 doesn't print,
            values 1 or above will produce prints.
        random_state: Seed for random number generation. Defaults to None.
        **kwargs: Additional parameters for Sklearn's LogisticRegression.

    Returns:
        The trained Logistic Regression classifier and metric scores as a dictionary.

    Raises:
        InvalidParameterValueException: If some of the numeric parameters are given invalid input values.
        NonMatchingParameterLengthsException: X and y have mismatching sizes.
    """
    if max_iter < 1:
        arcpy.AddError("Max iter must be > 0.")
        raise arcpy.ExecuteError
    if verbose < 0:
        arcpy.AddError("Verbose must be a non-negative number.")
        raise arcpy.ExecuteError

    model = LogisticRegression(
        penalty=penalty, max_iter=max_iter, random_state=random_state, solver=solver, verbose=verbose
    )

    model, metrics = train_and_validate_sklearn_model(
        X=X,
        y=y,
        model=model,
        validation_method=validation_method,
        metrics=metrics,
        split_size=split_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    return model, metrics