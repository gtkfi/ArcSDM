"""Shared MLP constants for activations, losses, optimizers, and metrics."""

# Activation functions
ACTIVATION_LINEAR = "Linear"
ACTIVATION_RELU = "ReLU"
ACTIVATION_SIGMOID = "Sigmoid"
ACTIVATION_SOFTMAX = "Softmax"
ACTIVATION_TANH = "Tanh"

# Loss functions
LOSS_HUBER = "Huber"
LOSS_L1 = "MAE"  # Mean Absolute Error
LOSS_MSE = "MSE"  # Mean Squared Error

# Optimizers
OPTIMIZER_ADAGRAD = "Adagrad"
OPTIMIZER_ADAM = "Adam"
OPTIMIZER_RMSPROP = "RMSprop"
OPTIMIZER_SGD = "SGD"

# Validation metrics
VALIDATION_ACCURACY = "Accuracy"
VALIDATION_F1 = "F1"
VALIDATION_L1 = "MAE"  # Mean Absolute Error
VALIDATION_MSE = "MSE"  # Mean Squared Error
VALIDATION_PRECISION = "Precision"
VALIDATION_R2 = "R-squared"
VALIDATION_RECALL = "Recall"
VALIDATION_RMSE = "RMSE"  # Root Mean Squared Error
