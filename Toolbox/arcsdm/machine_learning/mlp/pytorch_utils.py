import arcpy
import torch
import torch.nn as nn
import torch.optim as optim

import Toolbox.arcsdm.mlp.mlp_common


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def get_pytorch_optimizer(optimizer: str, params, learning_rate):
    """Create a Pytorch optimizer from given name and parameters."""
    optimizer = optimizer.lower().strip()
    if optimizer == Toolbox.arcsdm.mlp.mlp_common.OPTIMIZER_ADAM.lower():
        return optim.Adam(params, lr=learning_rate)
    elif optimizer == Toolbox.arcsdm.mlp.mlp_common.OPTIMIZER_ADAGRAD.lower():
        return optim.Adagrad(params, lr=learning_rate)
    elif optimizer == Toolbox.arcsdm.mlp.mlp_common.OPTIMIZER_RMSPROP.lower():
        return optim.RMSprop(params, lr=learning_rate)
    elif optimizer == Toolbox.arcsdm.mlp.mlp_common.OPTIMIZER_SGD.lower():
        return optim.SGD(params, lr=learning_rate)
    else:
        arcpy.AddError(f"Unidentified optimizer: {optimizer}")
        raise arcpy.ExecuteError


def get_pytorch_regression_loss(loss_function: str):
    """Create a Pytorch regression loss from given name."""
    selected_loss = str(loss_function).strip().lower() if loss_function is not None else ""

    if selected_loss in {
        Toolbox.arcsdm.mlp.mlp_common.LOSS_MSE.lower(),
        Toolbox.arcsdm.mlp.mlp_common.VALIDATION_MSE.lower()
    }:
        return nn.MSELoss()

    if selected_loss in {
        Toolbox.arcsdm.mlp.mlp_common.LOSS_L1.lower(),
        Toolbox.arcsdm.mlp.mlp_common.VALIDATION_L1.lower()
    }:
        return nn.L1Loss()

    if selected_loss in {
        Toolbox.arcsdm.mlp.mlp_common.LOSS_HUBER.lower()
    }:
        return nn.HuberLoss()

    arcpy.AddError(f"Unsupported loss function: {loss_function}")
    raise arcpy.ExecuteError


def correct(output, target, is_binary_classifier=True):
    if is_binary_classifier:
        class_pred = torch.sigmoid(output).round().int()  # logits -> probabilities -> labels
        correct_ones = class_pred == target.int()  # 1 for correct, 0 for incorrect
        return correct_ones.sum().item()           # count number of correct ones
    else:
        pred = output.argmax(dim=1)
        correct_ones = pred == target.int()
        return correct_ones.sum().item()


def train_classifier_epoch(
    device,
    data_loader,
    model,
    criterion,
    optimizer,
    target_dtype=torch.float32,
    binary_classifier=True
):
    try:
        model.train()

        num_batches = 0
        num_items = 0

        total_loss = 0
        total_correct = 0
        for data, target in data_loader:
            # Copy data and targets to GPU or CPU
            data = data.to(device).to(torch.float32)

            # Target dtype depends on loss function
            target = target.to(device).to(target_dtype)
            if binary_classifier and target.dim() == 1:
                target = target.view(-1, 1)

            # Do a forward pass
            output = model(data)

            # Calculate the loss
            loss = criterion(output, target)
            total_loss += loss
            num_batches += 1

            # Count number of correct
            total_correct += correct(output, target, is_binary_classifier=binary_classifier)
            num_items += len(target)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return {
            'loss': total_loss/num_batches,
            'accuracy': total_correct/num_items
            }
    except:
        raise


def train_regression_epoch(device, data_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data, target in data_loader:
        data = data.to(device).to(torch.float32)
        target = target.to(device).to(torch.float32).view(-1, 1)

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("nan")


def evaluate_classifier_epoch(
    device,
    test_loader,
    model,
    criterion,
    target_dtype=torch.float32,
    binary_classifier=True
):
    try:
        model.eval()

        num_batches = len(test_loader)
        num_items = len(test_loader.dataset)

        test_loss = 0
        total_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                # Copy data and targets to GPU
                data = data.to(device).to(torch.float32)

                # Target dtype depends on loss function
                target = target.to(device).to(target_dtype)
                if binary_classifier and target.dim() == 1:
                    target = target.view(-1, 1)

                # Do a forward pass
                output = model(data)

                # Calculate the loss
                loss = criterion(output, target)
                test_loss += loss.item()

                # Count number of correct digits
                total_correct += correct(output, target, is_binary_classifier=binary_classifier)

        return {
            'loss': test_loss/num_batches,
            'accuracy': total_correct/num_items
        }
    except:
        raise


def evaluate_regression_epoch(device, data_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).to(torch.float32)
            target = target.to(device).to(torch.float32).view(-1, 1)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("nan")


def predict(device, data_loader, model):
    try:
        model.eval()
        predicted = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device).to(torch.float32)

                output = model(data)
                predicted.append(output)

        return predicted
    except:
        raise
