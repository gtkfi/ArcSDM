import arcpy
from keras.callbacks import Callback

class ArcPyLoggingCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        arcpy.SetProgressor("step", "Training the model...", 0, self.epochs, 1)
        arcpy.AddMessage("Training started.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        arcpy.SetProgressorPosition(epoch + 1)
        arcpy.SetProgressorLabel(f"Epoch {epoch + 1}/{self.epochs}")
        arcpy.AddMessage(f"Epoch {epoch + 1}:")
        progress = (epoch + 1) / self.epochs * 100
        arcpy.AddMessage(f"Progress: {progress:.2f}%")
        for key, value in logs.items():
            arcpy.AddMessage(f"    {key}: {value}")

    def on_batch_end(self, batch, logs=None):
        # Suppress batch-level logs
        pass

    def on_train_end(self, logs=None):
        arcpy.AddMessage("Training completed.")
        arcpy.ResetProgressor()