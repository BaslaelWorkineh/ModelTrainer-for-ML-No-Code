from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    headers = models.JSONField()
    data = models.JSONField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File uploaded at {self.uploaded_at}"

class ModelSelection(models.Model):
    selected_model = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Model {self.selected_model} selected at {self.created_at}"

class Parameter(models.Model):
    """
    A generic model to store parameters for training without binding them to a specific model selection.
    """
    name = models.CharField(max_length=255)  # Parameter name, e.g., 'learning_rate'
    value = models.JSONField()              # Parameter value, can store JSON objects
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Parameter {self.name}: {self.value} (Created at {self.created_at})"
class TrainingHistory(models.Model):
    model_selection = models.ForeignKey(ModelSelection, on_delete=models.CASCADE)
    accuracy = models.FloatField()
    loss = models.FloatField()
    validation_accuracy = models.FloatField()
    validation_loss = models.FloatField()
    progress = models.IntegerField()
    trained_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Training for {self.model_selection.selected_model} at {self.trained_at}"
