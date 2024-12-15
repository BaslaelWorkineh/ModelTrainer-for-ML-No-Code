from rest_framework import serializers

class CsvUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    model_choice = serializers.ChoiceField(choices=[
        ("Linear Regression", "Linear Regression"),
        ("Logistic Regression", "Logistic Regression"),
        ("Decision Tree", "Decision Tree"),
        ("Neural Network", "Neural Network"),
    ])
    parameters = serializers.JSONField()
