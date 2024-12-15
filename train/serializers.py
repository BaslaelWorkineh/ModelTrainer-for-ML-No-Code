from rest_framework import serializers

class CsvUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    headers = serializers.JSONField()
    data = serializers.JSONField()
    model_choice = serializers.CharField()
    parameters = serializers.JSONField()
