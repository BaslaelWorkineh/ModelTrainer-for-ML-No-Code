import pandas as pd
import numpy as np
from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser
from rest_framework.views import APIView
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .serializers import CsvUploadSerializer

class ModelTrainingView(APIView):
    parser_classes = (MultiPartParser,)
    
    def post(self, request, *args, **kwargs):
        serializer = CsvUploadSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data['file']
            model_choice = serializer.validated_data['model_choice']
            parameters = serializer.validated_data['parameters']
            
            # Load and preprocess CSV data
            df = pd.read_csv(file)
            X = df.drop('target', axis=1)  # assuming 'target' column is the label
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Choose the model based on input
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(**parameters)
            elif model_choice == "Neural Network":
                model = MLPClassifier(**parameters)

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate the model
            accuracy = accuracy_score(y_test, model.predict(X_test))

            # Save the model (Optional: you can use joblib to save the trained model)
            # Example of saving the model:
            # import joblib
            # joblib.dump(model, 'trained_model.pkl')

            return JsonResponse({
                'accuracy': accuracy,
                'message': 'Model trained successfully!'
            })
        else:
            return JsonResponse(serializer.errors, status=400)
