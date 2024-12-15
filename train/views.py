import pandas as pd
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import UploadedFile, TrainingHistory, ModelSelection, Parameter
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from django.http import FileResponse

class FileUploadView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        headers = request.data.get('headers')
        data = request.data.get('data')
        
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if headers and data:
                # Save to the database (convert headers and data to proper types)
                headers = json.loads(headers)  # Convert the headers from JSON string
                data = json.loads(data)  # Convert data to JSON-compatible format

                uploaded_file = UploadedFile.objects.create(
                    file=file,
                    headers=headers,
                    data=data
                )

            return Response({
                "success": True,
                "message": "File uploaded and processed successfully."
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class SelectedModelView(APIView):
    def post(self, request):
        try:
            model = request.data.get('model')

            # Check if the model was provided in the request
            if not model:
                return Response({"error": "Model selection is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Create a new ModelSelection instance without parameters
            model_selection = ModelSelection.objects.create(
                selected_model=model,
            )

            return Response({"message": "Model data received successfully!"}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class ParametersView(APIView):
    def post(self, request, *args, **kwargs):
        # Extract parameters from the request
        general_params = request.data.get('general', {})
        model_specific_params = request.data.get('model_specific', {})

        if not general_params and not model_specific_params:
            return Response({'error': 'No parameters provided.'}, status=400)

        # Save general parameters
        for name, value in general_params.items():
            Parameter.objects.create(name=name, value=value)

        # Save model-specific parameters
        for name, value in model_specific_params.items():
            Parameter.objects.create(name=name, value=value)

        return Response({'message': 'Parameters saved successfully.'}, status=200)
   
class TrainModelAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        uploaded_file = UploadedFile.objects.last()
        if not uploaded_file:
            return Response({'error': 'No CSV file found in the database'}, status=400)

        parameters = Parameter.objects.all()
        if not parameters.exists():
            return Response({'error': 'No parameters found in the database'}, status=400)

        try:
            df = pd.DataFrame(uploaded_file.data)

            X, y = self.prepare_data(df)  # Prepare the data (features and target)
            params = {param.name: param.value for param in parameters}
            
            model_selection = ModelSelection.objects.last()
            if not model_selection:
                return Response({'error': 'No model selected in the database'}, status=400)

            model_name = model_selection.selected_model  # Get the model name from the database

            # Pass X and y to get_model
            model = self.get_model(model_name, params, y)  # Pass y as argument here
            model.fit(X, y)

            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            loss = 1 - accuracy

            # Store training history
            self.store_training_history(accuracy, loss)

            # Save the trained model to a file
            model_filename = 'trained_model.pkl'
            model_filepath = os.path.join('models', model_filename)  # Define the path
            os.makedirs(os.path.dirname(model_filepath), exist_ok=True)  # Create directory if not exists
            joblib.dump(model, model_filepath)  # Save model as a .pkl file

            return Response({
                'metrics': {
                    'accuracy': accuracy,
                    'loss': loss,
                    'validationAccuracy': accuracy,  # Placeholder if no validation accuracy is computed
                    'validationLoss': loss           # Placeholder if no validation loss is computed
                },
                'progress': 100,  # Assuming progress is complete for now
                'message': 'Training completed successfully.',
                'model_url': model_filepath  # Provide the model URL for downloading
            })


        except Exception as e:
            return Response({'error': str(e)}, status=500)

    def get_model(self, model_name, params, y):
        """
        Select and configure the model based on the provided name and target data (y).
        It distinguishes between classification and regression.
        """
        # Check if the target (y) is continuous or categorical
        if y.nunique() > 2:  # Continuous target variable (regression task)
            if model_name == 'Linear Regression':
                return LinearRegression()
            elif model_name == 'Decision Tree':
                return DecisionTreeRegressor()  # Use regression model
            else:
                raise ValueError(f"Model {model_name} is not suitable for regression.")
        
        else:  # Binary or multi-class classification
            if model_name == 'Logistic Regression':
                return LogisticRegression(max_iter=1000)
            elif model_name == 'Decision Tree':
                return DecisionTreeClassifier()  # Use classification model
            elif model_name == 'Neural Network':
                return MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000)
            else:
                raise ValueError(f"Model {model_name} is not suitable for classification.")

    def prepare_data(self, df):
        """
        Prepare the training data: Features (X) and Target (y).
        Assumes the last column is the target/label column.
        """
        print(f"Initial DataFrame shape: {df.shape}")
        print("Initial data types:")
        print(df.dtypes)

        # Convert empty strings to NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)  # Replace empty strings and whitespace with NaN

        # Drop the 'id' column if it's unnecessary for modeling
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # Separate features (X) and target (y)
        X = df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1]   # The last column as target (e.g., 'Depression')

        # Convert numeric columns to floats, handling non-numeric values
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Encode categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        encoder = LabelEncoder()
        for column in categorical_columns:
            if X[column].dtype == 'object':
                X[column] = encoder.fit_transform(X[column].astype(str))

        # Handle missing values in numeric columns by filling NaNs with mean values
        for column in numeric_columns:
            mean_value = X[column].mean()
            if pd.isna(mean_value):  # If mean is NaN (all values were NaN)
                X[column] = X[column].fillna(0)
            else:
                X[column] = X[column].fillna(mean_value)

        # Ensure all data is numeric before scaling
        X = X.astype(float)

        # Scale features
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            print(f"Scaling error: {str(e)}")
            print("Data shape:", X.shape)
            print("Data types:", X.dtypes)
            print("Sample of problematic data:")
            print(X.head())
            raise

        # Convert 'y' (target) to numeric (if possible)
        y = pd.to_numeric(y, errors='coerce')  # Convert to numeric, coerce errors to NaN

        # Handle missing values in 'y'
        y = y.fillna(y.mean())  # Fill NaNs with the mean (you can modify this if needed)

        # Check if y is binary or continuous
        if len(y.unique()) > 2:  # Continuous data for regression
            print("Regression problem detected.")
        else:  # Binary or categorical for classification
            print("Classification problem detected.")

        return X_scaled, y

    
    def store_training_history(self, accuracy, loss):
        """
        Store training history in the database.
        """
        try:
            # Get the most recent ModelSelection from the database
            model_selection = ModelSelection.objects.last()
            
            if not model_selection:
                raise ValueError("No model selection found in the database.")
            
            # Create TrainingHistory with the associated model selection
            TrainingHistory.objects.create(
                accuracy=accuracy,
                loss=loss,
                validation_accuracy=accuracy,  # Placeholder: Replace with actual validation accuracy if needed
                validation_loss=loss,
                progress=100,  # Assume training is 100% completed
                model_selection=model_selection  # Associate the ModelSelection
            )
        except Exception as e:
            raise ValueError(f"Error storing training history: {str(e)}")


class DownloadModelAPIView(APIView):
    def get(self, request, *args, **kwargs):
        model_filename = 'trained_model.pkl'  # Use the same filename as in the training view
        model_filepath = os.path.join('models', model_filename)

        # Check if the model file exists
        if os.path.exists(model_filepath):
            response = FileResponse(open(model_filepath, 'rb'), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename={model_filename}'
            return response
        else:
            return Response({'error': 'Model file not found.'}, status=404)