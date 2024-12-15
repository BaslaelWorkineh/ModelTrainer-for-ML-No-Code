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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np

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
        # Retrieve the latest uploaded file (CSV) for training
        uploaded_file = UploadedFile.objects.last()
        if not uploaded_file:
            return Response({'error': 'No CSV file found in the database'}, status=400)

        # Fetch all available parameters
        parameters = Parameter.objects.all()
        if not parameters.exists():
            return Response({'error': 'No parameters found in the database'}, status=400)

        try:
            # Load the uploaded file's data
            df = pd.DataFrame(uploaded_file.data)  # or pd.read_csv(uploaded_file.file) for CSV files
        except Exception as e:
            return Response({'error': f'Error loading CSV data: {str(e)}'}, status=500)

        # Prepare the data: assuming the last column is the target/label column
        X, y = self.prepare_data(df)  # Corrected call

        # Convert the parameters to a dictionary
        params = {param.name: param.value for param in parameters}

        # Dynamically determine the model name from the request
        model_name = request.data.get('model_name', 'linear_regression')  # You can adjust the logic here

        try:
            # Get the selected model
            model = self.get_model(model_name, params)
            model.fit(X, y)

            # Make predictions and calculate accuracy (or other relevant metrics)
            y_pred = model.predict(X)
            
            if model_name in ['linear_regression']:
                loss = ((y - y_pred) ** 2).mean()  # Mean Squared Error for regression models
                accuracy = 1 - loss  # For reporting purposes
            else:
                accuracy = accuracy_score(y, y_pred)  # For classification models
                loss = 1 - accuracy  # Placeholder for loss calculation

            # Store training history
            self.store_training_history(accuracy, loss)

            return Response({
                'accuracy': accuracy,
                'loss': loss,
                'message': 'Training completed successfully.'
            })

        except Exception as e:
            return Response({'error': str(e)}, status=500)


    def get_model(self, model_name, params):
        """
        Select and configure the model based on the provided name and parameters.
        """
        if model_name == 'linear_regression':
            # Parameters for Linear Regression (you can add more if needed)
            return LinearRegression()
        
        elif model_name == 'logistic_regression':
            # Parameters for Logistic Regression (you can add more if needed)
            return LogisticRegression(max_iter=1000)

        elif model_name == 'decision_tree':
            # Parameters for Decision Tree (you can add more if needed)
            return DecisionTreeClassifier()

        elif model_name == 'neural_network':
            # Parameters for Neural Network (you can add more if needed)
            return MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000)

        else:
            raise ValueError(f"Model {model_name} is not supported")

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
