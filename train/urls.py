from django.urls import path
from .views import ModelTrainingView

urlpatterns = [
    path('train/', ModelTrainingView.as_view(), name='train_model'),
]
