from django.urls import path
from .views import FileUploadView, SelectedModelView, ParametersView, TrainModelAPIView, DownloadModelAPIView

urlpatterns = [
    path('upload/', FileUploadView.as_view(), name='file_upload'),
    path('modelselected/', SelectedModelView.as_view(), name='model_selected'),
    path('parameters/', ParametersView.as_view(), name='parameters'),
    path('train/', TrainModelAPIView.as_view(), name='train_model'),
    path('download_model/', DownloadModelAPIView.as_view(), name='download_model'),
]
