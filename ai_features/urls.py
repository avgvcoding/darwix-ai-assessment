from django.urls import path
from . import views
from .title_suggestions import title_suggestions

urlpatterns = [
    # We will point 'transcribe/' to a view called `transcribe_audio`
    path('transcribe/', views.transcribe_audio, name="transcribe_audio"),
    path('title_suggestions/', title_suggestions, name="title_suggestions"),
]
