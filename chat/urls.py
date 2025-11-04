from django.urls import path
from . import views

urlpatterns = [
    # The main chat page
    path('', views.index, name='index'),
    
    # API endpoint for handling chat messages
    path('api/chat/', views.chat_api, name='chat_api'),
    
    # API endpoint for generating MCQs
    path('api/mcq/', views.mcq_api, name='mcq_api'),

    # New Feature to enhance the combination
    path('api/quiz/check', views.quiz_check_api, name='quiz_check_api'),
]

