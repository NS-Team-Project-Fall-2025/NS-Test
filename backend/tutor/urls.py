"""URL routing for the Tutor API."""
from django.urls import path

from . import views


urlpatterns = [
    path("health/", views.health_check, name="health"),
    path("config/", views.api_config, name="api-config"),
    path("kb/files/", views.list_knowledge_base_files, name="kb-files"),
    path("kb/upload/", views.upload_knowledge_base_file, name="kb-upload"),
    path("kb/rebuild/", views.rebuild_knowledge_base, name="kb-rebuild"),
    path("kb/clear/", views.clear_knowledge_base, name="kb-clear"),
    path("chat/stream/", views.tutor_chat_stream, name="chat-stream"),
    path("summary/", views.summarize_document, name="summary"),
    path("summary/stream/", views.summarize_document_stream, name="summary-stream"),
    path("quiz/generate/", views.generate_quiz, name="quiz-generate"),
    path("quiz/grade/", views.grade_quiz, name="quiz-grade"),
    path("quiz/history/", views.list_quiz_history, name="quiz-history"),
    path("quiz/history/<str:quiz_id>/", views.quiz_history_detail, name="quiz-history-detail"),
    path("sessions/", views.list_chat_sessions, name="sessions"),
    path("sessions/create/", views.create_chat_session, name="sessions-create"),
    path("sessions/<str:session_id>/", views.chat_session_detail, name="sessions-detail"),
    path("sessions/<str:session_id>/update/", views.update_chat_session, name="sessions-update"),
]
