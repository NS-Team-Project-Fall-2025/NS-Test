"""Lightweight middleware helpers."""
from __future__ import annotations

from django.http import HttpResponse


class SimpleCORSHeadersMiddleware:
    """Attach permissive CORS headers to API responses.

    This avoids introducing third-party dependencies while allowing the
    React/Next.js frontend to communicate with the backend during development.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response: HttpResponse = self.get_response(request)
        response.setdefault("Access-Control-Allow-Origin", "*")
        response.setdefault("Access-Control-Allow-Headers", "*")
        response.setdefault("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        return response
