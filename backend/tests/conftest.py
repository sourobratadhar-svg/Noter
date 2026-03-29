"""Shared pytest fixtures for RAG backend tests"""
import pytest
import requests
import os

@pytest.fixture
def api_client():
    """Shared requests session with proper headers"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session

@pytest.fixture
def base_url():
    """Get base URL from environment"""
    url = os.environ.get('EXPO_PUBLIC_BACKEND_URL')
    if not url:
        pytest.fail("EXPO_PUBLIC_BACKEND_URL not set in environment")
    return url.rstrip('/')
