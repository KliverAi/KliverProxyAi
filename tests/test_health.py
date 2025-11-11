import pytest


def test_root_endpoint(client):
    """Test root endpoint returns correct info"""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Kliver.AI Chat API"
    assert data["status"] == "running"
    assert data["version"] == "3.0.0"
    assert "docs" in data


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Kliver.AI Chat API"
    assert data["version"] == "3.0.0"
    assert "telemetry_enabled" in data
