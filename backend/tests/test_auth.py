import pytest


@pytest.mark.asyncio
async def test_login_with_valid_credentials_sets_cookie(client, test_user):
    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "testpassword"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Login successful"

    cookies = response.cookies
    assert "access_token" in cookies
    assert cookies["access_token"] != ""


@pytest.mark.asyncio
async def test_login_with_invalid_credentials_returns_401(client, test_user):
    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401
    assert "access_token" not in response.cookies


@pytest.mark.asyncio
async def test_access_protected_endpoint_without_cookie_returns_401(client):
    response = await client.get("/protected")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_access_protected_endpoint_with_valid_cookie_returns_200(client, test_user):
    login_response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "testpassword"},
    )
    assert login_response.status_code == 200

    cookies = login_response.cookies
    response = await client.get("/protected", cookies=cookies)
    assert response.status_code == 200
    assert response.json()["message"] == "Access granted"
    assert response.json()["user_id"] == test_user.id
    assert response.json()["email"] == test_user.email


@pytest.mark.asyncio
async def test_cookie_has_correct_attributes(client, test_user):
    response = await client.post(
        "/auth/login",
        data={"username": "test@example.com", "password": "testpassword"},
    )
    assert response.status_code == 200

    set_cookie_header = response.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie_header
    assert "SameSite=lax" in set_cookie_header or "samesite=lax" in set_cookie_header.lower()
