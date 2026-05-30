import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app

async def test():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        # Test successful login
        login_resp = await client.post('/auth/login', data={'username': 'test@example.com', 'password': 'testpassword'})
        print('=== LOGIN RESPONSE ===')
        print(f'Status: {login_resp.status_code}')
        print(f'Cookies: {dict(login_resp.cookies)}')
        print(f'Set-Cookie header: {login_resp.headers.get("set-cookie", "NONE")}')
        print()
        
        # Test failed login
        fail_resp = await client.post('/auth/login', data={'username': 'test@example.com', 'password': 'wrong'})
        print('=== FAILED LOGIN ===')
        print(f'Status: {fail_resp.status_code}')
        print(f'Body: {fail_resp.text}')
        print()
        
        # Test protected without auth
        prot_resp = await client.get('/protected')
        print('=== PROTECTED NO AUTH ===')
        print(f'Status: {prot_resp.status_code}')
        print(f'Body: {prot_resp.text}')

asyncio.run(test())
