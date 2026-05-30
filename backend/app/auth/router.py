from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.auth.security import verify_password, create_access_token, get_password_hash
from app.core.config import settings
from app.core.sanitization import strip_html_tags

router = APIRouter()


@router.post("/login")
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    username = strip_html_tags(form_data.username)
    password = strip_html_tags(form_data.password)

    result = await db.execute(select(User).where(User.email == username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": str(user.id)})

    secure = settings.ENVIRONMENT == "production" and settings.FRONTEND_URL.startswith("https")
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=86400,
    )

    return {"message": "Login successful"}


@router.post("/register")
async def register(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    username = strip_html_tags(form_data.username)
    password = strip_html_tags(form_data.password)

    result = await db.execute(select(User).where(User.email == username))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    hashed_password = get_password_hash(password)
    new_user = User(
        email=username,
        hashed_password=hashed_password,
        role="user",
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return {"message": "User registered successfully", "user_id": new_user.id}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Logout successful"}
