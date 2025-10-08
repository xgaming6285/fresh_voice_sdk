# -*- coding: utf-8 -*-
"""
Authentication and Authorization for CRM API
Handles JWT tokens, login, registration, and permission checking
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import os
import logging

from crm_database import (
    get_session, User, UserRole, UserManager
)

logger = logging.getLogger(__name__)

# Create API router
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security scheme
security = HTTPBearer()

# Pydantic models
class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    first_name: Optional[str]
    last_name: Optional[str]
    full_name: str
    organization: Optional[str]
    created_at: datetime
    created_by_id: Optional[int]

# JWT Token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    session = get_session()
    try:
        user_manager = UserManager(session)
        user = user_manager.get_user_by_id(user_id)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        return user
    finally:
        session.close()

# Dependency to check if user is admin
async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """Verify current user is an admin"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Dependency to check if user is superadmin
async def get_current_superadmin(current_user: User = Depends(get_current_user)) -> User:
    """Verify current user is a superadmin"""
    if current_user.role != UserRole.SUPERADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required"
        )
    return current_user

# Dependency to check if user has active subscription
async def check_subscription(current_user: User = Depends(get_current_user)) -> User:
    """Check if user has active subscription (for write operations)"""
    from crm_database import UserManager
    
    # Superadmin always passes
    if current_user.role == UserRole.SUPERADMIN:
        return current_user
    
    # For admins, check their own subscription
    if current_user.role == UserRole.ADMIN:
        if not current_user.is_subscription_active():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your subscription has expired. Please renew your subscription in the Billing page to continue using the service."
            )
        return current_user
    
    # For agents, check their admin's subscription
    if current_user.role == UserRole.AGENT:
        session = get_session()
        try:
            user_manager = UserManager(session)
            admin = user_manager.get_user_by_id(current_user.created_by_id)
            
            if not admin or not admin.is_subscription_active():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Your organization's subscription has expired. Please contact your administrator to renew the subscription."
                )
            return current_user
        finally:
            session.close()
    
    return current_user

def user_to_dict(user: User) -> Dict[str, Any]:
    """Convert user object to dictionary"""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": user.full_name,
        "organization": user.organization,
        "created_at": user.created_at.isoformat(),
        "created_by_id": user.created_by_id
    }

# Authentication endpoints
@auth_router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    """Register a new admin user (public endpoint - first registration)"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Check if username already exists
        existing_user = user_manager.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = user_manager.get_user_by_email(user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new admin user
        new_user = user_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=UserRole.ADMIN,
            first_name=user_data.first_name,
            last_name=user_data.last_name
        )
        
        # Generate access token
        access_token = create_access_token(
            data={"user_id": new_user.id, "role": new_user.role.value}
        )
        
        return TokenResponse(
            access_token=access_token,
            user=user_to_dict(new_user)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@auth_router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """Login with username and password"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Get user by username
        user = user_manager.get_user_by_username(credentials.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Verify password
        if not user.verify_password(credentials.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        session.commit()
        
        # Generate access token
        access_token = create_access_token(
            data={"user_id": user.id, "role": user.role.value}
        )
        
        return TokenResponse(
            access_token=access_token,
            user=user_to_dict(user)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@auth_router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role.value,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        full_name=current_user.full_name,
        organization=current_user.organization,
        created_at=current_user.created_at,
        created_by_id=current_user.created_by_id
    )

@auth_router.post("/verify-token")
async def verify_token(current_user: User = Depends(get_current_user)):
    """Verify if token is valid"""
    return {
        "valid": True,
        "user": user_to_dict(current_user)
    }

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=6)

@auth_router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user)
):
    """Change user password (not available for admins)"""
    try:
        # Admins cannot change their passwords
        if current_user.role == UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admins cannot change their passwords. Contact superadmin for password reset."
            )
        
        session = get_session()
        
        # Verify current password
        if not current_user.verify_password(password_data.current_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        current_user.hashed_password = User.hash_password(password_data.new_password)
        current_user.updated_at = datetime.utcnow()
        session.commit()
        
        return {
            "message": "Password changed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Export
__all__ = ['auth_router', 'get_current_user', 'get_current_admin', 'get_current_superadmin', 'check_subscription', 'user_to_dict']

