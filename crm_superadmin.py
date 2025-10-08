# -*- coding: utf-8 -*-
"""
Superadmin Management API
Endpoints for superadmin to manage client admins
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from crm_database import (
    get_session, User, UserRole, UserManager, Lead, Campaign
)
from crm_auth import get_current_superadmin, user_to_dict

logger = logging.getLogger(__name__)

# Create API router
superadmin_router = APIRouter(prefix="/api/superadmin", tags=["Superadmin"])

# Pydantic models
class AdminCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)
    organization: str = Field(min_length=2, max_length=200)
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class AdminUpdate(BaseModel):
    email: Optional[EmailStr] = None
    organization: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None

class AdminResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    organization: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    full_name: str
    is_active: bool
    created_at: datetime
    created_by_id: Optional[int]
    agent_count: int

class AgentWithAdminResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: str
    is_active: bool
    created_at: datetime
    admin_id: int
    admin_username: str
    admin_organization: Optional[str]

class SystemStatsResponse(BaseModel):
    total_admins: int
    total_agents: int
    total_leads: int
    total_campaigns: int

def get_admin_with_stats(user: User, session) -> Dict[str, Any]:
    """Convert admin user to dict with agent count"""
    # Count agents created by this admin
    agent_count = session.query(User).filter(
        User.created_by_id == user.id,
        User.role == UserRole.AGENT
    ).count()
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "organization": user.organization,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "created_by_id": user.created_by_id,
        "agent_count": agent_count
    }

# Endpoints
@superadmin_router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(current_user: User = Depends(get_current_superadmin)):
    """Get system-wide statistics"""
    try:
        session = get_session()
        
        total_admins = session.query(User).filter(User.role == UserRole.ADMIN).count()
        total_agents = session.query(User).filter(User.role == UserRole.AGENT).count()
        total_leads = session.query(Lead).count()
        total_campaigns = session.query(Campaign).count()
        
        return SystemStatsResponse(
            total_admins=total_admins,
            total_agents=total_agents,
            total_leads=total_leads,
            total_campaigns=total_campaigns
        )
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/admins", response_model=List[AdminResponse])
async def get_all_admins(current_user: User = Depends(get_current_superadmin)):
    """Get all admin users with their agent counts"""
    try:
        session = get_session()
        
        admins = session.query(User).filter(User.role == UserRole.ADMIN).all()
        
        admin_list = [get_admin_with_stats(admin, session) for admin in admins]
        
        return admin_list
    except Exception as e:
        logger.error(f"Error getting admins: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/admins", response_model=AdminResponse)
async def create_admin(
    admin_data: AdminCreate,
    current_user: User = Depends(get_current_superadmin)
):
    """Create a new admin (client)"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Check if username already exists
        existing_user = user_manager.get_user_by_username(admin_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = user_manager.get_user_by_email(admin_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new admin user
        new_admin = user_manager.create_user(
            username=admin_data.username,
            email=admin_data.email,
            password=admin_data.password,
            role=UserRole.ADMIN,
            first_name=admin_data.first_name,
            last_name=admin_data.last_name,
            created_by_id=current_user.id
        )
        
        # Set organization
        new_admin.organization = admin_data.organization
        session.commit()
        session.refresh(new_admin)
        
        return get_admin_with_stats(new_admin, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/admins/{admin_id}", response_model=AdminResponse)
async def get_admin(
    admin_id: int,
    current_user: User = Depends(get_current_superadmin)
):
    """Get specific admin details"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        return get_admin_with_stats(admin, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.put("/admins/{admin_id}", response_model=AdminResponse)
async def update_admin(
    admin_id: int,
    admin_data: AdminUpdate,
    current_user: User = Depends(get_current_superadmin)
):
    """Update admin details (no password changes allowed)"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Update fields
        update_data = admin_data.dict(exclude_unset=True)
        
        # Check if email is being changed and if it's already taken
        if 'email' in update_data and update_data['email'] != admin.email:
            existing_email = user_manager.get_user_by_email(update_data['email'])
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        for field, value in update_data.items():
            setattr(admin, field, value)
        
        admin.updated_at = datetime.utcnow()
        session.commit()
        session.refresh(admin)
        
        return get_admin_with_stats(admin, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/admins/{admin_id}/reset-password")
async def reset_admin_password(
    admin_id: int,
    password_data: dict,
    current_user: User = Depends(get_current_superadmin)
):
    """Reset admin password"""
    try:
        new_password = password_data.get("new_password")
        if not new_password or len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters"
            )
        
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Reset password
        admin.hashed_password = User.hash_password(new_password)
        admin.updated_at = datetime.utcnow()
        session.commit()
        
        return {
            "message": "Admin password reset successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting admin password: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.delete("/admins/{admin_id}")
async def delete_admin(
    admin_id: int,
    current_user: User = Depends(get_current_superadmin)
):
    """Delete admin and all their agents"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Get all agents created by this admin
        agents = session.query(User).filter(
            User.created_by_id == admin_id,
            User.role == UserRole.AGENT
        ).all()
        
        # Delete all agents (cascading will handle their leads/campaigns)
        for agent in agents:
            session.delete(agent)
        
        # Delete the admin
        session.delete(admin)
        session.commit()
        
        return {
            "message": f"Admin and {len(agents)} agent(s) deleted successfully",
            "deleted_agents": len(agents)
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting admin: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/agents", response_model=List[AgentWithAdminResponse])
async def get_all_agents(current_user: User = Depends(get_current_superadmin)):
    """Get all agents across all admins"""
    try:
        session = get_session()
        
        # Query agents with their admin info
        agents = session.query(User).filter(User.role == UserRole.AGENT).all()
        
        agent_list = []
        for agent in agents:
            # Get admin info
            admin = session.query(User).filter(User.id == agent.created_by_id).first()
            
            agent_list.append({
                "id": agent.id,
                "username": agent.username,
                "email": agent.email,
                "role": agent.role.value,
                "full_name": agent.full_name,
                "is_active": agent.is_active,
                "created_at": agent.created_at,
                "admin_id": admin.id if admin else None,
                "admin_username": admin.username if admin else "Unknown",
                "admin_organization": admin.organization if admin else None
            })
        
        return agent_list
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/admins/{admin_id}/agents")
async def get_admin_agents(
    admin_id: int,
    current_user: User = Depends(get_current_superadmin)
):
    """Get all agents for a specific admin"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        agents = user_manager.get_agents_by_admin(admin_id)
        
        return [user_to_dict(agent) for agent in agents]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting admin agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/agents/{agent_id}/reset-password")
async def reset_agent_password(
    agent_id: int,
    password_data: dict,
    current_user: User = Depends(get_current_superadmin)
):
    """Reset agent password"""
    try:
        new_password = password_data.get("new_password")
        if not new_password or len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters"
            )
        
        session = get_session()
        user_manager = UserManager(session)
        
        agent = user_manager.get_user_by_id(agent_id)
        
        if not agent or agent.role != UserRole.AGENT:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        # Reset password
        agent.hashed_password = User.hash_password(new_password)
        agent.updated_at = datetime.utcnow()
        session.commit()
        
        return {
            "message": "Agent password reset successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting agent password: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Export
__all__ = ['superadmin_router']

