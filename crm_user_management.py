# -*- coding: utf-8 -*-
"""
User Management API for Admins
Allows admins to create, update, and delete agents
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
import logging

from crm_database import (
    get_session, User, UserRole, UserManager, get_enum_value
)
from crm_auth import get_current_admin

logger = logging.getLogger(__name__)

# Create API router
user_router = APIRouter(prefix="/api/users", tags=["User Management"])

# Pydantic models
class AgentCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6)
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class AgentUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None

class AgentResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    first_name: Optional[str]
    last_name: Optional[str]
    full_name: str
    is_active: bool
    created_at: datetime
    created_by_id: int
    last_login: Optional[datetime]
    gate_slot: Optional[int]

# User Management endpoints
@user_router.get("/agents", response_model=List[AgentResponse])
async def get_my_agents(current_admin: User = Depends(get_current_admin)):
    """Get all agents created by current admin"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        agents = user_manager.get_agents_by_admin(current_admin.id)
        
        return [
            AgentResponse(
                id=agent.id,
                username=agent.username,
                email=agent.email,
                role=get_enum_value(agent.role),
                first_name=agent.first_name,
                last_name=agent.last_name,
                full_name=agent.full_name,
                is_active=agent.is_active,
                created_at=agent.created_at,
                created_by_id=agent.created_by_id,
                last_login=agent.last_login,
                gate_slot=agent.gate_slot
            )
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@user_router.post("/agents", response_model=AgentResponse)
async def create_agent(agent_data: AgentCreate, current_admin: User = Depends(get_current_admin)):
    """Create a new agent (admin only)"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        # Check agent slot limit
        current_agents_count = session.query(User).filter(
            User.created_by_id == current_admin.id,
            User.role == UserRole.AGENT
        ).count()
        
        max_agents = current_admin.max_agents or 0
        
        if current_agents_count >= max_agents:
            raise HTTPException(
                status_code=403,
                detail=f"You have reached your maximum number of agents ({max_agents}). Please purchase more agent slots from the Billing page."
            )
        
        # Check if username already exists
        existing_user = user_manager.get_user_by_username(agent_data.username)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = user_manager.get_user_by_email(agent_data.email)
        if existing_email:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Create new agent
        new_agent = user_manager.create_user(
            username=agent_data.username,
            email=agent_data.email,
            password=agent_data.password,
            role=UserRole.AGENT,
            created_by_id=current_admin.id,
            first_name=agent_data.first_name,
            last_name=agent_data.last_name
        )
        
        # Automatically assign a gate slot to the new agent
        assigned_slot = user_manager.assign_gate_slot(new_agent.id)
        if not assigned_slot:
            # If no slots available, delete the agent and raise error
            user_manager.delete_user(new_agent.id)
            raise HTTPException(
                status_code=500,
                detail="No available gate slots (9-19). All slots are currently assigned."
            )
        
        logger.info(f"✅ Assigned gate slot {assigned_slot} to agent {new_agent.username}")
        
        return AgentResponse(
            id=new_agent.id,
            username=new_agent.username,
            email=new_agent.email,
            role=get_enum_value(new_agent.role),
            first_name=new_agent.first_name,
            last_name=new_agent.last_name,
            full_name=new_agent.full_name,
            is_active=new_agent.is_active,
            created_at=new_agent.created_at,
            created_by_id=new_agent.created_by_id,
            last_login=new_agent.last_login,
            gate_slot=new_agent.gate_slot
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@user_router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: int, current_admin: User = Depends(get_current_admin)):
    """Get specific agent details"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        agent = user_manager.get_user_by_id(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify agent belongs to current admin
        if agent.created_by_id != current_admin.id:
            raise HTTPException(
                status_code=403,
                detail="You can only view agents you created"
            )
        
        return AgentResponse(
            id=agent.id,
            username=agent.username,
            email=agent.email,
            role=get_enum_value(agent.role),
            first_name=agent.first_name,
            last_name=agent.last_name,
            full_name=agent.full_name,
            is_active=agent.is_active,
            created_at=agent.created_at,
            created_by_id=agent.created_by_id,
            last_login=agent.last_login,
            gate_slot=agent.gate_slot
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@user_router.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: int, agent_update: AgentUpdate, current_admin: User = Depends(get_current_admin)):
    """Update agent information"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        agent = user_manager.get_user_by_id(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify agent belongs to current admin
        if agent.created_by_id != current_admin.id:
            raise HTTPException(
                status_code=403,
                detail="You can only update agents you created"
            )
        
        # Update fields
        update_data = agent_update.dict(exclude_unset=True)
        updated_agent = user_manager.update_user(agent_id, **update_data)
        
        return AgentResponse(
            id=updated_agent.id,
            username=updated_agent.username,
            email=updated_agent.email,
            role=get_enum_value(updated_agent.role),
            first_name=updated_agent.first_name,
            last_name=updated_agent.last_name,
            full_name=updated_agent.full_name,
            is_active=updated_agent.is_active,
            created_at=updated_agent.created_at,
            created_by_id=updated_agent.created_by_id,
            last_login=updated_agent.last_login,
            gate_slot=updated_agent.gate_slot
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@user_router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: int, current_admin: User = Depends(get_current_admin)):
    """Delete an agent (admin only)"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        agent = user_manager.get_user_by_id(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify agent belongs to current admin
        if agent.created_by_id != current_admin.id:
            raise HTTPException(
                status_code=403,
                detail="You can only delete agents you created"
            )
        
        # Free up the gate slot before deleting the agent
        if agent.gate_slot:
            user_manager.free_gate_slot(agent_id)
            logger.info(f"🔓 Freed gate slot {agent.gate_slot} from agent {agent.username}")
        
        # Delete agent (this will cascade delete their leads and campaigns)
        user_manager.delete_user(agent_id)
        
        return {"status": "success", "message": "Agent deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Export
__all__ = ['user_router']

