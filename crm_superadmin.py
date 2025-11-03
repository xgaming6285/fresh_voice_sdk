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
    get_session, User, UserRole, UserManager, Lead, Campaign,
    PaymentRequest, PaymentRequestStatus, SlotAdjustment, SystemSettings, get_enum_value
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

class SlotAdjustmentRequest(BaseModel):
    slots_change: int = Field(..., description="Number of slots to add (positive) or remove (negative)")
    reason: str = Field(..., min_length=3, max_length=500, description="Reason for the adjustment")

def get_admin_with_stats(user: User, session) -> Dict[str, Any]:
    """Convert admin user to dict with agent count"""
    # Count agents created by this admin
    agent_count = session.query(User).filter(
        User.created_by_id == user.id,
        User.role == UserRole.AGENT
    ).count()
    
    # Handle role - it might be an enum or already a string
    role_value = user.role.value if hasattr(user.role, 'value') else user.role
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": role_value,
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
        
        # Automatically assign a Google API key to the new admin
        assigned_api_key = user_manager.assign_api_key(new_admin.id)
        if assigned_api_key:
            logger.info(f"✅ Assigned Google API key to admin {new_admin.username}: {assigned_api_key[:20]}...{assigned_api_key[-4:]}")
        else:
            logger.warning(f"⚠️ Could not assign Google API key to admin {new_admin.username} - no keys available")
        
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
        session.add(admin)
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
        session.add(admin)
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
                "role": get_enum_value(agent.role),
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
        session.add(agent)
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

# Payment Management Endpoints

class PaymentRequestWithAdmin(BaseModel):
    id: int
    admin_id: int
    admin_username: str
    admin_email: str
    admin_organization: Optional[str]
    current_agents: int
    max_agents: int
    num_agents: int
    total_amount: float
    status: str
    payment_notes: Optional[str]
    admin_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime]

@superadmin_router.get("/payment-requests")
async def get_all_payment_requests(current_user: User = Depends(get_current_superadmin)):
    """Get all payment requests"""
    try:
        session = get_session()
        
        requests = session.query(PaymentRequest).order_by(
            PaymentRequest.status,
            PaymentRequest.created_at.desc()
        ).all()
        
        result = []
        for req in requests:
            admin = session.query(User).filter(User.id == req.admin_id).first()
            if not admin:
                continue
            
            # Count current agents
            current_agents = session.query(User).filter(
                User.created_by_id == admin.id,
                User.role == UserRole.AGENT
            ).count()
            
            result.append({
                "id": req.id,
                "admin_id": req.admin_id,
                "admin_username": admin.username,
                "admin_email": admin.email,
                "admin_organization": admin.organization,
                "current_agents": current_agents,
                "max_agents": admin.max_agents or 0,
                "num_agents": req.num_agents,
                "total_amount": req.total_amount,
                "status": get_enum_value(req.status),
                "payment_notes": req.payment_notes,
                "admin_notes": req.admin_notes,
                "created_at": req.created_at,
                "updated_at": req.updated_at,
                "approved_at": req.approved_at
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting payment requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/payment-requests/{request_id}/approve")
async def approve_payment_request(
    request_id: int,
    approval_data: dict,
    current_user: User = Depends(get_current_superadmin)
):
    """Approve a payment request and increase admin's max_agents"""
    try:
        session = get_session()
        
        payment_request = session.query(PaymentRequest).filter(
            PaymentRequest.id == request_id
        ).first()
        
        if not payment_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment request not found"
            )
        
        if payment_request.status != PaymentRequestStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment request is already {get_enum_value(payment_request.status)}"
            )
        
        # Get admin
        admin = session.query(User).filter(User.id == payment_request.admin_id).first()
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Update payment request
        payment_request.status = PaymentRequestStatus.APPROVED
        payment_request.approved_at = datetime.utcnow()
        payment_request.approved_by_id = current_user.id
        payment_request.admin_notes = approval_data.get("admin_notes", "")
        
        # Increase max_agents for admin
        admin.max_agents = (admin.max_agents or 0) + payment_request.num_agents
        admin.updated_at = datetime.utcnow()
        
        # Set subscription end date (30 days from now or extend existing)
        from datetime import timedelta
        if admin.subscription_end_date and admin.subscription_end_date > datetime.utcnow():
            # Extend existing subscription
            admin.subscription_end_date = admin.subscription_end_date + timedelta(days=30)
        else:
            # New or expired subscription
            admin.subscription_end_date = datetime.utcnow() + timedelta(days=30)
        
        session.add(payment_request)
        session.add(admin)
        session.commit()
        
        return {
            "message": f"Payment approved. Admin now has {admin.max_agents} agent slots with subscription until {admin.subscription_end_date.strftime('%Y-%m-%d')}.",
            "new_max_agents": admin.max_agents,
            "subscription_end_date": admin.subscription_end_date.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error approving payment request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/payment-requests/{request_id}/reject")
async def reject_payment_request(
    request_id: int,
    rejection_data: dict,
    current_user: User = Depends(get_current_superadmin)
):
    """Reject a payment request"""
    try:
        session = get_session()
        
        payment_request = session.query(PaymentRequest).filter(
            PaymentRequest.id == request_id
        ).first()
        
        if not payment_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment request not found"
            )
        
        if payment_request.status != PaymentRequestStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment request is already {get_enum_value(payment_request.status)}"
            )
        
        # Update payment request
        payment_request.status = PaymentRequestStatus.REJECTED
        payment_request.admin_notes = rejection_data.get("admin_notes", "")
        payment_request.updated_at = datetime.utcnow()
        
        session.add(payment_request)
        session.commit()
        
        return {
            "message": "Payment request rejected"
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error rejecting payment request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/settings/payment-wallet")
async def get_payment_wallet(current_user: User = Depends(get_current_superadmin)):
    """Get payment wallet address"""
    try:
        session = get_session()
        
        setting = session.query(SystemSettings).filter(
            SystemSettings.key == "payment_wallet_address"
        ).first()
        
        return {
            "wallet_address": setting.value if setting else None
        }
    except Exception as e:
        logger.error(f"Error getting payment wallet: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/settings/payment-wallet")
async def set_payment_wallet(
    wallet_data: dict,
    current_user: User = Depends(get_current_superadmin)
):
    """Set payment wallet address"""
    try:
        wallet_address = wallet_data.get("wallet_address", "").strip()
        
        session = get_session()
        
        setting = session.query(SystemSettings).filter(
            SystemSettings.key == "payment_wallet_address"
        ).first()
        
        if setting:
            setting.value = wallet_address
            setting.updated_at = datetime.utcnow()
            setting.updated_by_id = current_user.id
            session.add(setting)
        else:
            setting = SystemSettings(
                key="payment_wallet_address",
                value=wallet_address,
                updated_by_id=current_user.id
            )
            session.add(setting)
        
        session.commit()
        
        return {
            "message": "Payment wallet address updated",
            "wallet_address": wallet_address
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error setting payment wallet: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.post("/admins/{admin_id}/adjust-slots")
async def adjust_admin_slots(
    admin_id: int,
    adjustment_data: SlotAdjustmentRequest,
    current_user: User = Depends(get_current_superadmin)
):
    """Manually adjust admin's agent slots (increase or decrease) with reason tracking"""
    try:
        session = get_session()
        user_manager = UserManager(session)
        
        admin = user_manager.get_user_by_id(admin_id)
        
        if not admin or admin.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Calculate new max_agents
        previous_max_agents = admin.max_agents or 0
        new_max_agents = previous_max_agents + adjustment_data.slots_change
        
        if new_max_agents < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot reduce slots by {abs(adjustment_data.slots_change)}. Admin currently has {previous_max_agents} slots."
            )
        
        # Create slot adjustment record
        slot_adjustment = SlotAdjustment(
            admin_id=admin.id,
            adjusted_by_id=current_user.id,
            slots_change=adjustment_data.slots_change,
            reason=adjustment_data.reason,
            previous_max_agents=previous_max_agents,
            new_max_agents=new_max_agents,
            created_at=datetime.utcnow()
        )
        
        # Update admin's max_agents
        admin.max_agents = new_max_agents
        admin.updated_at = datetime.utcnow()
        
        session.add(slot_adjustment)
        session.add(admin)
        session.commit()
        
        action = "increased" if adjustment_data.slots_change > 0 else "decreased"
        
        return {
            "message": f"Admin agent slots {action} from {previous_max_agents} to {new_max_agents}",
            "previous_max_agents": previous_max_agents,
            "new_max_agents": new_max_agents,
            "slots_change": adjustment_data.slots_change,
            "reason": adjustment_data.reason
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error adjusting admin slots: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@superadmin_router.get("/admins/{admin_id}/slot-adjustments")
async def get_admin_slot_adjustments(
    admin_id: int,
    current_user: User = Depends(get_current_superadmin)
):
    """Get slot adjustment history for an admin"""
    try:
        session = get_session()
        
        # Verify admin exists
        admin = session.query(User).filter(
            User.id == admin_id,
            User.role == UserRole.ADMIN
        ).first()
        
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Admin not found"
            )
        
        # Get all adjustments for this admin
        adjustments = session.query(SlotAdjustment).filter(
            SlotAdjustment.admin_id == admin_id
        ).order_by(SlotAdjustment.created_at.desc()).all()
        
        result = []
        for adj in adjustments:
            # Get the superadmin who made the adjustment
            adjusted_by = session.query(User).filter(User.id == adj.adjusted_by_id).first()
            
            result.append({
                "id": adj.id,
                "admin_id": adj.admin_id,
                "adjusted_by_id": adj.adjusted_by_id,
                "adjusted_by_username": adjusted_by.username if adjusted_by else "Unknown",
                "slots_change": adj.slots_change,
                "reason": adj.reason,
                "previous_max_agents": adj.previous_max_agents,
                "new_max_agents": adj.new_max_agents,
                "created_at": adj.created_at
            })
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting slot adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Export
__all__ = ['superadmin_router']

