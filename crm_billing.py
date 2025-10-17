# -*- coding: utf-8 -*-
"""
Billing and Payment Management API
Endpoints for admins to manage billing and superadmin to handle payments
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from crm_database import (
    get_session, User, UserRole, PaymentRequest, PaymentRequestStatus, SlotAdjustment, SystemSettings, get_enum_value
)
from crm_auth import get_current_user, get_current_admin, get_current_superadmin

logger = logging.getLogger(__name__)

# Constants
PRICE_PER_AGENT = 300.0  # USD per month

# Create API router
billing_router = APIRouter(prefix="/api/billing", tags=["Billing"])

# Pydantic models
class BillingInfo(BaseModel):
    current_agents: int
    max_agents: int
    available_slots: int
    price_per_agent: float
    payment_wallet: Optional[str]
    is_subscription_active: bool
    subscription_end_date: Optional[str]
    days_remaining: Optional[int]

class PaymentRequestCreate(BaseModel):
    num_agents: int = Field(gt=0, description="Number of agent slots to purchase")
    payment_notes: Optional[str] = None

class PaymentRequestResponse(BaseModel):
    id: int
    admin_id: int
    admin_username: str
    admin_organization: Optional[str]
    num_agents: int
    total_amount: float
    status: str
    payment_notes: Optional[str]
    admin_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime]

# Admin Billing Endpoints

@billing_router.get("/info", response_model=BillingInfo)
async def get_billing_info(current_user: User = Depends(get_current_admin)):
    """Get billing information for current admin"""
    try:
        session = get_session()
        
        # Count current agents
        current_agents = session.query(User).filter(
            User.created_by_id == current_user.id,
            User.role == UserRole.AGENT
        ).count()
        
        # Get payment wallet address
        wallet_setting = session.query(SystemSettings).filter(
            SystemSettings.key == "payment_wallet_address"
        ).first()
        
        payment_wallet = wallet_setting.value if wallet_setting else None
        
        # Check subscription status
        is_active = current_user.is_subscription_active()
        subscription_end = None
        days_remaining = None
        
        if current_user.subscription_end_date:
            subscription_end = current_user.subscription_end_date.isoformat()
            if current_user.subscription_end_date > datetime.utcnow():
                delta = current_user.subscription_end_date - datetime.utcnow()
                days_remaining = delta.days
        
        return BillingInfo(
            current_agents=current_agents,
            max_agents=current_user.max_agents or 0,
            available_slots=(current_user.max_agents or 0) - current_agents,
            price_per_agent=PRICE_PER_AGENT,
            payment_wallet=payment_wallet,
            is_subscription_active=is_active,
            subscription_end_date=subscription_end,
            days_remaining=days_remaining
        )
    except Exception as e:
        logger.error(f"Error getting billing info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@billing_router.post("/request", response_model=PaymentRequestResponse)
async def create_payment_request(
    request_data: PaymentRequestCreate,
    current_user: User = Depends(get_current_admin)
):
    """Create a payment request for additional agent slots"""
    try:
        session = get_session()
        
        # Check if there's already a pending request
        pending_request = session.query(PaymentRequest).filter(
            PaymentRequest.admin_id == current_user.id,
            PaymentRequest.status == PaymentRequestStatus.PENDING
        ).first()
        
        if pending_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You already have a pending payment request"
            )
        
        # Calculate total amount
        total_amount = request_data.num_agents * PRICE_PER_AGENT
        
        # Create payment request
        payment_request = PaymentRequest(
            admin_id=current_user.id,
            num_agents=request_data.num_agents,
            total_amount=total_amount,
            payment_notes=request_data.payment_notes,
            status=PaymentRequestStatus.PENDING
        )
        
        session.add(payment_request)
        session.commit()
        session.refresh(payment_request)
        
        return PaymentRequestResponse(
            id=payment_request.id,
            admin_id=payment_request.admin_id,
            admin_username=current_user.username,
            admin_organization=current_user.organization,
            num_agents=payment_request.num_agents,
            total_amount=payment_request.total_amount,
            status=get_enum_value(payment_request.status),
            payment_notes=payment_request.payment_notes,
            admin_notes=payment_request.admin_notes,
            created_at=payment_request.created_at,
            updated_at=payment_request.updated_at,
            approved_at=payment_request.approved_at
        )
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating payment request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@billing_router.get("/requests", response_model=List[PaymentRequestResponse])
async def get_my_payment_requests(current_user: User = Depends(get_current_admin)):
    """Get all payment requests for current admin"""
    try:
        session = get_session()
        
        logger.info(f"Fetching payment requests for admin_id: {current_user.id}")
        
        requests = session.query(PaymentRequest).filter(
            PaymentRequest.admin_id == current_user.id
        ).order_by(PaymentRequest.created_at.desc()).all()
        
        logger.info(f"Found {len(requests)} payment requests")
        
        result = []
        for req in requests:
            logger.info(f"Request ID: {req.id}, Status: {get_enum_value(req.status)}, Admin ID: {req.admin_id}")
            result.append(
                PaymentRequestResponse(
                    id=req.id,
                    admin_id=req.admin_id,
                    admin_username=current_user.username,
                    admin_organization=current_user.organization,
                    num_agents=req.num_agents,
                    total_amount=req.total_amount,
                    status=get_enum_value(req.status),
                    payment_notes=req.payment_notes,
                    admin_notes=req.admin_notes,
                    created_at=req.created_at,
                    updated_at=req.updated_at,
                    approved_at=req.approved_at
                )
            )
        
        return result
    except Exception as e:
        logger.error(f"Error getting payment requests: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@billing_router.get("/slot-adjustments")
async def get_my_slot_adjustments(current_user: User = Depends(get_current_admin)):
    """Get slot adjustment history for current admin"""
    try:
        session = get_session()
        
        # Get all slot adjustments for this admin
        adjustments = session.query(SlotAdjustment).filter(
            SlotAdjustment.admin_id == current_user.id
        ).order_by(SlotAdjustment.created_at.desc()).all()
        
        result = []
        for adj in adjustments:
            # Get the superadmin who made the adjustment
            adjusted_by = session.query(User).filter(User.id == adj.adjusted_by_id).first()
            
            result.append({
                "id": adj.id,
                "adjusted_by_username": adjusted_by.username if adjusted_by else "System",
                "slots_change": adj.slots_change,
                "reason": adj.reason,
                "previous_max_agents": adj.previous_max_agents,
                "new_max_agents": adj.new_max_agents,
                "created_at": adj.created_at
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting slot adjustments: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Export
__all__ = ['billing_router', 'PRICE_PER_AGENT']

