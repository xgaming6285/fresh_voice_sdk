# -*- coding: utf-8 -*-
"""
CRM API Endpoints for Voice Agent
Handles leads, campaigns, and call orchestration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import json
import csv
import io
import asyncio
import logging
from pathlib import Path

from crm_database import (
    get_session, Lead, Campaign, CampaignLead, CallSession, User,
    Gender, CampaignStatus, CallStatus, UserRole,
    LeadManager, CampaignManager
)
from crm_auth import get_current_user, get_current_admin, check_subscription, user_to_dict

logger = logging.getLogger(__name__)

# Create API router
crm_router = APIRouter(prefix="/api/crm", tags=["CRM"])

# Pydantic models for API
class LeadCreate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: str
    country: Optional[str] = None
    country_code: Optional[str] = None
    gender: Optional[str] = "unknown"
    address: Optional[str] = None
    notes: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None

class LeadUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    custom_data: Optional[Dict[str, Any]] = None

class LeadResponse(BaseModel):
    id: int
    owner_id: int
    owner_name: str  # Name of agent who created this lead
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]
    phone: str
    country: Optional[str]
    country_code: Optional[str]
    gender: str
    address: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_called_at: Optional[datetime]
    call_count: int
    notes: Optional[str]
    full_phone: str
    full_name: str

class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    bot_config: Optional[Dict[str, Any]] = None
    dialing_config: Optional[Dict[str, Any]] = None
    schedule_config: Optional[Dict[str, Any]] = None

class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    bot_config: Optional[Dict[str, Any]] = None
    dialing_config: Optional[Dict[str, Any]] = None
    schedule_config: Optional[Dict[str, Any]] = None

class CampaignResponse(BaseModel):
    id: int
    owner_id: int
    owner_name: str  # Name of agent who created this campaign
    name: str
    description: Optional[str]
    status: str
    bot_config: Dict[str, Any]
    dialing_config: Dict[str, Any]
    schedule_config: Dict[str, Any]
    total_leads: int
    leads_called: int
    leads_answered: int
    leads_rejected: int
    leads_failed: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

class AddLeadsToCampaign(BaseModel):
    lead_ids: List[int]
    priority: int = 0

class LeadFilter(BaseModel):
    countries: Optional[List[str]] = None
    min_call_count: Optional[int] = None
    max_call_count: Optional[int] = None
    never_called: Optional[bool] = None

class CallSessionResponse(BaseModel):
    id: int
    session_id: str
    campaign_id: Optional[int]
    lead_id: Optional[int]
    caller_id: Optional[str]
    called_number: Optional[str]
    status: str
    started_at: datetime
    answered_at: Optional[datetime]
    ended_at: Optional[datetime]
    duration: Optional[int]
    talk_time: Optional[int]
    recording_path: Optional[str]
    transcript_status: Optional[str]
    transcript_language: Optional[str]
    sentiment_score: Optional[float]
    interest_level: Optional[int]
    key_points: Optional[List[str]]
    follow_up_required: bool
    follow_up_notes: Optional[str]

# Helper functions
def build_lead_response(lead: Lead, session) -> LeadResponse:
    """Build LeadResponse from Lead object"""
    owner = session.query(User).get(lead.owner_id)
    owner_name = owner.full_name if owner else "Unknown"
    
    return LeadResponse(
        id=lead.id,
        owner_id=lead.owner_id,
        owner_name=owner_name,
        first_name=lead.first_name,
        last_name=lead.last_name,
        email=lead.email,
        phone=lead.phone,
        country=lead.country,
        country_code=lead.country_code,
        gender=lead.gender.value if lead.gender else "unknown",
        address=lead.address,
        created_at=lead.created_at,
        updated_at=lead.updated_at,
        last_called_at=lead.last_called_at,
        call_count=lead.call_count,
        notes=lead.notes,
        full_phone=lead.full_phone,
        full_name=lead.full_name
    )

def build_campaign_response(campaign: Campaign, session) -> CampaignResponse:
    """Build CampaignResponse from Campaign object"""
    owner = session.query(User).get(campaign.owner_id)
    owner_name = owner.full_name if owner else "Unknown"
    
    return CampaignResponse(
        id=campaign.id,
        owner_id=campaign.owner_id,
        owner_name=owner_name,
        name=campaign.name,
        description=campaign.description,
        status=campaign.status.value,
        bot_config=campaign.bot_config or {},
        dialing_config=campaign.dialing_config or {},
        schedule_config=campaign.schedule_config or {},
        total_leads=campaign.total_leads,
        leads_called=campaign.leads_called,
        leads_answered=campaign.leads_answered,
        leads_rejected=campaign.leads_rejected,
        leads_failed=campaign.leads_failed,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
        started_at=campaign.started_at,
        completed_at=campaign.completed_at
    )

def get_accessible_lead_ids(user: User, session) -> List[int]:
    """Get IDs of all leads accessible to user (their own + their agents' if admin)"""
    if user.role == UserRole.ADMIN:
        # Admin can see their own leads and their agents' leads
        # Query agents directly from session instead of using lazy-loaded relationship
        from crm_database import UserManager
        user_manager = UserManager(session)
        agents = user_manager.get_agents_by_admin(user.id)
        agent_ids = [agent.id for agent in agents]
        accessible_ids = [user.id] + agent_ids
        leads = session.query(Lead.id).filter(Lead.owner_id.in_(accessible_ids)).all()
        return [lead.id for lead in leads]
    else:
        # Agent can only see their own leads
        leads = session.query(Lead.id).filter(Lead.owner_id == user.id).all()
        return [lead.id for lead in leads]

def get_accessible_campaign_ids(user: User, session) -> List[int]:
    """Get IDs of all campaigns accessible to user (their own + their agents' if admin)"""
    if user.role == UserRole.ADMIN:
        # Admin can see their own campaigns and their agents' campaigns
        # Query agents directly from session instead of using lazy-loaded relationship
        from crm_database import UserManager
        user_manager = UserManager(session)
        agents = user_manager.get_agents_by_admin(user.id)
        agent_ids = [agent.id for agent in agents]
        accessible_ids = [user.id] + agent_ids
        campaigns = session.query(Campaign.id).filter(Campaign.owner_id.in_(accessible_ids)).all()
        return [campaign.id for campaign in campaigns]
    else:
        # Agent can only see their own campaigns
        campaigns = session.query(Campaign.id).filter(Campaign.owner_id == user.id).all()
        return [campaign.id for campaign in campaigns]

# Lead endpoints
@crm_router.post("/leads", response_model=LeadResponse)
async def create_lead(lead: LeadCreate, current_user: User = Depends(check_subscription)):
    """Create a new lead"""
    try:
        session = get_session()
        lead_manager = LeadManager(session)
        
        # Convert string enums to proper enum values
        lead_data = lead.dict()
        lead_data['owner_id'] = current_user.id
        lead_data['gender'] = Gender(lead_data.get('gender', 'unknown'))
        
        db_lead = lead_manager.create_lead(lead_data)
        
        return build_lead_response(db_lead, session)
    except Exception as e:
        logger.error(f"Error creating lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/leads", response_model=Dict[str, Any])
async def get_leads(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=1000),
    country: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get paginated list of leads"""
    try:
        session = get_session()
        
        # Build query with ownership filter
        query = session.query(Lead)
        
        # Filter by accessible leads (user's own + their agents' if admin)
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        query = query.filter(Lead.id.in_(accessible_lead_ids))
        
        if country:
            query = query.filter(Lead.country == country)
        
        total = query.count()
        offset = (page - 1) * per_page
        
        leads = query.limit(per_page).offset(offset).all()
        
        leads_response = [build_lead_response(lead, session) for lead in leads]
        
        return {
            "leads": leads_response,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    except Exception as e:
        logger.error(f"Error getting leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/leads/{lead_id}", response_model=LeadResponse)
async def get_lead(lead_id: int, current_user: User = Depends(get_current_user)):
    """Get a specific lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Check access permission
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        if lead.id not in accessible_lead_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return build_lead_response(lead, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.put("/leads/{lead_id}", response_model=LeadResponse)
async def update_lead(lead_id: int, lead_update: LeadUpdate, current_user: User = Depends(check_subscription)):
    """Update a lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Check access permission
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        if lead.id not in accessible_lead_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        update_data = lead_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                if field == 'gender':
                    value = Gender(value)
                setattr(lead, field, value)
        
        session.commit()
        
        return build_lead_response(lead, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: int, current_user: User = Depends(check_subscription)):
    """Delete a lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Check access permission
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        if lead.id not in accessible_lead_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        session.delete(lead)
        session.commit()
        
        return {"status": "success", "message": "Lead deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/leads/import")
async def import_leads(file: UploadFile = File(...), current_user: User = Depends(check_subscription)):
    """Import leads from CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        text_content = contents.decode('utf-8')
        
        session = get_session()
        lead_manager = LeadManager(session)
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(text_content))
        leads_data = []
        
        for row in csv_reader:
            # Map CSV columns to lead fields
            lead_data = {
                'owner_id': current_user.id,
                'first_name': row.get('first_name', '').strip(),
                'last_name': row.get('last_name', '').strip(),
                'email': row.get('email', '').strip(),
                'phone': row.get('phone', '').strip(),
                'country': row.get('country', '').strip(),
                'country_code': row.get('prefix', row.get('country_code', '')).strip(),
                'gender': Gender(row.get('gender', 'unknown').lower()),
                'address': row.get('address', '').strip()
            }
            
            # Validate required fields
            if not lead_data['phone']:
                continue
            
            leads_data.append(lead_data)
        
        # Import leads
        import_batch_id = f"csv_import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        imported_count = lead_manager.bulk_import_leads(leads_data, import_batch_id)
        
        return {
            "status": "success",
            "imported": imported_count,
            "import_batch_id": import_batch_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Campaign endpoints
@crm_router.post("/campaigns", response_model=CampaignResponse)
async def create_campaign(campaign: CampaignCreate, current_user: User = Depends(check_subscription)):
    """Create a new campaign"""
    try:
        session = get_session()
        campaign_manager = CampaignManager(session)
        
        db_campaign = campaign_manager.create_campaign(
            owner_id=current_user.id,
            name=campaign.name,
            description=campaign.description,
            bot_config=campaign.bot_config
        )
        
        # Update with dialing and schedule config if provided
        if campaign.dialing_config:
            db_campaign.dialing_config = campaign.dialing_config
        if campaign.schedule_config:
            db_campaign.schedule_config = campaign.schedule_config
        session.commit()
        
        return build_campaign_response(db_campaign, session)
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns", response_model=List[CampaignResponse])
async def get_campaigns(status: Optional[str] = None, current_user: User = Depends(get_current_user)):
    """Get list of campaigns"""
    try:
        session = get_session()
        
        # Filter by accessible campaigns
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        query = session.query(Campaign).filter(Campaign.id.in_(accessible_campaign_ids))
        
        if status:
            query = query.filter(Campaign.status == CampaignStatus(status))
        
        campaigns = query.order_by(Campaign.created_at.desc()).all()
        
        return [build_campaign_response(c, session) for c in campaigns]
    except Exception as e:
        logger.error(f"Error getting campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: int, current_user: User = Depends(get_current_user)):
    """Get a specific campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Check access permission
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return build_campaign_response(campaign, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.put("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(campaign_id: int, campaign_update: CampaignUpdate, current_user: User = Depends(check_subscription)):
    """Update a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Check access permission
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update fields
        update_data = campaign_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                if field == 'status':
                    value = CampaignStatus(value)
                setattr(campaign, field, value)
        
        session.commit()
        
        return build_campaign_response(campaign, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/leads")
async def add_leads_to_campaign(campaign_id: int, request: AddLeadsToCampaign, current_user: User = Depends(check_subscription)):
    """Add leads to a campaign"""
    try:
        session = get_session()
        campaign_manager = CampaignManager(session)
        
        # Verify campaign exists and check access
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Verify all leads are accessible
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        for lead_id in request.lead_ids:
            if lead_id not in accessible_lead_ids:
                raise HTTPException(status_code=403, detail=f"Access denied to lead {lead_id}")
        
        # Add leads
        added_count = campaign_manager.add_leads_to_campaign(
            campaign_id=campaign_id,
            lead_ids=request.lead_ids,
            priority=request.priority
        )
        
        return {
            "status": "success",
            "leads_added": added_count,
            "campaign_id": campaign_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding leads to campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/leads/filter")
async def add_filtered_leads_to_campaign(campaign_id: int, filter: LeadFilter, current_user: User = Depends(get_current_user)):
    """Add leads to campaign based on filter criteria"""
    try:
        session = get_session()
        
        # Verify campaign exists and check access
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Build query with ownership filter
        accessible_lead_ids = get_accessible_lead_ids(current_user, session)
        query = session.query(Lead).filter(Lead.id.in_(accessible_lead_ids))
        
        if filter.countries:
            query = query.filter(Lead.country.in_(filter.countries))
        if filter.min_call_count is not None:
            query = query.filter(Lead.call_count >= filter.min_call_count)
        if filter.max_call_count is not None:
            query = query.filter(Lead.call_count <= filter.max_call_count)
        if filter.never_called:
            query = query.filter(Lead.call_count == 0)
        
        leads = query.all()
        lead_ids = [lead.id for lead in leads]
        
        if not lead_ids:
            return {
                "status": "success",
                "leads_added": 0,
                "campaign_id": campaign_id,
                "message": "No leads matched the filter criteria"
            }
        
        # Add leads to campaign
        campaign_manager = CampaignManager(session)
        added_count = campaign_manager.add_leads_to_campaign(
            campaign_id=campaign_id,
            lead_ids=lead_ids,
            priority=0
        )
        
        return {
            "status": "success",
            "leads_added": added_count,
            "campaign_id": campaign_id
        }
    except Exception as e:
        logger.error(f"Error adding filtered leads to campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns/{campaign_id}/leads")
async def get_campaign_leads(
    campaign_id: int,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=1000),
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get leads in a campaign"""
    try:
        session = get_session()
        
        # Verify campaign exists and check access
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        query = session.query(CampaignLead).filter(
            CampaignLead.campaign_id == campaign_id
        )
        
        if status:
            query = query.filter(CampaignLead.status == CallStatus(status))
        
        total = query.count()
        offset = (page - 1) * per_page
        
        campaign_leads = query.limit(per_page).offset(offset).all()
        
        results = []
        for cl in campaign_leads:
            lead = cl.lead
            results.append({
                "id": cl.id,
                "lead_id": cl.lead_id,
                "lead_name": lead.full_name,
                "lead_phone": lead.full_phone,
                "lead_country": lead.country,
                "status": cl.status.value,
                "priority": cl.priority,
                "call_attempts": cl.call_attempts,
                "last_attempt_at": cl.last_attempt_at,
                "scheduled_for": cl.scheduled_for,
                "call_session_id": cl.call_session_id,
                "call_duration": cl.call_duration,
                "call_result": cl.call_result,
                "completed_at": cl.completed_at
            })
        
        return {
            "campaign_leads": results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
    except Exception as e:
        logger.error(f"Error getting campaign leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/start", response_model=Dict[str, Any])
async def start_campaign(campaign_id: int, background_tasks: BackgroundTasks, current_user: User = Depends(check_subscription)):
    """Start a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Check access permission
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if campaign.status not in [CampaignStatus.DRAFT, CampaignStatus.READY, CampaignStatus.PAUSED]:
            raise HTTPException(
                status_code=400, 
                detail=f"Campaign cannot be started from status: {campaign.status.value}"
            )
        
        # Check if campaign has leads
        lead_count = session.query(CampaignLead).filter(
            CampaignLead.campaign_id == campaign_id
        ).count()
        
        if lead_count == 0:
            raise HTTPException(status_code=400, detail="Campaign has no leads")
        
        # Update campaign status
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.utcnow()
        session.commit()
        
        # Start campaign execution in background
        background_tasks.add_task(execute_campaign, campaign_id)
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "message": "Campaign started",
            "total_leads": lead_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/pause")
async def pause_campaign(campaign_id: int, current_user: User = Depends(get_current_user)):
    """Pause a running campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Check access permission
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if campaign.status != CampaignStatus.RUNNING:
            raise HTTPException(
                status_code=400,
                detail=f"Campaign is not running (status: {campaign.status.value})"
            )
        
        campaign.status = CampaignStatus.PAUSED
        session.commit()
        
        return {"status": "success", "message": "Campaign paused"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/stop")
async def stop_campaign(campaign_id: int, current_user: User = Depends(get_current_user)):
    """Stop a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Check access permission
        accessible_campaign_ids = get_accessible_campaign_ids(current_user, session)
        if campaign.id not in accessible_campaign_ids:
            raise HTTPException(status_code=403, detail="Access denied")
        
        campaign.status = CampaignStatus.CANCELLED
        campaign.completed_at = datetime.utcnow()
        session.commit()
        
        return {"status": "success", "message": "Campaign stopped"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Call session endpoints
@crm_router.get("/sessions", response_model=List[CallSessionResponse])
async def get_call_sessions(
    campaign_id: Optional[int] = None,
    lead_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user)
):
    """Get call sessions (filtered by user ownership)"""
    try:
        session = get_session()
        
        # Get accessible call session IDs based on user role
        # IMPORTANT: Always filter out sessions with NULL owner_id (orphaned sessions)
        accessible_session_ids = []
        if current_user.role == UserRole.SUPERADMIN:
            # Superadmin sees all call sessions (except orphaned ones)
            query = session.query(CallSession).filter(CallSession.owner_id.isnot(None))
        elif current_user.role == UserRole.ADMIN:
            # Admin sees their own sessions + their agents' sessions
            from crm_database import UserManager
            user_manager = UserManager(session)
            agents = user_manager.get_agents_by_admin(current_user.id)
            agent_ids = [agent.id for agent in agents]
            accessible_user_ids = [current_user.id] + agent_ids
            query = session.query(CallSession).filter(
                CallSession.owner_id.in_(accessible_user_ids),
                CallSession.owner_id.isnot(None)  # Extra safety: exclude NULL owner_id
            )
        else:  # AGENT
            # Agent only sees their own sessions
            query = session.query(CallSession).filter(
                CallSession.owner_id == current_user.id,
                CallSession.owner_id.isnot(None)  # Extra safety: exclude NULL owner_id
            )
        
        if campaign_id:
            query = query.filter(CallSession.campaign_id == campaign_id)
        if lead_id:
            query = query.filter(CallSession.lead_id == lead_id)
        if status:
            query = query.filter(CallSession.status == CallStatus(status))
        
        sessions = query.order_by(CallSession.started_at.desc()).limit(limit).all()
        
        return [
            CallSessionResponse(
                id=s.id,
                session_id=s.session_id,
                campaign_id=s.campaign_id,
                lead_id=s.lead_id,
                caller_id=s.caller_id,
                called_number=s.called_number,
                status=s.status.value,
                started_at=s.started_at,
                answered_at=s.answered_at,
                ended_at=s.ended_at,
                duration=s.duration,
                talk_time=s.talk_time,
                recording_path=s.recording_path,
                transcript_status=s.transcript_status,
                transcript_language=s.transcript_language,
                sentiment_score=s.sentiment_score,
                interest_level=s.interest_level,
                key_points=s.key_points,
                follow_up_required=s.follow_up_required,
                follow_up_notes=s.follow_up_notes
            ) for s in sessions
        ]
    except Exception as e:
        logger.error(f"Error getting call sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/sessions/{session_id}", response_model=CallSessionResponse)
async def get_call_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific call session by voice agent session ID"""
    try:
        session = get_session()
        call_session = session.query(CallSession).filter(
            CallSession.session_id == session_id
        ).first()
        
        if not call_session:
            # Check if recording exists on disk but not in database (orphaned recording)
            import os
            from pathlib import Path
            recording_dir = Path("sessions") / session_id
            if recording_dir.exists():
                logger.warning(f"Found orphaned recording for {session_id}, creating database entry")
                # Create database entry for orphaned recording
                call_session = CallSession(
                    session_id=session_id,
                    owner_id=current_user.id,  # Assign to current user
                    status=CallStatus.COMPLETED,
                    started_at=datetime.utcnow(),
                    ended_at=datetime.utcnow()
                )
                session.add(call_session)
                session.commit()
                session.refresh(call_session)
            else:
                raise HTTPException(status_code=404, detail="Call session not found")
        
        # Check access rights
        if current_user.role == UserRole.AGENT:
            # Agent can only see their own sessions
            if call_session.owner_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
        elif current_user.role == UserRole.ADMIN:
            # Admin can see their own sessions + their agents' sessions
            from crm_database import UserManager
            user_manager = UserManager(session)
            agents = user_manager.get_agents_by_admin(current_user.id)
            accessible_user_ids = [current_user.id] + [agent.id for agent in agents]
            if call_session.owner_id not in accessible_user_ids:
                raise HTTPException(status_code=403, detail="Access denied")
        # Superadmin can see all sessions
        
        return CallSessionResponse(
            id=call_session.id,
            session_id=call_session.session_id,
            campaign_id=call_session.campaign_id,
            lead_id=call_session.lead_id,
            caller_id=call_session.caller_id,
            called_number=call_session.called_number,
            status=call_session.status.value,
            started_at=call_session.started_at,
            answered_at=call_session.answered_at,
            ended_at=call_session.ended_at,
            duration=call_session.duration,
            talk_time=call_session.talk_time,
            recording_path=call_session.recording_path,
            transcript_status=call_session.transcript_status,
            transcript_language=call_session.transcript_language,
            sentiment_score=call_session.sentiment_score,
            interest_level=call_session.interest_level,
            key_points=call_session.key_points,
            follow_up_required=call_session.follow_up_required,
            follow_up_notes=call_session.follow_up_notes
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Campaign execution function
async def execute_campaign(campaign_id: int):
    """Execute a campaign - make calls to leads"""
    logger.info(f"Starting campaign execution for campaign {campaign_id}")
    
    # Import here to avoid circular imports
    from windows_voice_agent import sip_handler
    
    session = get_session()
    campaign_manager = CampaignManager(session)
    
    try:
        while True:
            # Check campaign status
            campaign = session.query(Campaign).get(campaign_id)
            if not campaign or campaign.status != CampaignStatus.RUNNING:
                logger.info(f"Campaign {campaign_id} is no longer running")
                break
            
            # Get next lead to call
            campaign_lead = campaign_manager.get_next_lead_to_call(campaign_id)
            if not campaign_lead:
                logger.info(f"No more leads to call in campaign {campaign_id}")
                campaign.status = CampaignStatus.COMPLETED
                campaign.completed_at = datetime.utcnow()
                session.commit()
                break
            
            lead = campaign_lead.lead
            logger.info(f"Calling lead {lead.id}: {lead.full_phone}")
            
            # Create call session record
            call_session = CallSession(
                session_id=f"campaign_{campaign_id}_lead_{lead.id}_{int(datetime.utcnow().timestamp())}",
                campaign_id=campaign_id,
                lead_id=lead.id,
                owner_id=current_user.id,  # Set owner to the user who started the campaign
                caller_id="+359898995151",  # From config
                called_number=lead.full_phone,
                status=CallStatus.DIALING
            )
            session.add(call_session)
            session.commit()
            
            # Make the call using the voice agent
            try:
                voice_session_id = sip_handler.make_outbound_call(lead.full_phone)
                
                if voice_session_id:
                    # Update call session with voice agent session ID
                    call_session.session_id = voice_session_id
                    call_session.status = CallStatus.RINGING
                    session.commit()
                    
                    # Wait for call to complete (simplified - in production you'd monitor call status)
                    await asyncio.sleep(5)  # Give time for call to establish
                    
                    # TODO: Monitor actual call status from voice agent
                    # For now, we'll simulate success
                    call_session.status = CallStatus.ANSWERED
                    call_session.answered_at = datetime.utcnow()
                    
                    # Update campaign lead status
                    campaign_manager.update_call_result(
                        campaign_id=campaign_id,
                        lead_id=lead.id,
                        session_id=voice_session_id,
                        status=CallStatus.ANSWERED
                    )
                else:
                    # Call failed to initiate
                    call_session.status = CallStatus.FAILED
                    campaign_lead.status = CallStatus.FAILED
                    session.commit()
                    
            except Exception as e:
                logger.error(f"Error making call to {lead.full_phone}: {e}")
                call_session.status = CallStatus.FAILED
                campaign_lead.status = CallStatus.FAILED
                session.commit()
            
            # Wait before next call (configurable in dialing_config)
            dialing_config = campaign.dialing_config or {}
            wait_between_calls = dialing_config.get('wait_between_calls', 5)
            await asyncio.sleep(wait_between_calls)
            
    except Exception as e:
        logger.error(f"Error executing campaign {campaign_id}: {e}")
        campaign = session.query(Campaign).get(campaign_id)
        if campaign:
            campaign.status = CampaignStatus.CANCELLED
            session.commit()
    finally:
        session.close()

# Export router
__all__ = ['crm_router']
