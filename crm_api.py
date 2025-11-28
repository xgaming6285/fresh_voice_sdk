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
    LeadManager, CampaignManager, joinedload, get_enum_value
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

class CampaignStartRequest(BaseModel):
    call_config: Optional[Dict[str, Any]] = None

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
    lead_country: Optional[str]
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
    # New fields from MongoDB
    transcripts: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    audio_files: Optional[Dict[str, str]] = None
    session_info: Optional[Dict[str, Any]] = None
    asterisk_linkedid: Optional[str] = None

# Helper functions
def build_call_session_response(call_session: CallSession) -> CallSessionResponse:
    """Build CallSessionResponse from CallSession object with MongoDB data"""
    # Extract duration from session_info if not available directly
    duration = call_session.duration
    session_info = getattr(call_session, 'session_info', None)
    if duration is None and session_info and isinstance(session_info, dict):
        duration = session_info.get('duration_seconds')
        # Convert to integer if it's a float
        if duration is not None:
            duration = int(duration)
    
    return CallSessionResponse(
        id=call_session.id,
        session_id=call_session.session_id,
        campaign_id=call_session.campaign_id,
        lead_id=call_session.lead_id,
        lead_country=call_session.lead.country if call_session.lead else None,
        caller_id=str(call_session.caller_id) if call_session.caller_id is not None else None,
        called_number=call_session.called_number,
        status=get_enum_value(call_session.status),
        started_at=call_session.started_at,
        answered_at=call_session.answered_at,
        ended_at=call_session.ended_at,
        duration=duration,
        talk_time=call_session.talk_time,
        recording_path=call_session.recording_path,
        transcript_status=call_session.transcript_status,
        transcript_language=call_session.transcript_language,
        sentiment_score=call_session.sentiment_score,
        interest_level=call_session.interest_level,
        key_points=call_session.key_points,
        follow_up_required=call_session.follow_up_required,
        follow_up_notes=call_session.follow_up_notes,
        # Include MongoDB fields if they exist
        transcripts=getattr(call_session, 'transcripts', None),
        analysis=getattr(call_session, 'analysis', None),
        audio_files=getattr(call_session, 'audio_files', None),
        session_info=session_info,
        asterisk_linkedid=getattr(call_session, 'asterisk_linkedid', None)
    )

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
        gender=get_enum_value(lead.gender) if lead.gender else "unknown",
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
        status=get_enum_value(campaign.status),
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
        leads = session.query(Lead).filter(Lead.owner_id.in_(accessible_ids)).all()
        return [lead.id for lead in leads]
    else:
        # Agent can only see their own leads
        leads = session.query(Lead).filter(Lead.owner_id == user.id).all()
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
        campaigns = session.query(Campaign).filter(Campaign.owner_id.in_(accessible_ids)).all()
        return [campaign.id for campaign in campaigns]
    else:
        # Agent can only see their own campaigns
        campaigns = session.query(Campaign).filter(Campaign.owner_id == user.id).all()
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
        
        # Build query with ownership filter - directly filter by owner_id instead of pre-fetching IDs
        query = session.query(Lead)
        
        # Get accessible owner IDs (not lead IDs) - more efficient
        if current_user.role == UserRole.SUPERADMIN:
            # Superadmin can see all leads
            pass  # No filter needed
        elif current_user.role == UserRole.ADMIN:
            # Admin can see their own leads and their agents' leads
            from crm_database import UserManager
            user_manager = UserManager(session)
            agents = user_manager.get_agents_by_admin(current_user.id)
            accessible_owner_ids = [current_user.id] + [agent.id for agent in agents]
            query = query.filter(Lead.owner_id.in_(accessible_owner_ids))
        else:
            # Agent can only see their own leads
            query = query.filter(Lead.owner_id == current_user.id)
        
        if country:
            query = query.filter(Lead.country == country)
        
        total = query.count()
        offset = (page - 1) * per_page
        
        leads = query.order_by(Lead.id.desc()).limit(per_page).offset(offset).all()
        
        # Batch-load all owners in ONE query to avoid N+1 problem
        owner_ids = list(set(lead.owner_id for lead in leads if lead.owner_id))
        owners_by_id = {}
        if owner_ids:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            owners_docs = db.users.find({"id": {"$in": owner_ids}})
            owners_by_id = {doc["id"]: doc.get("full_name", "Unknown") for doc in owners_docs}
        
        # Build responses with pre-loaded owner names
        leads_response = []
        for lead in leads:
            owner_name = owners_by_id.get(lead.owner_id, "Unknown")
            leads_response.append(LeadResponse(
                id=lead.id,
                owner_id=lead.owner_id,
                owner_name=owner_name,
                first_name=lead.first_name,
                last_name=lead.last_name,
                email=lead.email,
                phone=lead.phone,
                country=lead.country,
                country_code=lead.country_code,
                gender=get_enum_value(lead.gender) if lead.gender else "unknown",
                address=lead.address,
                created_at=lead.created_at,
                updated_at=lead.updated_at,
                last_called_at=lead.last_called_at,
                call_count=lead.call_count,
                notes=lead.notes,
                full_phone=lead.full_phone,
                full_name=lead.full_name
            ))
        
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
        
        # Save the lead (MongoDB requires explicit save)
        lead.save()
        
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
        
        lead.delete()  # MongoDB requires explicit delete
        
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
        db_campaign.save()  # MongoDB requires explicit save
        
        return build_campaign_response(db_campaign, session)
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns", response_model=Dict[str, Any])
async def get_campaigns(status: Optional[str] = None, current_user: User = Depends(get_current_user)):
    """Get list of campaigns"""
    try:
        session = get_session()
        
        # Build query with ownership filter - directly filter by owner_id instead of pre-fetching IDs
        query = session.query(Campaign)
        
        if current_user.role == UserRole.SUPERADMIN:
            # Superadmin can see all campaigns
            pass  # No filter needed
        elif current_user.role == UserRole.ADMIN:
            # Admin can see their own campaigns and their agents' campaigns
            from crm_database import UserManager
            user_manager = UserManager(session)
            agents = user_manager.get_agents_by_admin(current_user.id)
            accessible_owner_ids = [current_user.id] + [agent.id for agent in agents]
            query = query.filter(Campaign.owner_id.in_(accessible_owner_ids))
        else:
            # Agent can only see their own campaigns
            query = query.filter(Campaign.owner_id == current_user.id)
        
        if status:
            query = query.filter(Campaign.status == CampaignStatus(status))
        
        campaigns = query.order_by(Campaign.created_at.desc()).all()
        
        # Batch-load all owners in ONE query to avoid N+1 problem
        owner_ids = list(set(c.owner_id for c in campaigns if c.owner_id))
        owners_by_id = {}
        if owner_ids:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            owners_docs = db.users.find({"id": {"$in": owner_ids}})
            owners_by_id = {doc["id"]: doc.get("full_name", "Unknown") for doc in owners_docs}
        
        # Build responses with pre-loaded owner names
        campaigns_response = []
        for campaign in campaigns:
            owner_name = owners_by_id.get(campaign.owner_id, "Unknown")
            campaigns_response.append(CampaignResponse(
                id=campaign.id,
                owner_id=campaign.owner_id,
                owner_name=owner_name,
                name=campaign.name,
                description=campaign.description,
                status=get_enum_value(campaign.status),
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
            ))
        
        return {"campaigns": campaigns_response}
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
        
        # Save the campaign (MongoDB requires explicit save)
        campaign.save()
        
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
                "status": get_enum_value(cl.status),
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
async def start_campaign(
    campaign_id: int, 
    request: Optional[CampaignStartRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(), 
    current_user: User = Depends(check_subscription)
):
    """Start a campaign with optional call configuration"""
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
                detail=f"Campaign cannot be started from status: {get_enum_value(campaign.status)}"
            )
        
        # Check if campaign has leads
        lead_count = session.query(CampaignLead).filter(
            CampaignLead.campaign_id == campaign_id
        ).count()
        
        if lead_count == 0:
            raise HTTPException(status_code=400, detail="Campaign has no leads")
        
        # Store call config in bot_config if provided
        if request and request.call_config:
            if not campaign.bot_config:
                campaign.bot_config = {}
            campaign.bot_config['call_config'] = request.call_config
            logger.info(f"📞 Campaign {campaign_id} will use custom call config: {request.call_config.get('call_objective', 'N/A')}")
        
        # Update campaign status
        campaign.status = CampaignStatus.RUNNING
        campaign.started_at = datetime.utcnow()
        campaign.save()  # MongoDB requires explicit save
        
        # Start campaign execution in background
        background_tasks.add_task(execute_campaign, campaign_id)
        
        return {
            "status": "success",
            "campaign_id": campaign_id,
            "message": "Campaign started",
            "total_leads": lead_count,
            "has_call_config": bool(request and request.call_config)
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
                detail=f"Campaign is not running (status: {get_enum_value(campaign.status)})"
            )
        
        campaign.status = CampaignStatus.PAUSED
        campaign.save()  # MongoDB requires explicit save
        
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
        campaign.save()  # MongoDB requires explicit save
        
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
            query = session.query(CallSession).options(joinedload(CallSession.lead)).filter(CallSession.owner_id.isnot(None))
        elif current_user.role == UserRole.ADMIN:
            # Admin sees their own sessions + their agents' sessions
            from crm_database import UserManager
            user_manager = UserManager(session)
            agents = user_manager.get_agents_by_admin(current_user.id)
            agent_ids = [agent.id for agent in agents]
            accessible_user_ids = [current_user.id] + agent_ids
            query = session.query(CallSession).options(joinedload(CallSession.lead)).filter(
                CallSession.owner_id.in_(accessible_user_ids)
            )
        else:  # AGENT
            # Agent only sees their own sessions
            query = session.query(CallSession).options(joinedload(CallSession.lead)).filter(
                CallSession.owner_id == current_user.id
            )
        
        if campaign_id:
            query = query.filter(CallSession.campaign_id == campaign_id)
        if lead_id:
            query = query.filter(CallSession.lead_id == lead_id)
        if status:
            query = query.filter(CallSession.status == CallStatus(status))
        
        sessions = query.order_by(CallSession.started_at.desc()).limit(limit).all()
        
        # Batch-load all leads in ONE query to avoid N+1 problem
        lead_ids = [s.lead_id for s in sessions if s.lead_id]
        if lead_ids:
            from crm_database_mongodb import MongoDB
            db = MongoDB.get_db()
            leads_docs = db.leads.find({"id": {"$in": lead_ids}})
            leads_by_id = {doc["id"]: Lead.from_dict(doc) for doc in leads_docs}
            # Attach leads to sessions
            for s in sessions:
                if s.lead_id and s.lead_id in leads_by_id:
                    s._lead = leads_by_id[s.lead_id]
        
        return [build_call_session_response(s) for s in sessions]
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
        call_session = session.query(CallSession).options(joinedload(CallSession.lead)).filter(
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
                call_session.save()  # MongoDB requires explicit save
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
        
        return build_call_session_response(call_session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

# Campaign execution function
async def execute_campaign(campaign_id: int):
    """Execute a campaign - make calls to leads sequentially"""
    logger.info(f"Starting campaign execution for campaign {campaign_id}")
    
    # Import here to avoid circular imports
    from windows_voice_agent import sip_handler
    from crm_database import UserManager
    import requests
    
    session = get_session()
    campaign_manager = CampaignManager(session)
    
    try:
        # Get campaign and check for call config
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            logger.error(f"Campaign {campaign_id} not found")
            return
            
        call_config = None
        cached_greeting_file = None  # Pre-generated greeting for all leads
        cached_greeting_transcript = None
        
        if campaign.bot_config and 'call_config' in campaign.bot_config:
            call_config = campaign.bot_config['call_config']
            logger.info(f"📞 Campaign has custom call config: {call_config.get('call_objective', 'N/A')}")
            
            # Pre-generate greeting ONCE for the entire campaign
            # All leads with same language will reuse this greeting
            try:
                from greeting_generator_gemini import generate_greeting_for_lead
                from windows_voice_agent import detect_caller_country, get_language_config
                
                # Get first lead to detect language
                first_lead_query = session.query(CampaignLead).filter(
                    CampaignLead.campaign_id == campaign_id,
                    CampaignLead.status == CallStatus.PENDING
                ).first()
                
                if first_lead_query:
                    first_lead = first_lead_query.lead
                    logger.info(f"🎤 Pre-generating greeting for campaign (will be reused for all leads)...")
                    
                    # Detect language from first lead
                    caller_country = detect_caller_country(first_lead.full_phone)
                    language_info = get_language_config(caller_country)
                    
                    # Generate greeting
                    greeting_result = await generate_greeting_for_lead(
                        language=language_info['lang'],
                        language_code=language_info['code'],
                        call_config=call_config
                    )
                    
                    if greeting_result and greeting_result.get('success'):
                        cached_greeting_file = greeting_result.get('greeting_file')
                        cached_greeting_transcript = greeting_result.get('transcript')
                        logger.info(f"✅ Campaign greeting pre-generated and cached: {cached_greeting_file}")
                        logger.info(f"   This greeting will be reused for all {campaign.total_leads} leads")
                    else:
                        logger.warning("⚠️ Failed to pre-generate campaign greeting, will generate per-call")
            except Exception as e:
                logger.warning(f"⚠️ Error pre-generating campaign greeting: {e}. Will generate per-call.")
        
        # Get campaign owner (agent) to use their gate slot
        user_manager = UserManager(session)
        campaign_owner = user_manager.get_user_by_id(campaign.owner_id)
        gate_slot = None
        if campaign_owner and campaign_owner.gate_slot:
            gate_slot = campaign_owner.gate_slot
            logger.info(f"📞 Using gate slot {gate_slot} for agent {campaign_owner.username}")
        else:
            logger.warning(f"⚠️ Campaign owner has no assigned gate slot, will use default")
        
        # Get caller ID from agent/campaign owner
        caller_id = campaign_owner.gate_slot if campaign_owner and campaign_owner.gate_slot else "+359898995151"
        
        while True:
            # Check campaign status
            campaign = session.query(Campaign).get(campaign_id)
            logger.info(f"Campaign object: {campaign}, status: {campaign.status if campaign else 'None'}")
            if not campaign or campaign.status != CampaignStatus.RUNNING:
                logger.info(f"Campaign {campaign_id} is no longer running (status: {campaign.status if campaign else 'Not found'})")
                break
            
            # Get next lead to call
            campaign_lead = campaign_manager.get_next_lead_to_call(campaign_id)
            if not campaign_lead:
                logger.info(f"No more leads to call in campaign {campaign_id}")
                campaign.status = CampaignStatus.COMPLETED
                campaign.completed_at = datetime.utcnow()
                campaign.save()  # MongoDB requires explicit save
                break
            
            lead = campaign_lead.lead
            logger.info(f"🎯 Calling lead {lead.id}: {lead.full_name} at {lead.full_phone}")
            
            # Create call session record
            call_session = CallSession(
                session_id=f"campaign_{campaign_id}_lead_{lead.id}_{int(datetime.utcnow().timestamp())}",
                campaign_id=campaign_id,
                lead_id=lead.id,
                owner_id=campaign.owner_id,  # Set owner to the campaign owner
                caller_id=caller_id,
                called_number=lead.full_phone,
                status=CallStatus.DIALING
            )
            call_session.save()  # MongoDB requires explicit save
            
            # Prepare custom config for the call
            # Use cached greeting if available, otherwise generate per-call
            custom_config = {}
            if call_config:
                # Build the prompt from call config (same logic as CustomCallDialog)
                custom_config = {
                    'custom_prompt': build_prompt_from_config(call_config, lead),
                    'voice_name': call_config.get('voice_name', 'Puck'),
                }
                
                # If we have a cached greeting, use it directly (instant playback, no delay)
                if cached_greeting_file:
                    custom_config['greeting_file'] = cached_greeting_file
                    custom_config['greeting_transcript'] = cached_greeting_transcript
                    logger.info(f"📝 Using custom prompt with voice: {custom_config['voice_name']} + cached greeting")
                else:
                    # Otherwise, pass call_config for dynamic generation (has delay)
                    custom_config['call_config'] = call_config
                    custom_config['phone_number'] = lead.full_phone
                    logger.info(f"📝 Using custom prompt with voice: {custom_config['voice_name']} (will generate greeting per-call)")
            
            # Make the call using the voice agent with the agent's gate slot
            try:
                voice_session_id = sip_handler.make_outbound_call(
                    lead.full_phone, 
                    custom_config=custom_config if custom_config else None,
                    gate_slot=gate_slot,
                    owner_id=campaign.owner_id  # Pass owner_id for API key assignment
                )
                
                if voice_session_id:
                    # Update call session with voice agent session ID
                    call_session.session_id = voice_session_id
                    call_session.status = CallStatus.RINGING
                    call_session.save()  # MongoDB requires explicit save
                    
                    logger.info(f"✅ Call initiated: {voice_session_id}")
                    
                    # Wait for call to complete (sequential calling)
                    # Monitor call status from voice agent with shorter intervals for faster detection
                    max_wait_time = 300  # 5 minutes max per call
                    wait_time = 0
                    poll_interval = 2  # Check every 2 seconds for faster detection
                    
                    while wait_time < max_wait_time:
                        await asyncio.sleep(poll_interval)
                        wait_time += poll_interval
                        
                        # Refresh call session to check status (re-query from DB)
                        call_session = session.query(CallSession).filter(
                            CallSession.session_id == voice_session_id
                        ).first()
                        
                        if call_session and call_session.status in [CallStatus.COMPLETED, CallStatus.FAILED, CallStatus.REJECTED, CallStatus.NO_ANSWER]:
                            logger.info(f"📞 Call ended with status: {get_enum_value(call_session.status)}")
                            
                            # Update campaign lead status
                            campaign_manager.update_call_result(
                                campaign_id=campaign_id,
                                lead_id=lead.id,
                                session_id=voice_session_id,
                                status=call_session.status
                            )
                            
                            # Wait a short moment for cleanup to complete before next call
                            await asyncio.sleep(2)
                            break
                    
                    if wait_time >= max_wait_time:
                        logger.warning(f"⚠️ Call timeout reached for {lead.full_phone}")
                        call_session.status = CallStatus.FAILED
                        call_session.save()
                        campaign_manager.update_call_result(
                            campaign_id=campaign_id,
                            lead_id=lead.id,
                            session_id=voice_session_id,
                            status=CallStatus.FAILED
                        )
                else:
                    # Call failed to initiate
                    logger.error(f"❌ Failed to initiate call to {lead.full_phone}")
                    call_session.status = CallStatus.FAILED
                    campaign_lead.status = CallStatus.FAILED
                    call_session.save()  # MongoDB requires explicit save
                    campaign_lead.save()  # MongoDB requires explicit save
                    
            except Exception as e:
                logger.error(f"❌ Error making call to {lead.full_phone}: {e}")
                call_session.status = CallStatus.FAILED
                campaign_lead.status = CallStatus.FAILED
                call_session.save()  # MongoDB requires explicit save
                campaign_lead.save()  # MongoDB requires explicit save
            
            # Wait between calls (configurable in dialing_config)
            # Default is reduced since we already wait 2s after call completion for cleanup
            dialing_config = campaign.dialing_config or {}
            wait_between_calls = dialing_config.get('wait_between_calls', 3)  # Reduced from 10 to 3
            logger.info(f"⏳ Waiting {wait_between_calls}s before next call...")
            await asyncio.sleep(wait_between_calls)
            
    except Exception as e:
        logger.error(f"❌ Error executing campaign {campaign_id}: {e}", exc_info=True)
        campaign = session.query(Campaign).get(campaign_id)
        if campaign:
            campaign.status = CampaignStatus.CANCELLED
            campaign.save()  # MongoDB requires explicit save
    finally:
        session.close()


def build_prompt_from_config(call_config: dict, lead) -> str:
    """Build custom prompt from call config (similar to CustomCallDialog logic)"""
    company_name = call_config.get('company_name', '')
    caller_name = call_config.get('caller_name', '')
    product_name = call_config.get('product_name', '')
    call_objective = call_config.get('call_objective', 'sales')
    main_benefits = call_config.get('main_benefits', '')
    special_offer = call_config.get('special_offer', '')
    objection_strategy = call_config.get('objection_strategy', 'understanding')
    additional_prompt = call_config.get('additional_prompt', '')
    
    basePrompt = ""
    
    # Handle confirm_order differently - it's customer support, not sales
    if call_objective == "confirm_order":
        basePrompt = f"You are {caller_name} from {company_name}, a customer support representative. "
        basePrompt += f"You are calling to confirm a customer's existing order. The customer has ordered {product_name}. "
        basePrompt += "FOLLOW THIS FLOW EXACTLY: "
        basePrompt += f"1) Greet and introduce yourself: 'Hello, I'm calling from {company_name}.' "
        basePrompt += "2) Ask if you're speaking with the right person by their full name. If they say NO, apologize and say you're looking for [customer name], then end call politely. If YES, continue. "
        basePrompt += f"3) Inform about the order: 'You have an order for {product_name}.' Ask if they confirm this order. "
        basePrompt += "4) If they confirm, say 'Okay, we will make a delivery in 3 business days.' Then confirm ALL details: 'Just to confirm, you are [full name] with phone number [phone], and the delivery address is [address]?' "
        basePrompt += "5) Wait for their confirmation of details. If correct, thank them and end call. If incorrect, ask for correct information. "
        basePrompt += "Be professional, clear, courteous, and helpful throughout the call. "
        
        if special_offer:
            basePrompt += f"Additional information: {special_offer}. "
        
        basePrompt += "If they have questions or concerns, answer them patiently and professionally."
    
    # Special handling for AI Real Estate Services - Professional B2B sales
    elif call_objective == "ai_sales_services":
        basePrompt = f"You are {caller_name} from {company_name}, a professional B2B sales consultant specializing in AI automation solutions for real estate agencies. "
        basePrompt += f"Your product is {product_name}. "
        basePrompt += "\n\n🎯 CALL STRUCTURE - FOLLOW STRICTLY:\n"
        basePrompt += "1) After the greeting, introduce yourself briefly and state your purpose in ONE sentence\n"
        basePrompt += "2) Ask ONE qualifying question: 'Do you currently handle property inquiries manually?'\n"
        basePrompt += "3) WAIT for their response. Do NOT continue talking.\n"
        basePrompt += "4) Based on their answer, present ONE key benefit from these options:\n"
        basePrompt += f"   - {main_benefits}\n"
        basePrompt += "   - IMPORTANT: Specifically mention that the AI can automate COLD CALLING to potential buyers/sellers, qualifying leads automatically 24/7\n"
        basePrompt += "   - Emphasize: No more manual cold calling - the AI handles outbound prospecting calls\n"
        basePrompt += "5) Offer the demo: 'We can show you in a quick 15-minute demo how this works, including live cold calling automation.'\n"
        basePrompt += "6) If interested, suggest a specific time: 'Would tomorrow afternoon or Friday morning work better for you?'\n"
        basePrompt += "7) If they have concerns, address them one at a time.\n"
        basePrompt += "\n\n⚠️ CRITICAL RULES:\n"
        basePrompt += "- ASK ONLY ONE QUESTION AT A TIME\n"
        basePrompt += "- WAIT for their response before speaking again\n"
        basePrompt += "- Keep responses under 2-3 sentences\n"
        basePrompt += "- Be professional and consultative, NOT pushy\n"
        basePrompt += "- Listen actively to their needs\n"
        basePrompt += "- Always mention cold calling automation as a key differentiator\n"
        basePrompt += "- If they're not interested, thank them politely and end the call\n"
        
        if special_offer:
            basePrompt += f"\n\n💡 SPECIAL OFFER (mention only if they show interest): {special_offer}\n"
        
        if objection_strategy == "educational":
            basePrompt += "\nWhen handling objections: Provide educational information with facts. Share industry statistics or success stories. "
        elif objection_strategy == "understanding":
            basePrompt += "\nWhen handling objections: Show empathy and understanding. Acknowledge their concerns before addressing them. "
    
    # Special handling for companions services
    elif call_objective == "companions_services":
        basePrompt = f"You are {caller_name} from {company_name}, a professional client relations specialist. "
        basePrompt += f"Your service is {product_name}. "
        basePrompt += "\n\n🎯 CALL STRUCTURE - FOLLOW STRICTLY:\n"
        basePrompt += "1) After greeting, introduce yourself and verify you're speaking to the right person\n"
        basePrompt += "2) State your purpose professionally in ONE sentence\n"
        basePrompt += "3) Ask ONE question: 'Are you interested in learning about our premium services?'\n"
        basePrompt += "4) WAIT for response. Do NOT continue talking.\n"
        basePrompt += "5) If interested, briefly mention ONE key benefit, then offer to send information\n"
        basePrompt += "6) Suggest a meeting time if appropriate\n"
        basePrompt += "\n\n⚠️ CRITICAL RULES:\n"
        basePrompt += "- ASK ONLY ONE QUESTION AT A TIME\n"
        basePrompt += "- Be discreet and professional at all times\n"
        basePrompt += "- Keep responses brief (1-2 sentences)\n"
        basePrompt += "- WAIT for their response before continuing\n"
        basePrompt += "- Respect their privacy and discretion\n"
        
        if main_benefits:
            basePrompt += f"\n\nKey benefits to mention (one at a time): {main_benefits}\n"
        
        if special_offer:
            basePrompt += f"\n\nSpecial offer (mention only if interested): {special_offer}\n"
    
    else:
        # For standard sales-oriented calls
        basePrompt = f"You are {caller_name} from {company_name}, a professional sales representative for {product_name}. "
        
        if call_objective == "sales":
            basePrompt += "You are making sales calls to sell this product. Focus on converting prospects into customers by highlighting product benefits and closing the sale. "
        elif call_objective == "followup":
            basePrompt += "You are following up on a previous interaction. Be friendly and check on their interest while guiding toward a purchase decision. "
        elif call_objective == "survey":
            basePrompt += "You are conducting a survey but also identifying sales opportunities. Ask relevant questions while presenting the product benefits. "
        elif call_objective == "appointment":
            basePrompt += "You are cold calling to set appointments or qualify leads. Focus on building rapport, understanding their needs, and scheduling a follow-up meeting or call. "
        elif call_objective == "promotion_offer":
            basePrompt += "You are calling to present a special promotional offer. Be enthusiastic but not pushy. "
        
        basePrompt += "\n\n⚠️ IMPORTANT: ASK ONE QUESTION AT A TIME and WAIT for responses. Keep your statements brief and conversational.\n\n"
        
        if main_benefits:
            basePrompt += f"Key benefits to emphasize: {main_benefits}. "
        
        if special_offer:
            basePrompt += f"Current offers: {special_offer}. "
        
        if objection_strategy == "understanding":
            basePrompt += "Handle objections with empathy and understanding. Listen to their concerns and address them thoughtfully. "
        elif objection_strategy == "educational":
            basePrompt += "Handle objections by providing educational information and facts to overcome doubts. "
        elif objection_strategy == "aggressive":
            basePrompt += "Handle objections persistently. Push back on concerns and maintain strong sales pressure. "
        
        basePrompt += "Always try to close the sale and handle objections professionally."
    
    if additional_prompt:
        basePrompt += f"\n\nAdditional Instructions: {additional_prompt}"
    
    return basePrompt

# Export router
__all__ = ['crm_router']
