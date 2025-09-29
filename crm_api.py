# -*- coding: utf-8 -*-
"""
CRM API Endpoints for Voice Agent
Handles leads, campaigns, and call orchestration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, File, UploadFile
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
    get_session, Lead, Campaign, CampaignLead, CallSession,
    LeadType, Gender, CampaignStatus, CallStatus,
    LeadManager, CampaignManager
)

logger = logging.getLogger(__name__)

# Create API router
crm_router = APIRouter(prefix="/api/crm", tags=["CRM"])

# Pydantic models for API
class LeadCreate(BaseModel):
    lead_type: str = Field(default="cold")
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
    lead_type: Optional[str] = None
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
    lead_type: str
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
    lead_types: Optional[List[str]] = None
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

# Lead endpoints
@crm_router.post("/leads", response_model=LeadResponse)
async def create_lead(lead: LeadCreate):
    """Create a new lead"""
    try:
        session = get_session()
        lead_manager = LeadManager(session)
        
        # Convert string enums to proper enum values
        lead_data = lead.dict()
        lead_data['lead_type'] = LeadType(lead_data['lead_type'])
        lead_data['gender'] = Gender(lead_data.get('gender', 'unknown'))
        
        db_lead = lead_manager.create_lead(lead_data)
        
        return LeadResponse(
            id=db_lead.id,
            lead_type=db_lead.lead_type.value,
            first_name=db_lead.first_name,
            last_name=db_lead.last_name,
            email=db_lead.email,
            phone=db_lead.phone,
            country=db_lead.country,
            country_code=db_lead.country_code,
            gender=db_lead.gender.value,
            address=db_lead.address,
            created_at=db_lead.created_at,
            updated_at=db_lead.updated_at,
            last_called_at=db_lead.last_called_at,
            call_count=db_lead.call_count,
            notes=db_lead.notes,
            full_phone=db_lead.full_phone,
            full_name=db_lead.full_name
        )
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
    lead_type: Optional[str] = None
):
    """Get paginated list of leads"""
    try:
        session = get_session()
        lead_manager = LeadManager(session)
        
        offset = (page - 1) * per_page
        
        # Convert lead_type string to enum if provided
        lead_type_enum = LeadType(lead_type) if lead_type else None
        
        leads, total = lead_manager.get_leads_by_criteria(
            country=country,
            lead_type=lead_type_enum,
            limit=per_page,
            offset=offset
        )
        
        leads_response = []
        for lead in leads:
            leads_response.append(LeadResponse(
                id=lead.id,
                lead_type=lead.lead_type.value,
                first_name=lead.first_name,
                last_name=lead.last_name,
                email=lead.email,
                phone=lead.phone,
                country=lead.country,
                country_code=lead.country_code,
                gender=lead.gender.value,
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
async def get_lead(lead_id: int):
    """Get a specific lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        return LeadResponse(
            id=lead.id,
            lead_type=lead.lead_type.value,
            first_name=lead.first_name,
            last_name=lead.last_name,
            email=lead.email,
            phone=lead.phone,
            country=lead.country,
            country_code=lead.country_code,
            gender=lead.gender.value,
            address=lead.address,
            created_at=lead.created_at,
            updated_at=lead.updated_at,
            last_called_at=lead.last_called_at,
            call_count=lead.call_count,
            notes=lead.notes,
            full_phone=lead.full_phone,
            full_name=lead.full_name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.put("/leads/{lead_id}", response_model=LeadResponse)
async def update_lead(lead_id: int, lead_update: LeadUpdate):
    """Update a lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Update fields
        update_data = lead_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                if field == 'lead_type':
                    value = LeadType(value)
                elif field == 'gender':
                    value = Gender(value)
                setattr(lead, field, value)
        
        session.commit()
        
        return LeadResponse(
            id=lead.id,
            lead_type=lead.lead_type.value,
            first_name=lead.first_name,
            last_name=lead.last_name,
            email=lead.email,
            phone=lead.phone,
            country=lead.country,
            country_code=lead.country_code,
            gender=lead.gender.value,
            address=lead.address,
            created_at=lead.created_at,
            updated_at=lead.updated_at,
            last_called_at=lead.last_called_at,
            call_count=lead.call_count,
            notes=lead.notes,
            full_phone=lead.full_phone,
            full_name=lead.full_name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: int):
    """Delete a lead"""
    try:
        session = get_session()
        lead = session.query(Lead).get(lead_id)
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
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
async def import_leads(file: UploadFile = File(...)):
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
                'lead_type': LeadType(row.get('lead_type', 'cold').lower()),
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
async def create_campaign(campaign: CampaignCreate):
    """Create a new campaign"""
    try:
        session = get_session()
        campaign_manager = CampaignManager(session)
        
        db_campaign = campaign_manager.create_campaign(
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
        
        return CampaignResponse(
            id=db_campaign.id,
            name=db_campaign.name,
            description=db_campaign.description,
            status=db_campaign.status.value,
            bot_config=db_campaign.bot_config or {},
            dialing_config=db_campaign.dialing_config or {},
            schedule_config=db_campaign.schedule_config or {},
            total_leads=db_campaign.total_leads,
            leads_called=db_campaign.leads_called,
            leads_answered=db_campaign.leads_answered,
            leads_rejected=db_campaign.leads_rejected,
            leads_failed=db_campaign.leads_failed,
            created_at=db_campaign.created_at,
            updated_at=db_campaign.updated_at,
            started_at=db_campaign.started_at,
            completed_at=db_campaign.completed_at
        )
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns", response_model=List[CampaignResponse])
async def get_campaigns(status: Optional[str] = None):
    """Get list of campaigns"""
    try:
        session = get_session()
        
        query = session.query(Campaign)
        if status:
            query = query.filter(Campaign.status == CampaignStatus(status))
        
        campaigns = query.order_by(Campaign.created_at.desc()).all()
        
        return [
            CampaignResponse(
                id=c.id,
                name=c.name,
                description=c.description,
                status=c.status.value,
                bot_config=c.bot_config or {},
                dialing_config=c.dialing_config or {},
                schedule_config=c.schedule_config or {},
                total_leads=c.total_leads,
                leads_called=c.leads_called,
                leads_answered=c.leads_answered,
                leads_rejected=c.leads_rejected,
                leads_failed=c.leads_failed,
                created_at=c.created_at,
                updated_at=c.updated_at,
                started_at=c.started_at,
                completed_at=c.completed_at
            ) for c in campaigns
        ]
    except Exception as e:
        logger.error(f"Error getting campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: int):
    """Get a specific campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return CampaignResponse(
            id=campaign.id,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.put("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(campaign_id: int, campaign_update: CampaignUpdate):
    """Update a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        # Update fields
        update_data = campaign_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                if field == 'status':
                    value = CampaignStatus(value)
                setattr(campaign, field, value)
        
        session.commit()
        
        return CampaignResponse(
            id=campaign.id,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@crm_router.post("/campaigns/{campaign_id}/leads")
async def add_leads_to_campaign(campaign_id: int, request: AddLeadsToCampaign):
    """Add leads to a campaign"""
    try:
        session = get_session()
        campaign_manager = CampaignManager(session)
        
        # Verify campaign exists
        campaign = session.query(Campaign).get(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
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
async def add_filtered_leads_to_campaign(campaign_id: int, filter: LeadFilter):
    """Add leads to campaign based on filter criteria"""
    try:
        session = get_session()
        
        # Build query
        query = session.query(Lead)
        
        if filter.countries:
            query = query.filter(Lead.country.in_(filter.countries))
        if filter.lead_types:
            lead_type_enums = [LeadType(lt) for lt in filter.lead_types]
            query = query.filter(Lead.lead_type.in_(lead_type_enums))
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
    status: Optional[str] = None
):
    """Get leads in a campaign"""
    try:
        session = get_session()
        
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
async def start_campaign(campaign_id: int, background_tasks: BackgroundTasks):
    """Start a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
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
async def pause_campaign(campaign_id: int):
    """Pause a running campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
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
async def stop_campaign(campaign_id: int):
    """Stop a campaign"""
    try:
        session = get_session()
        campaign = session.query(Campaign).get(campaign_id)
        
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
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
    limit: int = Query(100, ge=1, le=1000)
):
    """Get call sessions"""
    try:
        session = get_session()
        
        query = session.query(CallSession)
        
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
async def get_call_session(session_id: str):
    """Get a specific call session by voice agent session ID"""
    try:
        session = get_session()
        call_session = session.query(CallSession).filter(
            CallSession.session_id == session_id
        ).first()
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
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
