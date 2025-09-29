# -*- coding: utf-8 -*-
"""
CRM Database Models for Voice Agent
Handles leads, campaigns, and call sessions
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum
import os

Base = declarative_base()

# Enums
class LeadType(enum.Enum):
    FTD = "ftd"  # First Time Deposit
    COLD = "cold"
    FILLER = "filler"
    LIVE = "live"

class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"

class CampaignStatus(enum.Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class CallStatus(enum.Enum):
    PENDING = "pending"
    DIALING = "dialing"
    RINGING = "ringing"
    IN_CALL = "in_call"
    ANSWERED = "answered"
    REJECTED = "rejected"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    FAILED = "failed"
    COMPLETED = "completed"

class Lead(Base):
    """Lead/Contact model"""
    __tablename__ = 'leads'
    
    id = Column(Integer, primary_key=True)
    lead_type = Column(Enum(LeadType), nullable=False, default=LeadType.COLD)
    first_name = Column(String(100))
    last_name = Column(String(100))
    email = Column(String(200))
    phone = Column(String(50), nullable=False)  # Phone without country code
    country = Column(String(100))  # Country name
    country_code = Column(String(10))  # Country calling code (e.g., +359)
    gender = Column(Enum(Gender), default=Gender.UNKNOWN)
    address = Column(Text)
    
    # Additional fields for tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_called_at = Column(DateTime)
    call_count = Column(Integer, default=0)
    notes = Column(Text)
    custom_data = Column(JSON)  # For storing additional flexible data
    
    # Import tracking
    import_batch_id = Column(String(100))  # Track which import batch this lead came from
    
    # Relationships
    campaign_leads = relationship("CampaignLead", back_populates="lead", cascade="all, delete-orphan")
    call_sessions = relationship("CallSession", back_populates="lead", cascade="all, delete-orphan")
    
    @property
    def full_phone(self):
        """Get full phone number with country code"""
        return f"{self.country_code}{self.phone}" if self.country_code else self.phone
    
    @property
    def full_name(self):
        """Get full name"""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else "Unknown"

class Campaign(Base):
    """Campaign model for organizing and executing call campaigns"""
    __tablename__ = 'campaigns'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(Enum(CampaignStatus), default=CampaignStatus.DRAFT)
    
    # Campaign configuration
    bot_config = Column(JSON)  # Store bot configuration (voice settings, script, etc.)
    dialing_config = Column(JSON)  # Dialing settings (concurrent calls, retry attempts, etc.)
    schedule_config = Column(JSON)  # When to run (time windows, days of week, etc.)
    
    # Statistics
    total_leads = Column(Integer, default=0)
    leads_called = Column(Integer, default=0)
    leads_answered = Column(Integer, default=0)
    leads_rejected = Column(Integer, default=0)
    leads_failed = Column(Integer, default=0)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    campaign_leads = relationship("CampaignLead", back_populates="campaign", cascade="all, delete-orphan")
    call_sessions = relationship("CallSession", back_populates="campaign", cascade="all, delete-orphan")

class CampaignLead(Base):
    """Many-to-many relationship between campaigns and leads with additional data"""
    __tablename__ = 'campaign_leads'
    
    id = Column(Integer, primary_key=True)
    campaign_id = Column(Integer, ForeignKey('campaigns.id'), nullable=False)
    lead_id = Column(Integer, ForeignKey('leads.id'), nullable=False)
    
    # Status for this lead in this campaign
    status = Column(Enum(CallStatus), default=CallStatus.PENDING)
    priority = Column(Integer, default=0)  # Higher priority = called first
    
    # Call tracking
    call_attempts = Column(Integer, default=0)
    last_attempt_at = Column(DateTime)
    scheduled_for = Column(DateTime)  # When to call this lead
    
    # Results
    call_session_id = Column(String(100))  # Link to voice agent session
    call_duration = Column(Integer)  # In seconds
    call_result = Column(String(100))  # Quick result summary
    call_notes = Column(Text)  # AI-generated or manual notes
    
    # Timestamps
    added_at = Column(DateTime, default=datetime.utcnow)
    called_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="campaign_leads")
    lead = relationship("Lead", back_populates="campaign_leads")

class CallSession(Base):
    """Track individual call sessions with links to voice agent sessions"""
    __tablename__ = 'call_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)  # Voice agent session ID
    campaign_id = Column(Integer, ForeignKey('campaigns.id'))
    lead_id = Column(Integer, ForeignKey('leads.id'))
    
    # Call details
    caller_id = Column(String(50))  # Our number
    called_number = Column(String(50))  # Lead's number
    status = Column(Enum(CallStatus), default=CallStatus.PENDING)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    answered_at = Column(DateTime)
    ended_at = Column(DateTime)
    duration = Column(Integer)  # Total duration in seconds
    talk_time = Column(Integer)  # Actual conversation time in seconds
    
    # Recording and transcription
    recording_path = Column(String(500))  # Path to recording files
    transcript_status = Column(String(50))  # pending, processing, completed, failed
    transcript_language = Column(String(10))
    
    # AI Analysis (can be populated after call)
    sentiment_score = Column(Float)  # -1 to 1
    interest_level = Column(Integer)  # 1-10
    key_points = Column(JSON)  # List of key points from conversation
    follow_up_required = Column(Boolean, default=False)
    follow_up_notes = Column(Text)
    
    # Raw data storage
    call_metadata = Column(JSON)  # Store any additional call data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    campaign = relationship("Campaign", back_populates="call_sessions")
    lead = relationship("Lead", back_populates="call_sessions")

# Database setup functions
def get_database_url():
    """Get database URL from environment or use default SQLite"""
    return os.getenv('DATABASE_URL', 'sqlite:///voice_agent_crm.db')

def init_database():
    """Initialize database and create tables"""
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Get database session"""
    engine = init_database()
    Session = sessionmaker(bind=engine)
    return Session()

# Utility functions for common queries
class LeadManager:
    """Manager class for lead operations"""
    
    def __init__(self, session):
        self.session = session
    
    def create_lead(self, lead_data):
        """Create a new lead"""
        lead = Lead(**lead_data)
        self.session.add(lead)
        self.session.commit()
        return lead
    
    def bulk_import_leads(self, leads_data, import_batch_id=None):
        """Import multiple leads at once"""
        if import_batch_id is None:
            import_batch_id = f"import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        leads = []
        for lead_data in leads_data:
            lead_data['import_batch_id'] = import_batch_id
            lead = Lead(**lead_data)
            leads.append(lead)
        
        self.session.bulk_save_objects(leads)
        self.session.commit()
        return len(leads)
    
    def get_leads_by_criteria(self, country=None, lead_type=None, limit=None, offset=0):
        """Get leads by various criteria"""
        query = self.session.query(Lead)
        
        if country:
            query = query.filter(Lead.country == country)
        if lead_type:
            query = query.filter(Lead.lead_type == lead_type)
        
        total = query.count()
        
        if limit:
            query = query.limit(limit).offset(offset)
        
        return query.all(), total

class CampaignManager:
    """Manager class for campaign operations"""
    
    def __init__(self, session):
        self.session = session
    
    def create_campaign(self, name, description=None, bot_config=None):
        """Create a new campaign"""
        campaign = Campaign(
            name=name,
            description=description,
            bot_config=bot_config or {},
            dialing_config={},
            schedule_config={}
        )
        self.session.add(campaign)
        self.session.commit()
        return campaign
    
    def add_leads_to_campaign(self, campaign_id, lead_ids, priority=0):
        """Add leads to a campaign"""
        campaign_leads = []
        for lead_id in lead_ids:
            cl = CampaignLead(
                campaign_id=campaign_id,
                lead_id=lead_id,
                priority=priority
            )
            campaign_leads.append(cl)
        
        self.session.bulk_save_objects(campaign_leads)
        
        # Update campaign total leads
        campaign = self.session.query(Campaign).get(campaign_id)
        campaign.total_leads = len(campaign_leads)
        
        self.session.commit()
        return len(campaign_leads)
    
    def get_next_lead_to_call(self, campaign_id):
        """Get the next lead to call in a campaign"""
        campaign_lead = self.session.query(CampaignLead).filter(
            CampaignLead.campaign_id == campaign_id,
            CampaignLead.status == CallStatus.PENDING
        ).order_by(
            CampaignLead.priority.desc(),
            CampaignLead.added_at
        ).first()
        
        if campaign_lead:
            # Mark as dialing
            campaign_lead.status = CallStatus.DIALING
            campaign_lead.call_attempts += 1
            campaign_lead.last_attempt_at = datetime.utcnow()
            self.session.commit()
        
        return campaign_lead
    
    def update_call_result(self, campaign_id, lead_id, session_id, status, duration=None):
        """Update the result of a call"""
        campaign_lead = self.session.query(CampaignLead).filter(
            CampaignLead.campaign_id == campaign_id,
            CampaignLead.lead_id == lead_id
        ).first()
        
        if campaign_lead:
            campaign_lead.status = status
            campaign_lead.call_session_id = session_id
            if duration:
                campaign_lead.call_duration = duration
            if status in [CallStatus.ANSWERED, CallStatus.REJECTED, CallStatus.NO_ANSWER]:
                campaign_lead.completed_at = datetime.utcnow()
            
            # Update lead's last called info
            lead = self.session.query(Lead).get(lead_id)
            lead.last_called_at = datetime.utcnow()
            lead.call_count += 1
            
            # Update campaign statistics
            campaign = self.session.query(Campaign).get(campaign_id)
            campaign.leads_called += 1
            if status == CallStatus.ANSWERED:
                campaign.leads_answered += 1
            elif status == CallStatus.REJECTED:
                campaign.leads_rejected += 1
            elif status in [CallStatus.FAILED, CallStatus.NO_ANSWER]:
                campaign.leads_failed += 1
            
            self.session.commit()

if __name__ == "__main__":
    # Initialize database
    engine = init_database()
    print("Database initialized successfully!")
