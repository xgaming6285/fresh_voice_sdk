# -*- coding: utf-8 -*-
"""
CRM Database Models for Voice Agent - MongoDB Version
Handles leads, campaigns, and call sessions using MongoDB
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from datetime import datetime
from passlib.context import CryptContext
import enum
import os
from typing import Optional, List, Dict, Any
from bson import ObjectId

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Enums
class UserRole(enum.Enum):
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    AGENT = "agent"

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

class PaymentRequestStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# MongoDB Connection
_client = None
_db = None

def get_database_url():
    """Get MongoDB connection URL from environment or use default"""
    return os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')

def get_database_name():
    """Get database name from environment or use default"""
    return os.getenv('MONGODB_DATABASE', 'voice_agent_crm')

def init_database():
    """Initialize MongoDB connection and create indexes"""
    global _client, _db
    
    if _client is None:
        mongo_url = get_database_url()
        db_name = get_database_name()
        
        _client = MongoClient(mongo_url)
        _db = _client[db_name]
        
        # Create indexes
        # Users collection
        _db.users.create_index([("username", ASCENDING)], unique=True)
        _db.users.create_index([("email", ASCENDING)], unique=True)
        _db.users.create_index([("created_by_id", ASCENDING)])
        _db.users.create_index([("role", ASCENDING)])
        
        # Leads collection
        _db.leads.create_index([("owner_id", ASCENDING)])
        _db.leads.create_index([("country", ASCENDING)])
        _db.leads.create_index([("phone", ASCENDING)])
        
        # Campaigns collection
        _db.campaigns.create_index([("owner_id", ASCENDING)])
        _db.campaigns.create_index([("status", ASCENDING)])
        
        # Campaign Leads collection
        _db.campaign_leads.create_index([("campaign_id", ASCENDING)])
        _db.campaign_leads.create_index([("lead_id", ASCENDING)])
        _db.campaign_leads.create_index([("status", ASCENDING)])
        _db.campaign_leads.create_index([("campaign_id", ASCENDING), ("lead_id", ASCENDING)], unique=True)
        
        # Call Sessions collection
        _db.call_sessions.create_index([("session_id", ASCENDING)], unique=True)
        _db.call_sessions.create_index([("owner_id", ASCENDING)])
        _db.call_sessions.create_index([("campaign_id", ASCENDING)])
        _db.call_sessions.create_index([("lead_id", ASCENDING)])
        _db.call_sessions.create_index([("started_at", DESCENDING)])
        
        # Payment Requests collection
        _db.payment_requests.create_index([("admin_id", ASCENDING)])
        _db.payment_requests.create_index([("status", ASCENDING)])
        
        # System Settings collection
        _db.system_settings.create_index([("key", ASCENDING)], unique=True)
        
        print(f"✅ MongoDB initialized: {db_name} at {mongo_url}")
    
    return _db

def get_session():
    """Get database instance (maintains compatibility with SQLAlchemy interface)"""
    if _db is None:
        init_database()
    return MongoDBSession(_db)


class MongoDBSession:
    """
    Wrapper class to provide SQLAlchemy-like interface for MongoDB
    This maintains compatibility with existing code
    """
    def __init__(self, db):
        self.db = db
        self._changes = []  # Track changes for commit/rollback
        
    def query(self, model):
        """Create a query object for the model"""
        return MongoQuery(self.db, model)
    
    def add(self, obj):
        """Add object to session (will be saved on commit)"""
        self._changes.append(('add', obj))
    
    def delete(self, obj):
        """Mark object for deletion"""
        self._changes.append(('delete', obj))
    
    def commit(self):
        """Commit all changes"""
        for action, obj in self._changes:
            if action == 'add':
                obj.save(self.db)
            elif action == 'delete':
                obj.delete(self.db)
        self._changes = []
    
    def rollback(self):
        """Rollback changes"""
        self._changes = []
    
    def close(self):
        """Close session (no-op for MongoDB but maintains compatibility)"""
        pass
    
    def refresh(self, obj):
        """Refresh object from database"""
        if hasattr(obj, 'id') and obj.id:
            collection_name = obj.__class__.__name__.lower() + 's'
            if collection_name.endswith('ys'):
                collection_name = collection_name[:-2] + 'ies'
            doc = self.db[collection_name].find_one({"_id": ObjectId(obj.id)})
            if doc:
                obj.__dict__.update(obj._from_dict(doc).__dict__)


class MongoQuery:
    """Query builder for MongoDB (SQLAlchemy-like interface)"""
    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.filters = {}
        self.sort_fields = []
        self.limit_val = None
        self.offset_val = 0
        self.options_val = []
        
        # Determine collection name
        collection_name = model.__name__.lower() + 's'
        if collection_name == 'campaignleads':
            collection_name = 'campaign_leads'
        elif collection_name == 'callsessions':
            collection_name = 'call_sessions'
        elif collection_name == 'paymentrequests':
            collection_name = 'payment_requests'
        elif collection_name == 'systemsettingss':
            collection_name = 'system_settings'
        
        self.collection = self.db[collection_name]
    
    def filter(self, *args, **kwargs):
        """Add filter conditions"""
        for condition in args:
            if hasattr(condition, 'compile'):
                # Handle SQLAlchemy-like conditions
                self.filters.update(condition.compile())
        
        # Handle simple key-value filters
        for key, value in kwargs.items():
            self.filters[key] = value
        
        return self
    
    def filter_by(self, **kwargs):
        """Filter by keyword arguments"""
        self.filters.update(kwargs)
        return self
    
    def order_by(self, *args):
        """Add ordering"""
        for field in args:
            if hasattr(field, 'compile_order'):
                self.sort_fields.append(field.compile_order())
            else:
                # Simple field name
                self.sort_fields.append((str(field), ASCENDING))
        return self
    
    def limit(self, val):
        """Limit results"""
        self.limit_val = val
        return self
    
    def offset(self, val):
        """Offset results"""
        self.offset_val = val
        return self
    
    def options(self, *args):
        """Query options (for compatibility, mostly ignored)"""
        self.options_val.extend(args)
        return self
    
    def count(self):
        """Count results"""
        return self.collection.count_documents(self.filters)
    
    def all(self):
        """Get all results"""
        cursor = self.collection.find(self.filters)
        
        if self.sort_fields:
            cursor = cursor.sort(self.sort_fields)
        
        if self.offset_val:
            cursor = cursor.skip(self.offset_val)
        
        if self.limit_val:
            cursor = cursor.limit(self.limit_val)
        
        return [self.model._from_dict(doc) for doc in cursor]
    
    def first(self):
        """Get first result"""
        doc = self.collection.find_one(self.filters)
        return self.model._from_dict(doc) if doc else None
    
    def get(self, id_val):
        """Get by ID"""
        try:
            if isinstance(id_val, str):
                id_val = ObjectId(id_val)
            elif isinstance(id_val, int):
                # For integer IDs, search by id field
                doc = self.collection.find_one({"id": id_val})
                return self.model._from_dict(doc) if doc else None
            
            doc = self.collection.find_one({"_id": id_val})
            return self.model._from_dict(doc) if doc else None
        except:
            return None


# Base Model Class
class MongoModel:
    """Base class for all MongoDB models"""
    def __init__(self, **kwargs):
        self._id = kwargs.get('_id')
        self.id = kwargs.get('id')
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, db):
        """Save to database"""
        collection_name = self.__class__.__name__.lower() + 's'
        if collection_name == 'campaignlead':
            collection_name = 'campaign_leads'
        elif collection_name == 'callsession':
            collection_name = 'call_sessions'
        elif collection_name == 'paymentrequest':
            collection_name = 'payment_requests'
        elif collection_name == 'systemsettings':
            collection_name = 'system_settings'
        
        collection = db[collection_name]
        doc = self._to_dict()
        
        if self._id:
            # Update existing
            collection.replace_one({"_id": self._id}, doc)
        else:
            # Insert new
            result = collection.insert_one(doc)
            self._id = result.inserted_id
            if not self.id:
                # Generate integer ID for compatibility
                self.id = int(str(result.inserted_id)[-8:], 16)
                collection.update_one({"_id": self._id}, {"$set": {"id": self.id}})
    
    def delete(self, db):
        """Delete from database"""
        if self._id:
            collection_name = self.__class__.__name__.lower() + 's'
            db[collection_name].delete_one({"_id": self._id})
    
    def _to_dict(self):
        """Convert to dictionary for MongoDB"""
        doc = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, enum.Enum):
                doc[key] = value.value
            elif isinstance(value, datetime):
                doc[key] = value
            else:
                doc[key] = value
        return doc
    
    @classmethod
    def _from_dict(cls, doc):
        """Create instance from MongoDB document"""
        if not doc:
            return None
        return cls(**doc)


# User Model
class User(MongoModel):
    """User model for authentication and authorization"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.username = kwargs.get('username')
        self.email = kwargs.get('email')
        self.hashed_password = kwargs.get('hashed_password')
        self.role = kwargs.get('role')
        if isinstance(self.role, str):
            self.role = UserRole(self.role)
        
        self.created_by_id = kwargs.get('created_by_id')
        self.organization = kwargs.get('organization')
        self.max_agents = kwargs.get('max_agents', 0)
        self.subscription_end_date = kwargs.get('subscription_end_date')
        
        self.first_name = kwargs.get('first_name')
        self.last_name = kwargs.get('last_name')
        self.is_active = kwargs.get('is_active', True)
        
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.last_login = kwargs.get('last_login')
        
        super().__init__(**kwargs)
    
    @property
    def full_name(self):
        """Get full name"""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else self.username
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def is_subscription_active(self) -> bool:
        """Check if user's subscription is active (for admins and their agents)"""
        if self.role == UserRole.SUPERADMIN:
            return True  # Superadmin always active
        
        if self.role == UserRole.AGENT:
            # For agents, check their admin's subscription
            return True  # Will be checked via their admin in the endpoint
        
        if self.role == UserRole.ADMIN:
            # Admin must have subscription_end_date and it must be in the future
            if not self.subscription_end_date:
                return False
            return self.subscription_end_date > datetime.utcnow()
        
        return False


# Lead Model
class Lead(MongoModel):
    """Lead/Contact model"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.owner_id = kwargs.get('owner_id')
        self.first_name = kwargs.get('first_name')
        self.last_name = kwargs.get('last_name')
        self.email = kwargs.get('email')
        self.phone = kwargs.get('phone')
        self.country = kwargs.get('country')
        self.country_code = kwargs.get('country_code')
        self.gender = kwargs.get('gender', Gender.UNKNOWN)
        if isinstance(self.gender, str):
            self.gender = Gender(self.gender)
        
        self.address = kwargs.get('address')
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.last_called_at = kwargs.get('last_called_at')
        self.call_count = kwargs.get('call_count', 0)
        self.notes = kwargs.get('notes')
        self.custom_data = kwargs.get('custom_data')
        self.import_batch_id = kwargs.get('import_batch_id')
        
        super().__init__(**kwargs)
    
    @property
    def full_phone(self):
        """Get full phone number with country code"""
        return f"{self.country_code}{self.phone}" if self.country_code else self.phone
    
    @property
    def full_name(self):
        """Get full name"""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else "Unknown"


# Campaign Model
class Campaign(MongoModel):
    """Campaign model for organizing and executing call campaigns"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.owner_id = kwargs.get('owner_id')
        self.name = kwargs.get('name')
        self.description = kwargs.get('description')
        self.status = kwargs.get('status', CampaignStatus.DRAFT)
        if isinstance(self.status, str):
            self.status = CampaignStatus(self.status)
        
        self.bot_config = kwargs.get('bot_config', {})
        self.dialing_config = kwargs.get('dialing_config', {})
        self.schedule_config = kwargs.get('schedule_config', {})
        
        self.total_leads = kwargs.get('total_leads', 0)
        self.leads_called = kwargs.get('leads_called', 0)
        self.leads_answered = kwargs.get('leads_answered', 0)
        self.leads_rejected = kwargs.get('leads_rejected', 0)
        self.leads_failed = kwargs.get('leads_failed', 0)
        
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.started_at = kwargs.get('started_at')
        self.completed_at = kwargs.get('completed_at')
        
        super().__init__(**kwargs)


# CampaignLead Model
class CampaignLead(MongoModel):
    """Many-to-many relationship between campaigns and leads with additional data"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.campaign_id = kwargs.get('campaign_id')
        self.lead_id = kwargs.get('lead_id')
        
        self.status = kwargs.get('status', CallStatus.PENDING)
        if isinstance(self.status, str):
            self.status = CallStatus(self.status)
        
        self.priority = kwargs.get('priority', 0)
        self.call_attempts = kwargs.get('call_attempts', 0)
        self.last_attempt_at = kwargs.get('last_attempt_at')
        self.scheduled_for = kwargs.get('scheduled_for')
        
        self.call_session_id = kwargs.get('call_session_id')
        self.call_duration = kwargs.get('call_duration')
        self.call_result = kwargs.get('call_result')
        self.call_notes = kwargs.get('call_notes')
        
        self.added_at = kwargs.get('added_at', datetime.utcnow())
        self.called_at = kwargs.get('called_at')
        self.completed_at = kwargs.get('completed_at')
        
        # Lazy-loaded relationships
        self._lead = None
        self._campaign = None
        
        super().__init__(**kwargs)
    
    @property
    def lead(self):
        """Get associated lead"""
        if self._lead is None and self.lead_id:
            db = init_database()
            doc = db.leads.find_one({"id": self.lead_id})
            if doc:
                self._lead = Lead._from_dict(doc)
        return self._lead
    
    @property
    def campaign(self):
        """Get associated campaign"""
        if self._campaign is None and self.campaign_id:
            db = init_database()
            doc = db.campaigns.find_one({"id": self.campaign_id})
            if doc:
                self._campaign = Campaign._from_dict(doc)
        return self._campaign


# CallSession Model
class CallSession(MongoModel):
    """Track individual call sessions with links to voice agent sessions"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.session_id = kwargs.get('session_id')
        self.campaign_id = kwargs.get('campaign_id')
        self.lead_id = kwargs.get('lead_id')
        self.owner_id = kwargs.get('owner_id')
        
        self.caller_id = kwargs.get('caller_id')
        self.called_number = kwargs.get('called_number')
        self.status = kwargs.get('status', CallStatus.PENDING)
        if isinstance(self.status, str):
            self.status = CallStatus(self.status)
        
        self.started_at = kwargs.get('started_at', datetime.utcnow())
        self.answered_at = kwargs.get('answered_at')
        self.ended_at = kwargs.get('ended_at')
        self.duration = kwargs.get('duration')
        self.talk_time = kwargs.get('talk_time')
        
        self.recording_path = kwargs.get('recording_path')
        self.transcript_status = kwargs.get('transcript_status')
        self.transcript_language = kwargs.get('transcript_language')
        
        self.sentiment_score = kwargs.get('sentiment_score')
        self.interest_level = kwargs.get('interest_level')
        self.key_points = kwargs.get('key_points')
        self.follow_up_required = kwargs.get('follow_up_required', False)
        self.follow_up_notes = kwargs.get('follow_up_notes')
        
        self.call_metadata = kwargs.get('call_metadata')
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        
        # Lazy-loaded relationships
        self._lead = None
        self._campaign = None
        
        super().__init__(**kwargs)
    
    @property
    def lead(self):
        """Get associated lead"""
        if self._lead is None and self.lead_id:
            db = init_database()
            doc = db.leads.find_one({"id": self.lead_id})
            if doc:
                self._lead = Lead._from_dict(doc)
        return self._lead
    
    @property
    def campaign(self):
        """Get associated campaign"""
        if self._campaign is None and self.campaign_id:
            db = init_database()
            doc = db.campaigns.find_one({"id": self.campaign_id})
            if doc:
                self._campaign = Campaign._from_dict(doc)
        return self._campaign


# PaymentRequest Model
class PaymentRequest(MongoModel):
    """Payment request model for admin agent slot purchases"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.admin_id = kwargs.get('admin_id')
        self.num_agents = kwargs.get('num_agents')
        self.total_amount = kwargs.get('total_amount')
        self.status = kwargs.get('status', PaymentRequestStatus.PENDING)
        if isinstance(self.status, str):
            self.status = PaymentRequestStatus(self.status)
        
        self.payment_notes = kwargs.get('payment_notes')
        self.admin_notes = kwargs.get('admin_notes')
        
        self.created_at = kwargs.get('created_at', datetime.utcnow())
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.approved_at = kwargs.get('approved_at')
        self.approved_by_id = kwargs.get('approved_by_id')
        
        super().__init__(**kwargs)


# SystemSettings Model
class SystemSettings(MongoModel):
    """System-wide settings"""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.key = kwargs.get('key')
        self.value = kwargs.get('value')
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.updated_by_id = kwargs.get('updated_by_id')
        
        super().__init__(**kwargs)


# Manager Classes
class LeadManager:
    """Manager class for lead operations"""
    
    def __init__(self, session):
        self.session = session
        self.db = session.db
    
    def create_lead(self, lead_data):
        """Create a new lead"""
        lead = Lead(**lead_data)
        lead.save(self.db)
        return lead
    
    def bulk_import_leads(self, leads_data, import_batch_id=None):
        """Import multiple leads at once"""
        if import_batch_id is None:
            import_batch_id = f"import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        leads = []
        for lead_data in leads_data:
            lead_data['import_batch_id'] = import_batch_id
            lead = Lead(**lead_data)
            leads.append(lead._to_dict())
        
        if leads:
            result = self.db.leads.insert_many(leads)
            return len(result.inserted_ids)
        return 0
    
    def get_leads_by_criteria(self, owner_id=None, country=None, limit=None, offset=0):
        """Get leads by various criteria"""
        filters = {}
        if owner_id:
            filters['owner_id'] = owner_id
        if country:
            filters['country'] = country
        
        total = self.db.leads.count_documents(filters)
        
        cursor = self.db.leads.find(filters).skip(offset)
        if limit:
            cursor = cursor.limit(limit)
        
        leads = [Lead._from_dict(doc) for doc in cursor]
        
        return leads, total


class CampaignManager:
    """Manager class for campaign operations"""
    
    def __init__(self, session):
        self.session = session
        self.db = session.db
    
    def create_campaign(self, owner_id, name, description=None, bot_config=None):
        """Create a new campaign"""
        campaign = Campaign(
            owner_id=owner_id,
            name=name,
            description=description,
            bot_config=bot_config or {},
            dialing_config={},
            schedule_config={}
        )
        campaign.save(self.db)
        return campaign
    
    def add_leads_to_campaign(self, campaign_id, lead_ids, priority=0):
        """Add leads to a campaign"""
        campaign_leads = []
        for lead_id in lead_ids:
            cl_dict = {
                'campaign_id': campaign_id,
                'lead_id': lead_id,
                'priority': priority,
                'status': CallStatus.PENDING.value,
                'call_attempts': 0,
                'added_at': datetime.utcnow()
            }
            campaign_leads.append(cl_dict)
        
        if campaign_leads:
            try:
                self.db.campaign_leads.insert_many(campaign_leads, ordered=False)
            except Exception as e:
                # Handle duplicate key errors
                pass
        
        # Update campaign total leads
        total_leads = self.db.campaign_leads.count_documents({'campaign_id': campaign_id})
        self.db.campaigns.update_one(
            {'id': campaign_id},
            {'$set': {'total_leads': total_leads}}
        )
        
        return len(campaign_leads)
    
    def get_next_lead_to_call(self, campaign_id):
        """Get the next lead to call in a campaign"""
        doc = self.db.campaign_leads.find_one_and_update(
            {
                'campaign_id': campaign_id,
                'status': CallStatus.PENDING.value
            },
            {
                '$set': {
                    'status': CallStatus.DIALING.value,
                    'last_attempt_at': datetime.utcnow()
                },
                '$inc': {'call_attempts': 1}
            },
            sort=[('priority', DESCENDING), ('added_at', ASCENDING)]
        )
        
        return CampaignLead._from_dict(doc) if doc else None
    
    def update_call_result(self, campaign_id, lead_id, session_id, status, duration=None):
        """Update the result of a call"""
        # Update campaign lead
        update_data = {
            'status': status.value if isinstance(status, enum.Enum) else status,
            'call_session_id': session_id
        }
        
        if duration:
            update_data['call_duration'] = duration
        
        if status in [CallStatus.ANSWERED, CallStatus.REJECTED, CallStatus.NO_ANSWER]:
            update_data['completed_at'] = datetime.utcnow()
        
        self.db.campaign_leads.update_one(
            {'campaign_id': campaign_id, 'lead_id': lead_id},
            {'$set': update_data}
        )
        
        # Update lead's last called info
        self.db.leads.update_one(
            {'id': lead_id},
            {
                '$set': {'last_called_at': datetime.utcnow()},
                '$inc': {'call_count': 1}
            }
        )
        
        # Update campaign statistics
        campaign_update = {'$inc': {'leads_called': 1}}
        
        if status == CallStatus.ANSWERED:
            campaign_update['$inc']['leads_answered'] = 1
        elif status == CallStatus.REJECTED:
            campaign_update['$inc']['leads_rejected'] = 1
        elif status in [CallStatus.FAILED, CallStatus.NO_ANSWER]:
            campaign_update['$inc']['leads_failed'] = 1
        
        self.db.campaigns.update_one(
            {'id': campaign_id},
            campaign_update
        )


class UserManager:
    """Manager class for user operations"""
    
    def __init__(self, session):
        self.session = session
        self.db = session.db
    
    def create_user(self, username, email, password, role, created_by_id=None, first_name=None, last_name=None):
        """Create a new user"""
        user = User(
            username=username,
            email=email,
            hashed_password=User.hash_password(password),
            role=role,
            created_by_id=created_by_id,
            first_name=first_name,
            last_name=last_name
        )
        user.save(self.db)
        return user
    
    def get_user_by_username(self, username):
        """Get user by username"""
        doc = self.db.users.find_one({'username': username})
        return User._from_dict(doc) if doc else None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        doc = self.db.users.find_one({'email': email})
        return User._from_dict(doc) if doc else None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        doc = self.db.users.find_one({'id': user_id})
        return User._from_dict(doc) if doc else None
    
    def get_agents_by_admin(self, admin_id):
        """Get all agents created by an admin"""
        cursor = self.db.users.find({
            'created_by_id': admin_id,
            'role': UserRole.AGENT.value
        })
        return [User._from_dict(doc) for doc in cursor]
    
    def update_user(self, user_id, **kwargs):
        """Update user fields"""
        doc = self.db.users.find_one({'id': user_id})
        if doc:
            user = User._from_dict(doc)
            for key, value in kwargs.items():
                if key == 'password':
                    user.hashed_password = User.hash_password(value)
                elif hasattr(user, key):
                    setattr(user, key, value)
            user.updated_at = datetime.utcnow()
            user.save(self.db)
            return user
        return None
    
    def delete_user(self, user_id):
        """Delete a user"""
        # Delete user's leads
        self.db.leads.delete_many({'owner_id': user_id})
        
        # Delete user's campaigns
        campaign_ids = [doc['id'] for doc in self.db.campaigns.find({'owner_id': user_id}, {'id': 1})]
        if campaign_ids:
            self.db.campaign_leads.delete_many({'campaign_id': {'$in': campaign_ids}})
            self.db.campaigns.delete_many({'owner_id': user_id})
        
        # Delete user's call sessions
        self.db.call_sessions.delete_many({'owner_id': user_id})
        
        # Delete the user
        result = self.db.users.delete_one({'id': user_id})
        return result.deleted_count > 0


# Helper function for joinedload compatibility
def joinedload(relationship):
    """Dummy function for SQLAlchemy joinedload compatibility"""
    return relationship


if __name__ == "__main__":
    # Initialize database
    db = init_database()
    print("✅ MongoDB database initialized successfully!")
    print(f"   Collections: {db.list_collection_names()}")

