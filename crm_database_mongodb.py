# -*- coding: utf-8 -*-
"""
CRM Database Layer for Voice Agent - MongoDB Implementation
Handles leads, campaigns, and call sessions using MongoDB
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
from passlib.context import CryptContext
import enum
import os
from typing import List, Optional, Dict, Any
from bson import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Password hashing - with proper configuration for bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__min_rounds=4,
    bcrypt__max_rounds=12,
    bcrypt__default_rounds=12
)

# Helper function for safe enum value extraction
def get_enum_value(enum_or_str):
    """
    Safely get the value from an enum or return the string if it's already a string.
    This handles cases where MongoDB might return strings instead of enums.
    """
    if enum_or_str is None:
        return None
    return enum_or_str.value if hasattr(enum_or_str, 'value') else enum_or_str

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
class MongoDB:
    _client = None
    _db = None
    
    @classmethod
    def get_client(cls):
        """Get MongoDB client"""
        if cls._client is None:
            # Check for MongoDB connection string (Atlas or full URI)
            mongo_uri = os.getenv('MONGO_DB') or os.getenv('MONGODB_URI')
            
            if mongo_uri:
                # Use connection string (MongoDB Atlas or full URI)
                cls._client = MongoClient(mongo_uri)
            else:
                # Fallback to host:port format for local MongoDB
                mongo_host = os.getenv('MONGODB_HOST', 'localhost')
                mongo_port = int(os.getenv('MONGODB_PORT', '27017'))
                cls._client = MongoClient(mongo_host, mongo_port)
        return cls._client
    
    @classmethod
    def get_db(cls):
        """Get MongoDB database"""
        if cls._db is None:
            client = cls.get_client()
            db_name = os.getenv('MONGODB_DATABASE', 'voice_agent_crm')
            cls._db = client[db_name]
        return cls._db

def init_database():
    """Initialize database and create indexes"""
    db = MongoDB.get_db()
    
    # Create indexes for users collection
    db.users.create_index([("username", ASCENDING)], unique=True)
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.users.create_index([("created_by_id", ASCENDING)])
    db.users.create_index([("role", ASCENDING)])
    # Compound index for admin access check (get agents by admin)
    db.users.create_index([("created_by_id", ASCENDING), ("role", ASCENDING)])
    
    # Create indexes for leads collection
    db.leads.create_index([("owner_id", ASCENDING)])
    db.leads.create_index([("country", ASCENDING)])
    db.leads.create_index([("import_batch_id", ASCENDING)])
    
    # Create indexes for campaigns collection
    db.campaigns.create_index([("owner_id", ASCENDING)])
    db.campaigns.create_index([("status", ASCENDING)])
    
    # Create indexes for campaign_leads collection
    db.campaign_leads.create_index([("campaign_id", ASCENDING)])
    db.campaign_leads.create_index([("lead_id", ASCENDING)])
    db.campaign_leads.create_index([("status", ASCENDING)])
    
    # Create indexes for call_sessions collection
    db.call_sessions.create_index([("session_id", ASCENDING)], unique=True)
    db.call_sessions.create_index([("owner_id", ASCENDING)])
    db.call_sessions.create_index([("campaign_id", ASCENDING)])
    db.call_sessions.create_index([("lead_id", ASCENDING)])
    
    # Create indexes for payment_requests collection
    db.payment_requests.create_index([("admin_id", ASCENDING)])
    db.payment_requests.create_index([("status", ASCENDING)])
    
    # Create indexes for slot_adjustments collection
    db.slot_adjustments.create_index([("admin_id", ASCENDING)])
    db.slot_adjustments.create_index([("created_at", DESCENDING)])
    
    # Create indexes for system_settings collection
    db.system_settings.create_index([("key", ASCENDING)], unique=True)
    
    return db

# Mock session class for compatibility with existing code
class MongoSession:
    """Mock session class to maintain compatibility with SQLAlchemy-style code"""
    def __init__(self):
        self.db = MongoDB.get_db()
        self._pending_saves = []
        self._pending_deletes = []
    
    def add(self, obj):
        """Add object to session (will be saved on commit)"""
        self._pending_saves.append(obj)
    
    def delete(self, obj):
        """Delete object from session (will be deleted on commit)"""
        self._pending_deletes.append(obj)
    
    def refresh(self, obj):
        """Refresh object from database"""
        # For MongoDB, we need to reload from database
        if hasattr(obj, 'id') and obj.id:
            collection_name = self._get_collection_name(obj)
            doc = self.db[collection_name].find_one({"id": obj.id})
            if doc:
                # Update object attributes
                for key, value in doc.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
    
    def _get_collection_name(self, obj):
        """Get collection name from object"""
        mapping = {
            'User': 'users',
            'Lead': 'leads',
            'Campaign': 'campaigns',
            'CampaignLead': 'campaign_leads',
            'CallSession': 'call_sessions',
            'PaymentRequest': 'payment_requests',
            'SlotAdjustment': 'slot_adjustments',
            'SystemSettings': 'system_settings'
        }
        return mapping.get(obj.__class__.__name__, obj.__class__.__name__.lower() + 's')
    
    def close(self):
        """Close session (no-op for MongoDB)"""
        self._pending_saves.clear()
        self._pending_deletes.clear()
    
    def commit(self):
        """Commit changes - save/delete pending objects"""
        # Save all pending objects
        for obj in self._pending_saves:
            if hasattr(obj, 'save'):
                obj.save()
        
        # Delete all pending objects
        for obj in self._pending_deletes:
            if hasattr(obj, 'delete'):
                obj.delete()
        
        # Clear pending operations
        self._pending_saves.clear()
        self._pending_deletes.clear()
    
    def rollback(self):
        """Rollback changes - clear pending operations"""
        self._pending_saves.clear()
        self._pending_deletes.clear()
    
    def query(self, model):
        """Return a query builder for the model"""
        return MongoQuery(self.db, model)

class MongoQuery:
    """Query builder to maintain compatibility with SQLAlchemy-style queries"""
    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.filters = {}
        self.sort_field = None
        self.sort_direction = ASCENDING
        self.limit_val = None
        self.offset_val = 0
        self.collection_name = self._get_collection_name(model)
        self._options = []
    
    def _get_collection_name(self, model):
        """Map model class to collection name"""
        mapping = {
            'User': 'users',
            'Lead': 'leads',
            'Campaign': 'campaigns',
            'CampaignLead': 'campaign_leads',
            'CallSession': 'call_sessions',
            'PaymentRequest': 'payment_requests',
            'SlotAdjustment': 'slot_adjustments',
            'SystemSettings': 'system_settings'
        }
        
        # Handle different input types
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, '__name__'):
            # Regular class
            model_name = model.__name__
        elif hasattr(model, 'name'):
            # ColumnExpression or similar
            model_name = model.name
        else:
            # Try to get class name from type
            model_name = type(model).__name__
        
        return mapping.get(model_name, model_name.lower() + 's')
    
    def _convert_value(self, value):
        """Convert value to MongoDB-compatible format"""
        # Convert enums to their string values
        if isinstance(value, enum.Enum):
            return value.value
        # Convert lists/tuples that might contain enums
        elif isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]
        return value
    
    def filter(self, *args, **kwargs):
        """Add filter conditions"""
        for arg in args:
            # Handle FilterExpression objects
            if isinstance(arg, FilterExpression):
                # Handle OR conditions
                if arg.is_or and arg.or_conditions:
                    or_conditions = []
                    for condition in arg.or_conditions:
                        field = condition.field
                        operator = condition.operator
                        value = self._convert_value(condition.value)
                        
                        if operator == '==':
                            or_conditions.append({field: value})
                        elif operator == '!=':
                            or_conditions.append({field: {"$ne": value}})
                        elif operator == 'in':
                            or_conditions.append({field: {"$in": value}})
                        elif operator == 'is_not':
                            or_conditions.append({field: {"$ne": value}})
                        elif operator == 'contains':
                            # Use regex for substring match
                            or_conditions.append({field: {"$regex": str(value), "$options": "i"}})
                        else:
                            or_conditions.append({field: value})
                    
                    # Add OR conditions to filters
                    if "$or" in self.filters:
                        self.filters["$or"].extend(or_conditions)
                    else:
                        self.filters["$or"] = or_conditions
                else:
                    # Handle single condition
                    field = arg.field
                    operator = arg.operator
                    value = self._convert_value(arg.value)
                    
                    if operator == '==':
                        self.filters[field] = value
                    elif operator == '!=':
                        self.filters[field] = {"$ne": value}
                    elif operator == 'in':
                        self.filters[field] = {"$in": value}
                    elif operator == 'is_not':
                        self.filters[field] = {"$ne": value}
                    elif operator == 'contains':
                        # Use regex for substring match
                        self.filters[field] = {"$regex": str(value), "$options": "i"}
                    else:
                        self.filters[field] = value
            # Handle ColumnExpression with special methods
            elif hasattr(arg, 'field') and hasattr(arg, 'value'):
                self.filters[arg.field] = self._convert_value(arg.value)
        
        for key, value in kwargs.items():
            self.filters[key] = self._convert_value(value)
        
        return self
    
    def filter_by(self, **kwargs):
        """Add filter conditions by keyword"""
        self.filters.update(kwargs)
        return self
    
    def options(self, *args):
        """Handle query options (like joinedload) - stored but not used in MongoDB"""
        self._options.extend(args)
        return self
    
    def order_by(self, *args):
        """Set sort order"""
        if args:
            field = args[0]
            if hasattr(field, '_desc') and field._desc:
                self.sort_field = field.key
                self.sort_direction = DESCENDING
            else:
                self.sort_field = field.key if hasattr(field, 'key') else str(field)
                self.sort_direction = ASCENDING
        return self
    
    def limit(self, limit):
        """Set limit"""
        self.limit_val = limit
        return self
    
    def offset(self, offset):
        """Set offset"""
        self.offset_val = offset
        return self
    
    def count(self):
        """Count matching documents"""
        collection = self.db[self.collection_name]
        return collection.count_documents(self.filters)
    
    def all(self):
        """Get all matching documents"""
        collection = self.db[self.collection_name]
        cursor = collection.find(self.filters)
        
        if self.sort_field:
            cursor = cursor.sort(self.sort_field, self.sort_direction)
        
        if self.offset_val:
            cursor = cursor.skip(self.offset_val)
        
        if self.limit_val:
            cursor = cursor.limit(self.limit_val)
        
        # Convert documents to model instances
        return [self._doc_to_model(doc) for doc in cursor]
    
    def first(self):
        """Get first matching document"""
        collection = self.db[self.collection_name]
        doc = collection.find_one(self.filters)
        return self._doc_to_model(doc) if doc else None
    
    def get(self, id):
        """Get document by ID"""
        collection = self.db[self.collection_name]
        doc = collection.find_one({"id": id})
        return self._doc_to_model(doc) if doc else None
    
    def _doc_to_model(self, doc):
        """Convert MongoDB document to model instance"""
        if not doc:
            return None
        
        # Create appropriate model instance based on collection
        if self.collection_name == 'users':
            return User.from_dict(doc)
        elif self.collection_name == 'leads':
            return Lead.from_dict(doc)
        elif self.collection_name == 'campaigns':
            return Campaign.from_dict(doc)
        elif self.collection_name == 'campaign_leads':
            return CampaignLead.from_dict(doc)
        elif self.collection_name == 'call_sessions':
            return CallSession.from_dict(doc)
        elif self.collection_name == 'payment_requests':
            return PaymentRequest.from_dict(doc)
        elif self.collection_name == 'slot_adjustments':
            return SlotAdjustment.from_dict(doc)
        elif self.collection_name == 'system_settings':
            return SystemSettings.from_dict(doc)
        
        return doc

def get_session():
    """Get database session"""
    return MongoSession()

# Column descriptor for SQLAlchemy-style class attribute access
class Column:
    """Descriptor to handle class-level attribute access like SQLAlchemy columns"""
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class attribute access (e.g., User.id)
            return ColumnExpression(self.name)
        # Instance attribute access
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

class ColumnExpression:
    """Represents a column expression for query building"""
    def __init__(self, name):
        self.name = name
        self.key = name
    
    def __eq__(self, other):
        return FilterExpression(self.name, '==', other)
    
    def __ne__(self, other):
        return FilterExpression(self.name, '!=', other)
    
    def in_(self, values):
        return FilterExpression(self.name, 'in', values)
    
    def isnot(self, value):
        return FilterExpression(self.name, 'is_not', value)
    
    def contains(self, value):
        """Check if column contains value (substring match)"""
        return FilterExpression(self.name, 'contains', value)
    
    def desc(self):
        """Mark column for descending sort"""
        expr = ColumnExpression(self.name)
        expr._desc = True
        return expr

class Relationship:
    """Mock relationship for SQLAlchemy compatibility"""
    def __init__(self, name):
        self.name = name
        self.key = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            # Class attribute access
            return self
        # Instance attribute access - return the lazy-loaded property
        return obj.__dict__.get(f'_{self.name}')

class FilterExpression:
    """Represents a filter expression"""
    def __init__(self, field, operator, value):
        self.left = ColumnExpression(field) if isinstance(field, str) else field
        self.right = type('Value', (), {'value': value})()
        self.operator = operator
        self.field = field if isinstance(field, str) else None
        self.value = value
        self.is_or = False  # Flag to indicate if this is an OR expression
        self.or_conditions = []  # List of conditions for OR
    
    def __or__(self, other):
        """Support | operator for OR conditions"""
        or_expr = FilterExpression(None, 'or', None)
        or_expr.is_or = True
        
        # Collect all conditions
        conditions = []
        if self.is_or:
            conditions.extend(self.or_conditions)
        else:
            conditions.append(self)
        
        if other.is_or:
            conditions.extend(other.or_conditions)
        else:
            conditions.append(other)
        
        or_expr.or_conditions = conditions
        return or_expr

# Model Classes
class User:
    """User model for authentication and authorization"""
    
    # Class-level column descriptors for SQLAlchemy-style access
    id = Column('id')
    username = Column('username')
    email = Column('email')
    hashed_password = Column('hashed_password')
    role = Column('role')
    created_by_id = Column('created_by_id')
    organization = Column('organization')
    max_agents = Column('max_agents')
    subscription_end_date = Column('subscription_end_date')
    first_name = Column('first_name')
    last_name = Column('last_name')
    is_active = Column('is_active')
    created_at = Column('created_at')
    updated_at = Column('updated_at')
    last_login = Column('last_login')
    gate_slot = Column('gate_slot')
    google_api_key = Column('google_api_key')
    
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
        self.gate_slot = kwargs.get('gate_slot')  # Gate slot number (9-19) for outbound calls
        self.google_api_key = kwargs.get('google_api_key')  # Google API key for Gemini
    
    @property
    def full_name(self):
        """Get full name"""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else self.username
    
    def verify_password(self, password: str) -> bool:
        """Verify password - truncate to 72 bytes for bcrypt compatibility"""
        # Bcrypt only supports passwords up to 72 bytes
        # Truncate the password if it's longer (encode to bytes first to handle UTF-8 correctly)
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password = password_bytes[:72].decode('utf-8', errors='ignore')
        return pwd_context.verify(password, self.hashed_password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password - truncate to 72 bytes for bcrypt compatibility"""
        # Bcrypt only supports passwords up to 72 bytes
        # Truncate the password if it's longer (encode to bytes first to handle UTF-8 correctly)
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password = password_bytes[:72].decode('utf-8', errors='ignore')
        return pwd_context.hash(password)
    
    def is_subscription_active(self) -> bool:
        """Check if user's subscription is active (for admins and their agents)"""
        if self.role == UserRole.SUPERADMIN:
            return True
        
        if self.role == UserRole.AGENT:
            return True  # Checked via their admin
        
        if self.role == UserRole.ADMIN:
            if not self.subscription_end_date:
                return False
            return self.subscription_end_date > datetime.utcnow()
        
        return False
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "hashed_password": self.hashed_password,
            "role": get_enum_value(self.role),
            "created_by_id": self.created_by_id,
            "organization": self.organization,
            "max_agents": self.max_agents,
            "subscription_end_date": self.subscription_end_date,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login,
            "gate_slot": self.gate_slot,
            "google_api_key": self.google_api_key
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            # Update existing
            db.users.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            # Insert new
            if not doc.get('id'):
                # Generate new ID
                max_id = db.users.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.users.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.users.delete_one({"id": self.id})

class Lead:
    """Lead/Contact model"""
    
    # Class-level column descriptors
    id = Column('id')
    owner_id = Column('owner_id')
    first_name = Column('first_name')
    last_name = Column('last_name')
    email = Column('email')
    phone = Column('phone')
    country = Column('country')
    country_code = Column('country_code')
    gender = Column('gender')
    address = Column('address')
    created_at = Column('created_at')
    updated_at = Column('updated_at')
    last_called_at = Column('last_called_at')
    call_count = Column('call_count')
    notes = Column('notes')
    custom_data = Column('custom_data')
    import_batch_id = Column('import_batch_id')
    
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
    
    @property
    def full_phone(self):
        """Get full phone number with country code"""
        return f"{self.country_code}{self.phone}" if self.country_code else self.phone
    
    @property
    def full_name(self):
        """Get full name"""
        parts = [p for p in [self.first_name, self.last_name] if p]
        return " ".join(parts) if parts else "Unknown"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "country": self.country,
            "country_code": self.country_code,
            "gender": get_enum_value(self.gender),
            "address": self.address,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_called_at": self.last_called_at,
            "call_count": self.call_count,
            "notes": self.notes,
            "custom_data": self.custom_data,
            "import_batch_id": self.import_batch_id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.leads.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.leads.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.leads.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.leads.delete_one({"id": self.id})

class Campaign:
    """Campaign model for organizing and executing call campaigns"""
    
    # Class-level column descriptors
    id = Column('id')
    owner_id = Column('owner_id')
    name = Column('name')
    description = Column('description')
    status = Column('status')
    bot_config = Column('bot_config')
    dialing_config = Column('dialing_config')
    schedule_config = Column('schedule_config')
    total_leads = Column('total_leads')
    leads_called = Column('leads_called')
    leads_answered = Column('leads_answered')
    leads_rejected = Column('leads_rejected')
    leads_failed = Column('leads_failed')
    created_at = Column('created_at')
    updated_at = Column('updated_at')
    started_at = Column('started_at')
    completed_at = Column('completed_at')
    
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
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "description": self.description,
            "status": get_enum_value(self.status),
            "bot_config": self.bot_config,
            "dialing_config": self.dialing_config,
            "schedule_config": self.schedule_config,
            "total_leads": self.total_leads,
            "leads_called": self.leads_called,
            "leads_answered": self.leads_answered,
            "leads_rejected": self.leads_rejected,
            "leads_failed": self.leads_failed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.campaigns.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.campaigns.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.campaigns.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.campaigns.delete_one({"id": self.id})

class CampaignLead:
    """Many-to-many relationship between campaigns and leads with additional data"""
    
    # Class-level column descriptors
    id = Column('id')
    campaign_id = Column('campaign_id')
    lead_id = Column('lead_id')
    status = Column('status')
    priority = Column('priority')
    call_attempts = Column('call_attempts')
    last_attempt_at = Column('last_attempt_at')
    scheduled_for = Column('scheduled_for')
    call_session_id = Column('call_session_id')
    call_duration = Column('call_duration')
    call_result = Column('call_result')
    call_notes = Column('call_notes')
    added_at = Column('added_at')
    called_at = Column('called_at')
    completed_at = Column('completed_at')
    
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
        
        # Lazy load lead if needed
        self._lead = None
    
    @property
    def lead(self):
        """Get associated lead"""
        if not self._lead and self.lead_id:
            db = MongoDB.get_db()
            lead_doc = db.leads.find_one({"id": self.lead_id})
            if lead_doc:
                self._lead = Lead.from_dict(lead_doc)
        return self._lead
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "lead_id": self.lead_id,
            "status": get_enum_value(self.status),
            "priority": self.priority,
            "call_attempts": self.call_attempts,
            "last_attempt_at": self.last_attempt_at,
            "scheduled_for": self.scheduled_for,
            "call_session_id": self.call_session_id,
            "call_duration": self.call_duration,
            "call_result": self.call_result,
            "call_notes": self.call_notes,
            "added_at": self.added_at,
            "called_at": self.called_at,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.campaign_leads.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.campaign_leads.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.campaign_leads.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.campaign_leads.delete_one({"id": self.id})

class CallSession:
    """Track individual call sessions with links to voice agent sessions"""
    
    # Class-level column descriptors
    id = Column('id')
    session_id = Column('session_id')
    campaign_id = Column('campaign_id')
    lead_id = Column('lead_id')
    owner_id = Column('owner_id')
    caller_id = Column('caller_id')
    called_number = Column('called_number')
    status = Column('status')
    started_at = Column('started_at')
    answered_at = Column('answered_at')
    ended_at = Column('ended_at')
    duration = Column('duration')
    talk_time = Column('talk_time')
    recording_path = Column('recording_path')
    transcript_status = Column('transcript_status')
    transcript_language = Column('transcript_language')
    sentiment_score = Column('sentiment_score')
    interest_level = Column('interest_level')
    key_points = Column('key_points')
    follow_up_required = Column('follow_up_required')
    follow_up_notes = Column('follow_up_notes')
    call_metadata = Column('call_metadata')
    created_at = Column('created_at')
    updated_at = Column('updated_at')
    # MongoDB-specific fields
    transcripts = Column('transcripts')
    analysis = Column('analysis')
    audio_files = Column('audio_files')
    session_info = Column('session_info')
    asterisk_linkedid = Column('asterisk_linkedid')
    
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
        
        # MongoDB-specific fields for session data
        self.transcripts = kwargs.get('transcripts')
        self.analysis = kwargs.get('analysis')
        self.audio_files = kwargs.get('audio_files')
        self.session_info = kwargs.get('session_info')
        self.asterisk_linkedid = kwargs.get('asterisk_linkedid')
        
        # Lazy load lead if needed
        self._lead = None
    
    @property
    def lead(self):
        """Get associated lead"""
        if not self._lead and self.lead_id:
            db = MongoDB.get_db()
            lead_doc = db.leads.find_one({"id": self.lead_id})
            if lead_doc:
                self._lead = Lead.from_dict(lead_doc)
        return self._lead
    
    def to_dict(self):
        """Convert to dictionary"""
        data = {
            "id": self.id,
            "session_id": self.session_id,
            "campaign_id": self.campaign_id,
            "lead_id": self.lead_id,
            "owner_id": self.owner_id,
            "caller_id": self.caller_id,
            "called_number": self.called_number,
            "status": get_enum_value(self.status),
            "started_at": self.started_at,
            "answered_at": self.answered_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "talk_time": self.talk_time,
            "recording_path": self.recording_path,
            "transcript_status": self.transcript_status,
            "transcript_language": self.transcript_language,
            "sentiment_score": self.sentiment_score,
            "interest_level": self.interest_level,
            "key_points": self.key_points,
            "follow_up_required": self.follow_up_required,
            "follow_up_notes": self.follow_up_notes,
            "call_metadata": self.call_metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # Add MongoDB-specific fields if they exist
        if hasattr(self, 'asterisk_linkedid') and self.asterisk_linkedid is not None:
            data['asterisk_linkedid'] = self.asterisk_linkedid
        if hasattr(self, 'transcripts') and self.transcripts is not None:
            data['transcripts'] = self.transcripts
        if hasattr(self, 'analysis') and self.analysis is not None:
            data['analysis'] = self.analysis
        if hasattr(self, 'audio_files') and self.audio_files is not None:
            data['audio_files'] = self.audio_files
        if hasattr(self, 'session_info') and self.session_info is not None:
            data['session_info'] = self.session_info
            
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.call_sessions.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.call_sessions.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.call_sessions.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.call_sessions.delete_one({"id": self.id})

class PaymentRequest:
    """Payment request model for admin agent slot purchases"""
    
    # Class-level column descriptors
    id = Column('id')
    admin_id = Column('admin_id')
    num_agents = Column('num_agents')
    total_amount = Column('total_amount')
    status = Column('status')
    payment_notes = Column('payment_notes')
    admin_notes = Column('admin_notes')
    created_at = Column('created_at')
    updated_at = Column('updated_at')
    approved_at = Column('approved_at')
    approved_by_id = Column('approved_by_id')
    
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
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "admin_id": self.admin_id,
            "num_agents": self.num_agents,
            "total_amount": self.total_amount,
            "status": get_enum_value(self.status),
            "payment_notes": self.payment_notes,
            "admin_notes": self.admin_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approved_at": self.approved_at,
            "approved_by_id": self.approved_by_id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.payment_requests.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.payment_requests.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.payment_requests.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.payment_requests.delete_one({"id": self.id})

class SlotAdjustment:
    """Manual slot adjustment model for tracking superadmin changes to admin agent slots"""
    
    # Class-level column descriptors
    id = Column('id')
    admin_id = Column('admin_id')
    adjusted_by_id = Column('adjusted_by_id')
    slots_change = Column('slots_change')  # Positive for increase, negative for decrease
    reason = Column('reason')
    previous_max_agents = Column('previous_max_agents')
    new_max_agents = Column('new_max_agents')
    created_at = Column('created_at')
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.admin_id = kwargs.get('admin_id')
        self.adjusted_by_id = kwargs.get('adjusted_by_id')
        self.slots_change = kwargs.get('slots_change')
        self.reason = kwargs.get('reason', '')
        self.previous_max_agents = kwargs.get('previous_max_agents')
        self.new_max_agents = kwargs.get('new_max_agents')
        self.created_at = kwargs.get('created_at', datetime.utcnow())
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "admin_id": self.admin_id,
            "adjusted_by_id": self.adjusted_by_id,
            "slots_change": self.slots_change,
            "reason": self.reason,
            "previous_max_agents": self.previous_max_agents,
            "new_max_agents": self.new_max_agents,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.slot_adjustments.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.slot_adjustments.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.slot_adjustments.insert_one(doc)
        
        return self
    
    def delete(self):
        """Delete from database"""
        db = MongoDB.get_db()
        db.slot_adjustments.delete_one({"id": self.id})

class SystemSettings:
    """System-wide settings"""
    
    # Class-level column descriptors
    id = Column('id')
    key = Column('key')
    value = Column('value')
    updated_at = Column('updated_at')
    updated_by_id = Column('updated_by_id')
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.key = kwargs.get('key')
        self.value = kwargs.get('value')
        self.updated_at = kwargs.get('updated_at', datetime.utcnow())
        self.updated_by_id = kwargs.get('updated_by_id')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at,
            "updated_by_id": self.updated_by_id
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)
    
    def save(self):
        """Save to database"""
        db = MongoDB.get_db()
        doc = self.to_dict()
        
        if self.id:
            db.system_settings.update_one({"id": self.id}, {"$set": doc}, upsert=True)
        else:
            if not doc.get('id'):
                max_id = db.system_settings.find_one(sort=[("id", DESCENDING)])
                doc['id'] = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
                self.id = doc['id']
            
            db.system_settings.insert_one(doc)
        
        return self

# Manager Classes
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
        user.save()
        return user
    
    def get_user_by_username(self, username):
        """Get user by username"""
        doc = self.db.users.find_one({"username": username})
        return User.from_dict(doc) if doc else None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        doc = self.db.users.find_one({"email": email})
        return User.from_dict(doc) if doc else None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        doc = self.db.users.find_one({"id": user_id})
        return User.from_dict(doc) if doc else None
    
    def get_agents_by_admin(self, admin_id):
        """Get all agents created by an admin"""
        docs = self.db.users.find({"created_by_id": admin_id, "role": UserRole.AGENT.value})
        return [User.from_dict(doc) for doc in docs]
    
    def update_user(self, user_id, **kwargs):
        """Update user fields"""
        user = self.get_user_by_id(user_id)
        if user:
            for key, value in kwargs.items():
                if key == 'password':
                    user.hashed_password = User.hash_password(value)
                elif hasattr(user, key):
                    setattr(user, key, value)
            user.save()
        return user
    
    def delete_user(self, user_id):
        """Delete a user"""
        user = self.get_user_by_id(user_id)
        if user:
            user.delete()
            # Also delete their leads and campaigns
            self.db.leads.delete_many({"owner_id": user_id})
            self.db.campaigns.delete_many({"owner_id": user_id})
            return True
        return False
    
    def get_available_gate_slots(self):
        """Get list of available gate slots (9-19)"""
        # Get all assigned gate slots
        assigned_slots = set()
        agents = self.db.users.find({"role": UserRole.AGENT.value, "gate_slot": {"$ne": None}})
        for agent in agents:
            if agent.get('gate_slot'):
                assigned_slots.add(agent['gate_slot'])
        
        # Available slots are 9-19 (inclusive)
        all_slots = set(range(9, 20))  # 9 to 19 inclusive
        available_slots = sorted(all_slots - assigned_slots)
        return available_slots
    
    def assign_gate_slot(self, user_id):
        """Automatically assign a free gate slot to an agent"""
        available_slots = self.get_available_gate_slots()
        
        if not available_slots:
            return None  # No available slots
        
        # Assign the first available slot
        slot = available_slots[0]
        user = self.get_user_by_id(user_id)
        if user:
            user.gate_slot = slot
            user.save()
            return slot
        return None
    
    def free_gate_slot(self, user_id):
        """Free up the gate slot assigned to an agent"""
        user = self.get_user_by_id(user_id)
        if user and user.gate_slot:
            user.gate_slot = None
            user.save()
            return True
        return False
    
    def get_available_api_keys(self):
        """Get list of available Google API keys from environment"""
        import os
        available_keys = []
        
        # Check GOOGLE_API_KEY (primary)
        primary_key = os.getenv('GOOGLE_API_KEY')
        if primary_key:
            available_keys.append(primary_key)
        
        # Check GOOGLE_API_KEY_2 through GOOGLE_API_KEY_10
        for i in range(2, 11):
            key = os.getenv(f'GOOGLE_API_KEY_{i}')
            if key:
                available_keys.append(key)
        
        return available_keys
    
    def get_assigned_api_keys(self):
        """Get set of already assigned API keys"""
        assigned_keys = set()
        users = self.db.users.find({"google_api_key": {"$ne": None}})
        for user in users:
            if user.get('google_api_key'):
                assigned_keys.add(user['google_api_key'])
        return assigned_keys
    
    def assign_api_key(self, user_id):
        """Automatically assign a free Google API key to a user"""
        available_keys = self.get_available_api_keys()
        assigned_keys = self.get_assigned_api_keys()
        
        # Find keys that are available but not yet assigned
        free_keys = [key for key in available_keys if key not in assigned_keys]
        
        if not free_keys:
            # If all keys are assigned, round-robin assign (reuse keys)
            # This allows multiple users to share keys if we run out
            if available_keys:
                # Get count of users per key and assign to the one with least users
                key_usage = {}
                for key in available_keys:
                    count = self.db.users.count_documents({"google_api_key": key})
                    key_usage[key] = count
                
                # Get key with minimum usage
                assigned_key = min(key_usage.items(), key=lambda x: x[1])[0]
            else:
                return None  # No API keys available at all
        else:
            # Assign the first free key
            assigned_key = free_keys[0]
        
        user = self.get_user_by_id(user_id)
        if user:
            user.google_api_key = assigned_key
            user.save()
            return assigned_key
        return None

class LeadManager:
    """Manager class for lead operations"""
    
    def __init__(self, session):
        self.session = session
        self.db = session.db
    
    def create_lead(self, lead_data):
        """Create a new lead"""
        lead = Lead(**lead_data)
        lead.save()
        return lead
    
    def bulk_import_leads(self, leads_data, import_batch_id=None):
        """Import multiple leads at once"""
        if import_batch_id is None:
            import_batch_id = f"import_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Get next ID
        max_id = self.db.leads.find_one(sort=[("id", DESCENDING)])
        next_id = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
        
        docs = []
        for lead_data in leads_data:
            lead_data['import_batch_id'] = import_batch_id
            lead_data['id'] = next_id
            lead = Lead(**lead_data)
            docs.append(lead.to_dict())
            next_id += 1
        
        if docs:
            self.db.leads.insert_many(docs)
        
        return len(docs)
    
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
        
        leads = [Lead.from_dict(doc) for doc in cursor]
        
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
        campaign.save()
        return campaign
    
    def add_leads_to_campaign(self, campaign_id, lead_ids, priority=0):
        """Add leads to a campaign"""
        # Get next ID
        max_id = self.db.campaign_leads.find_one(sort=[("id", DESCENDING)])
        next_id = (max_id['id'] + 1) if max_id and 'id' in max_id else 1
        
        docs = []
        for lead_id in lead_ids:
            cl = CampaignLead(
                id=next_id,
                campaign_id=campaign_id,
                lead_id=lead_id,
                priority=priority
            )
            docs.append(cl.to_dict())
            next_id += 1
        
        if docs:
            self.db.campaign_leads.insert_many(docs)
        
        # Update campaign total leads
        campaign_doc = self.db.campaigns.find_one({"id": campaign_id})
        if campaign_doc:
            campaign = Campaign.from_dict(campaign_doc)
            campaign.total_leads = len(lead_ids)
            campaign.save()
        
        return len(docs)
    
    def get_next_lead_to_call(self, campaign_id):
        """Get the next lead to call in a campaign"""
        doc = self.db.campaign_leads.find_one(
            {"campaign_id": campaign_id, "status": CallStatus.PENDING.value},
            sort=[("priority", DESCENDING), ("added_at", ASCENDING)]
        )
        
        if doc:
            campaign_lead = CampaignLead.from_dict(doc)
            # Mark as dialing
            campaign_lead.status = CallStatus.DIALING
            campaign_lead.call_attempts += 1
            campaign_lead.last_attempt_at = datetime.utcnow()
            campaign_lead.save()
            return campaign_lead
        
        return None
    
    def update_call_result(self, campaign_id, lead_id, session_id, status, duration=None):
        """Update the result of a call"""
        # Update campaign lead
        cl_doc = self.db.campaign_leads.find_one({
            "campaign_id": campaign_id,
            "lead_id": lead_id
        })
        
        if cl_doc:
            campaign_lead = CampaignLead.from_dict(cl_doc)
            campaign_lead.status = status
            campaign_lead.call_session_id = session_id
            if duration:
                campaign_lead.call_duration = duration
            if status in [CallStatus.ANSWERED, CallStatus.REJECTED, CallStatus.NO_ANSWER]:
                campaign_lead.completed_at = datetime.utcnow()
            campaign_lead.save()
            
            # Update lead's last called info
            lead_doc = self.db.leads.find_one({"id": lead_id})
            if lead_doc:
                lead = Lead.from_dict(lead_doc)
                lead.last_called_at = datetime.utcnow()
                lead.call_count += 1
                lead.save()
            
            # Update campaign statistics
            campaign_doc = self.db.campaigns.find_one({"id": campaign_id})
            if campaign_doc:
                campaign = Campaign.from_dict(campaign_doc)
                campaign.leads_called += 1
                if status == CallStatus.ANSWERED:
                    campaign.leads_answered += 1
                elif status == CallStatus.REJECTED:
                    campaign.leads_rejected += 1
                elif status in [CallStatus.FAILED, CallStatus.NO_ANSWER]:
                    campaign.leads_failed += 1
                campaign.save()

if __name__ == "__main__":
    # Initialize database
    db = init_database()
    print("MongoDB database initialized successfully!")
    print(f"Database: {db.name}")
    print(f"Collections: {db.list_collection_names()}")

