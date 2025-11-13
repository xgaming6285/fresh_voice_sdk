# -*- coding: utf-8 -*-
"""
Session MongoDB Helper
Updates MongoDB with session transcripts, analysis, and metadata
"""

import json
import os
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_mongodb_connection():
    """Get MongoDB connection"""
    try:
        # Check for MongoDB connection string (Atlas or full URI)
        mongo_uri = os.getenv('MONGO_DB') or os.getenv('MONGODB_URI')
        
        if mongo_uri:
            # Use connection string (MongoDB Atlas or full URI)
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        else:
            # Fallback to host:port format for local MongoDB
            mongo_host = os.getenv('MONGODB_HOST', 'localhost')
            mongo_port = int(os.getenv('MONGODB_PORT', '27017'))
            client = MongoClient(mongo_host, mongo_port, serverSelectionTimeoutMS=2000)
        
        db_name = os.getenv('MONGODB_DATABASE', 'voice_agent_crm')
        db = client[db_name]
        
        # Test connection
        client.admin.command('ping')
        return db
    except Exception as e:
        print(f"WARNING: MongoDB connection failed: {e}")
        return None

def update_session_in_mongodb(session_id, update_data):
    """
    Update session in MongoDB with transcripts, analysis, or other data
    
    Args:
        session_id: The session ID
        update_data: Dictionary of data to update
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_mongodb_connection()
        if db is None:
            return False
        
        # Add timestamp
        update_data['updated_at'] = datetime.utcnow()
        
        # Update the session
        result = db.call_sessions.update_one(
            {"session_id": session_id},
            {"$set": update_data}
        )
        
        return result.matched_count > 0
    
    except Exception as e:
        print(f"ERROR updating session in MongoDB: {e}")
        return False

def save_transcripts_to_mongodb(session_id, session_dir):
    """
    Read transcript files and save to MongoDB
    
    Args:
        session_id: The session ID
        session_dir: Path to session directory
    """
    try:
        transcripts = {}
        
        for transcript_type in ['incoming', 'outgoing', 'mixed']:
            # Find transcript file
            transcript_files = list(Path(session_dir).glob(f"{transcript_type}_*.txt"))
            
            if transcript_files:
                transcript_file = transcript_files[0]
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse transcript file - skip header lines
                    lines = content.split('\n')
                    transcript_text = '\n'.join(lines[6:]).strip() if len(lines) > 6 else content
                    
                    transcripts[transcript_type] = {
                        'content': transcript_text,   # Frontend expects 'content'
                        'success': True,
                        'file': transcript_file.name,
                        'length': len(transcript_text),
                        'transcribed_at': datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    print(f"ERROR reading {transcript_type} transcript: {e}")
        
        if transcripts:
            update_session_in_mongodb(session_id, {'transcripts': transcripts})
            print(f"✅ Saved {len(transcripts)} transcripts to MongoDB for session {session_id}")
            return True
        
        return False
    
    except Exception as e:
        print(f"ERROR saving transcripts to MongoDB: {e}")
        return False


def save_transcripts_to_mongodb_with_conversation(session_id, session_dir, conversation_data=None):
    """
    Read transcript files and save to MongoDB with conversation structure
    
    Args:
        session_id: The session ID
        session_dir: Path to session directory
        conversation_data: Optional structured conversation data [{"speaker": "agent", "text": "..."}, ...]
    """
    try:
        transcripts = {}
        
        for transcript_type in ['incoming', 'outgoing', 'mixed']:
            # Find transcript file
            transcript_files = list(Path(session_dir).glob(f"{transcript_type}_*.txt"))
            
            if transcript_files:
                transcript_file = transcript_files[0]
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse transcript file - skip header lines
                    lines = content.split('\n')
                    transcript_text = '\n'.join(lines[6:]).strip() if len(lines) > 6 else content
                    
                    transcript_entry = {
                        'content': transcript_text,   # Frontend expects 'content'
                        'success': True,
                        'file': transcript_file.name,
                        'length': len(transcript_text),
                        'transcribed_at': datetime.utcnow().isoformat()
                    }
                    
                    # Add conversation structure for mixed (full conversation)
                    if transcript_type == 'mixed' and conversation_data:
                        transcript_entry['conversation'] = conversation_data
                    
                    transcripts[transcript_type] = transcript_entry
                    
                except Exception as e:
                    print(f"ERROR reading {transcript_type} transcript: {e}")
        
        if transcripts:
            update_session_in_mongodb(session_id, {'transcripts': transcripts})
            print(f"✅ Saved {len(transcripts)} transcripts to MongoDB for session {session_id}")
            if conversation_data:
                print(f"💬 Including structured conversation with {len(conversation_data)} turns")
            return True
        
        return False
    
    except Exception as e:
        print(f"ERROR saving transcripts to MongoDB: {e}")
        return False

def save_analysis_to_mongodb(session_id, session_dir):
    """
    Read analysis_result.json and save to MongoDB
    
    Args:
        session_id: The session ID
        session_dir: Path to session directory
    """
    try:
        analysis_file = Path(session_dir) / "analysis_result.json"
        
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            update_data = {
                'analysis': analysis,
                'analysis_saved_at': datetime.utcnow().isoformat()
            }
            
            # Update CRM session status if we have interest level
            if 'status' in analysis:
                status_mapping = {
                    'interested': 'answered',
                    'not_interested': 'rejected',
                    'callback': 'answered',
                    'no_answer': 'no_answer'
                }
                crm_status = status_mapping.get(analysis['status'].lower(), 'completed')
                update_data['status'] = crm_status
            
            update_session_in_mongodb(session_id, update_data)
            print(f"✅ Saved analysis to MongoDB for session {session_id}")
            return True
        
        return False
    
    except Exception as e:
        print(f"ERROR saving analysis to MongoDB: {e}")
        return False

def save_session_info_to_mongodb(session_id, session_dir):
    """
    Read session_info.json and save to MongoDB
    
    Args:
        session_id: The session ID
        session_dir: Path to session directory
    """
    try:
        info_file = Path(session_dir) / "session_info.json"
        
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            
            # Also save audio file paths
            audio_files = {}
            for audio_type in ['incoming', 'outgoing', 'mixed']:
                audio_pattern = list(Path(session_dir).glob(f"{audio_type}_*.wav"))
                if audio_pattern:
                    audio_files[audio_type] = str(audio_pattern[0])
            
            update_data = {
                'session_info': session_info,
                'audio_files': audio_files
            }
            
            update_session_in_mongodb(session_id, update_data)
            print(f"✅ Saved session info to MongoDB for session {session_id}")
            return True
        
        return False
    
    except Exception as e:
        print(f"ERROR saving session info to MongoDB: {e}")
        return False

def save_complete_session_to_mongodb(session_id, session_dir):
    """
    Save all session data (transcripts, analysis, info) to MongoDB
    
    Args:
        session_id: The session ID
        session_dir: Path to session directory
    """
    print(f"\n💾 Saving session {session_id} to MongoDB...")
    
    # Save session info first
    save_session_info_to_mongodb(session_id, session_dir)
    
    # Save transcripts if they exist
    save_transcripts_to_mongodb(session_id, session_dir)
    
    # Save analysis if it exists
    save_analysis_to_mongodb(session_id, session_dir)
    
    print(f"✅ Complete session data saved to MongoDB")

if __name__ == "__main__":
    # Test the helper
    print("Testing MongoDB helper...")
    db = get_mongodb_connection()
    if db:
        print(f"✅ Connected to MongoDB: {db.name}")
        print(f"   Collections: {db.list_collection_names()}")
    else:
        print("❌ MongoDB connection failed")

