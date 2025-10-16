# -*- coding: utf-8 -*-
"""
Migrate Session Data to MongoDB
Imports all transcripts, analysis, and metadata from sessions/ folder to MongoDB
"""

import os
import json
from pathlib import Path
from datetime import datetime
from pymongo import MongoClient

def migrate_session_data():
    """Migrate all session data from local files to MongoDB"""
    
    print("=" * 60)
    print("Session Data Migration to MongoDB")
    print("=" * 60)
    
    # Connect to MongoDB
    try:
        client = MongoClient('localhost', 27017)
        db = client['voice_agent_crm']
        print(f"\nConnected to MongoDB: {db.name}")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        return
    
    # Get sessions directory
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        print(f"ERROR: Sessions directory not found: {sessions_dir}")
        return
    
    # Get all session folders
    session_folders = [d for d in sessions_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(session_folders)} session folders")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for session_folder in session_folders:
        session_id = session_folder.name
        
        try:
            # Check if session exists in MongoDB
            session_doc = db.call_sessions.find_one({"session_id": session_id})
            
            if not session_doc:
                print(f"\nSkipping {session_id}: Not found in MongoDB")
                skipped += 1
                continue
            
            print(f"\nProcessing session: {session_id}")
            
            # Prepare update data
            update_data = {}
            
            # Read analysis_result.json
            analysis_file = session_folder / "analysis_result.json"
            if analysis_file.exists():
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    update_data['analysis'] = analysis
                    print(f"  - Loaded analysis: {analysis.get('status', 'N/A')}")
                except Exception as e:
                    print(f"  - Error reading analysis: {e}")
            
            # Read session_info.json
            info_file = session_folder / "session_info.json"
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        session_info = json.load(f)
                    update_data['session_info'] = session_info
                    print(f"  - Loaded session info")
                except Exception as e:
                    print(f"  - Error reading session info: {e}")
            
            # Read transcripts
            transcripts = {}
            
            for transcript_type in ['incoming', 'outgoing', 'mixed']:
                # Find transcript file (could have different timestamps)
                transcript_files = list(session_folder.glob(f"{transcript_type}_*.txt"))
                
                if transcript_files:
                    transcript_file = transcript_files[0]
                    try:
                        with open(transcript_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Parse transcript file format
                        lines = content.split('\n')
                        transcript_text = '\n'.join(lines[6:]).strip() if len(lines) > 6 else content
                        
                        transcripts[transcript_type] = {
                            'text': transcript_text,
                            'file': transcript_file.name,
                            'length': len(transcript_text)
                        }
                        
                        print(f"  - Loaded {transcript_type} transcript ({len(transcript_text)} chars)")
                    except Exception as e:
                        print(f"  - Error reading {transcript_type} transcript: {e}")
            
            if transcripts:
                update_data['transcripts'] = transcripts
            
            # List audio files (store paths, not content)
            audio_files = {}
            for audio_type in ['incoming', 'outgoing', 'mixed']:
                audio_file_pattern = list(session_folder.glob(f"{audio_type}_*.wav"))
                if audio_file_pattern:
                    audio_files[audio_type] = str(audio_file_pattern[0].relative_to(sessions_dir.parent))
            
            if audio_files:
                update_data['audio_files'] = audio_files
            
            # Update MongoDB document
            if update_data:
                update_data['migrated_at'] = datetime.utcnow()
                
                result = db.call_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": update_data}
                )
                
                if result.modified_count > 0:
                    print(f"  SUCCESS: Updated MongoDB document")
                    migrated += 1
                else:
                    print(f"  INFO: No changes needed")
                    migrated += 1
            else:
                print(f"  WARNING: No data to migrate")
                skipped += 1
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total sessions: {len(session_folders)}")
    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print("=" * 60)
    
    # Verify one session
    if migrated > 0:
        print("\nVerifying sample session...")
        sample = db.call_sessions.find_one({"analysis": {"$exists": True}})
        if sample:
            print(f"\nSample session: {sample['session_id']}")
            print(f"  - Has analysis: {bool(sample.get('analysis'))}")
            print(f"  - Has transcripts: {bool(sample.get('transcripts'))}")
            print(f"  - Has audio files: {bool(sample.get('audio_files'))}")
            if sample.get('analysis'):
                print(f"  - Analysis summary: {sample['analysis'].get('summary', 'N/A')[:100]}...")

if __name__ == "__main__":
    migrate_session_data()

