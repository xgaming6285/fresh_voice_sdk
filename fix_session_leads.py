#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to fix CallSession records that are missing lead_id
by matching phone numbers with leads
"""

from crm_database import get_session, CallSession, Lead
from sqlalchemy import or_

def fix_session_leads():
    """Update CallSessions with missing lead_id by matching phone numbers"""
    db = get_session()
    
    try:
        # Find all sessions without lead_id
        sessions_without_leads = db.query(CallSession).filter(
            CallSession.lead_id.is_(None)
        ).all()
        
        print(f"Found {len(sessions_without_leads)} sessions without lead_id")
        
        fixed_count = 0
        for session in sessions_without_leads:
            # Try to match by called_number (for outbound calls) or caller_id (for inbound calls)
            phone_to_search = session.called_number or session.caller_id
            
            if not phone_to_search:
                continue
            
            # Remove common formatting
            phone_clean = phone_to_search.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
            
            # Try to find matching lead by various patterns
            # Since leads store phone without country code but have country_code separately,
            # we need to get all leads and check if their full_phone matches
            all_leads = db.query(Lead).all()
            lead = None
            
            for potential_lead in all_leads:
                # Construct full phone from lead
                lead_full = (potential_lead.country_code or '') + potential_lead.phone
                lead_full_clean = lead_full.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
                
                # Check if they match
                if phone_clean == lead_full_clean or phone_clean.endswith(lead_full_clean) or lead_full_clean.endswith(phone_clean):
                    lead = potential_lead
                    break
                
                # Also check last 10 digits (common in some regions)
                if len(phone_clean) >= 10 and len(lead_full_clean) >= 10:
                    if phone_clean[-10:] == lead_full_clean[-10:]:
                        lead = potential_lead
                        break
            
            if lead:
                session.lead_id = lead.id
                fixed_count += 1
                print(f"  [OK] Matched session {session.session_id} to lead {lead.id} ({lead.full_name}) - {lead.country}")
        
        # Commit all changes
        db.commit()
        print(f"\n[SUCCESS] Fixed {fixed_count} sessions with lead information")
        print(f"[INFO] Could not match {len(sessions_without_leads) - fixed_count} sessions")
        
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Fixing CallSession records with missing lead_id...")
    fix_session_leads()
    print("Done!")

