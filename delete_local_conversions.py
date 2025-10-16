# -*- coding: utf-8 -*-
"""
Delete Local Conversion Files
Removes analysis_result.json files with 'interested' status
"""

import os
import json
from pathlib import Path

def delete_local_conversions():
    """Delete local analysis files with interested status"""
    try:
        print("=" * 70)
        print("Delete Local Conversion Files")
        print("=" * 70)
        
        sessions_dir = Path("sessions")
        
        if not sessions_dir.exists():
            print("\nNo sessions directory found")
            return
        
        deleted_count = 0
        kept_count = 0
        
        print("\nScanning local session files...")
        
        for session_folder in sessions_dir.iterdir():
            if not session_folder.is_dir():
                continue
            
            analysis_file = session_folder / "analysis_result.json"
            
            if analysis_file.exists():
                try:
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    
                    if analysis.get('status') == 'interested':
                        # Delete the analysis file
                        analysis_file.unlink()
                        print(f"[DELETED] {session_folder.name}/analysis_result.json")
                        deleted_count += 1
                    else:
                        kept_count += 1
                        
                except Exception as e:
                    print(f"[ERROR] {session_folder.name}: {e}")
        
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print("=" * 70)
        print(f"Deleted: {deleted_count} analysis files with 'interested' status")
        print(f"Kept: {kept_count} analysis files with other statuses")
        print("=" * 70)
        
        if deleted_count > 0:
            print("\nConversion records removed from local files!")
            print("Refresh your Conversions page to see the changes.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    delete_local_conversions()

