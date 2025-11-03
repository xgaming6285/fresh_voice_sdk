"""
Script to assign Google API keys to existing users in the database
Run this once to assign GOOGLE_API_KEY_2 through GOOGLE_API_KEY_10 to existing admins and agents
"""

import os
from dotenv import load_dotenv
from crm_database_mongodb import get_session, User, UserRole, UserManager

# Load environment variables
load_dotenv()

def assign_api_keys_to_existing_users():
    """Assign API keys to all existing admins and agents"""
    session = get_session()
    
    try:
        user_manager = UserManager(session)
        
        # Get available API keys from environment
        available_keys = user_manager.get_available_api_keys()
        
        print(f"✅ Found {len(available_keys)} API keys in environment")
        for i, key in enumerate(available_keys):
            print(f"   {i+1}. {key[:20]}...{key[-4:]}")
        
        # Get all admins and agents without API keys
        users_without_keys = session.db.users.find({
            "$or": [
                {"role": UserRole.ADMIN.value},
                {"role": UserRole.AGENT.value}
            ],
            "$or": [
                {"google_api_key": None},
                {"google_api_key": {"$exists": False}}
            ]
        })
        
        users_list = list(users_without_keys)
        
        if not users_list:
            print("\n✅ All users already have API keys assigned!")
            return
        
        print(f"\n📋 Found {len(users_list)} users without API keys:")
        for user_data in users_list:
            user = User.from_dict(user_data)
            print(f"   - {user.username} ({user.role.value if hasattr(user.role, 'value') else user.role})")
        
        # Assign API keys
        print("\n🔑 Assigning API keys...")
        assigned_count = 0
        
        for user_data in users_list:
            user = User.from_dict(user_data)
            
            # Skip superadmins
            if user.role == UserRole.SUPERADMIN:
                print(f"   ⏭️  Skipping superadmin: {user.username}")
                continue
            
            # Assign API key
            assigned_key = user_manager.assign_api_key(user.id)
            
            if assigned_key:
                print(f"   ✅ Assigned key to {user.username}: {assigned_key[:20]}...{assigned_key[-4:]}")
                assigned_count += 1
            else:
                print(f"   ❌ Failed to assign key to {user.username}")
        
        print(f"\n🎉 Successfully assigned API keys to {assigned_count} users!")
        
        # Show summary
        print("\n📊 Summary:")
        print(f"   Total users: {len(users_list)}")
        print(f"   Assigned: {assigned_count}")
        print(f"   Available keys: {len(available_keys)}")
        
        # Show key distribution
        print("\n📈 API Key Distribution:")
        for key in available_keys:
            count = session.db.users.count_documents({"google_api_key": key})
            print(f"   {key[:20]}...{key[-4:]}: {count} user(s)")
        
    except Exception as e:
        print(f"❌ Error assigning API keys: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Assigning Google API Keys to Users")
    print("=" * 60)
    assign_api_keys_to_existing_users()

