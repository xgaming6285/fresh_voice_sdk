"""
Script to initialize the superadmin user
Run this once to create the superadmin account
"""

from crm_database import User, UserRole, get_session
from sqlalchemy.orm import Session

def create_superadmin():
    """Create or update the superadmin user"""
    session = get_session()
    
    try:
        # Check if superadmin already exists
        superadmin = session.query(User).filter_by(username="superadmin").first()
        
        if superadmin:
            print("✅ Superadmin already exists!")
            print(f"   Username: {superadmin.username}")
            print(f"   Email: {superadmin.email}")
            print(f"   Role: {superadmin.role.value}")
            return
        
        # Create new superadmin
        superadmin = User(
            username="superadmin",
            email="superadmin@system.local",
            role=UserRole.SUPERADMIN,
            first_name="Super",
            last_name="Admin",
            organization="System",
            is_active=True
        )
        superadmin.hashed_password = User.hash_password("123123")
        
        session.add(superadmin)
        session.commit()
        
        print("✅ Superadmin created successfully!")
        print(f"   Username: superadmin")
        print(f"   Password: 123123")
        print(f"   Email: {superadmin.email}")
        print("⚠️  Please keep these credentials secure!")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error creating superadmin: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    create_superadmin()

