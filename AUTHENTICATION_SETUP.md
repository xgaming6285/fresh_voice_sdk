# CRM Authentication System - Setup Complete

## Overview

The CRM system now has a complete authentication and authorization system with role-based access control.

## Key Features Implemented

### 1. **Database Models** (crm_database.py)

- **User Model**: Added with username, email, password hashing, roles (admin/agent), and hierarchical relationships
- **UserRole Enum**: ADMIN and AGENT roles
- **Ownership**: All leads and campaigns now have `owner_id` field
- **LeadType Removed**: The lead_type enum has been completely removed and replaced with ownership tracking
- **UserManager**: New manager class for user operations

### 2. **Backend Authentication** (crm_auth.py)

- `/api/auth/register` - Register new admin users
- `/api/auth/login` - Login with username/password
- `/api/auth/me` - Get current user info
- `/api/auth/verify-token` - Verify JWT token
- JWT token-based authentication with Bearer tokens
- Password hashing using bcrypt

### 3. **User Management** (crm_user_management.py)

- `/api/users/agents` - Get all agents created by current admin (admin only)
- `/api/users/agents` (POST) - Create new agent (admin only)
- `/api/users/agents/{id}` (GET) - Get specific agent details
- `/api/users/agents/{id}` (PUT) - Update agent information
- `/api/users/agents/{id}` (DELETE) - Delete agent (cascades to their leads/campaigns)

### 4. **Updated CRM API** (crm_api.py)

- **All endpoints now require authentication**
- **Data filtering by ownership**:
  - Agents can only see their own leads and campaigns
  - Admins can see their own data AND all their agents' data
- **LeadResponse and CampaignResponse** now include:
  - `owner_id` - ID of the user who created the record
  - `owner_name` - Full name of the user who created the record
- **Removed** all lead_type references and replaced with agent ownership

### 5. **Frontend Authentication**

#### AuthContext (src/contexts/AuthContext.js)

- Manages authentication state globally
- Provides login, register, logout functions
- Automatic token management and user loading
- Role checking helpers (isAdmin(), isAgent())

#### New Pages

- **Login Page** (`src/pages/Login.js`) - User login interface
- **Register Page** (`src/pages/Register.js`) - Admin registration
- **Agents Page** (`src/pages/Agents.js`) - Admin interface to manage agents

#### Updated Components

- **App.js** - Protected routes, public routes, AuthProvider integration
- **Layout.js** - User menu, logout, role badge, admin-only Agents menu item
- **Leads.js** - Removed lead_type, now shows "Added By" (owner_name)
- **API Service** (`src/services/api.js`) - Token management and authentication APIs

## Data Isolation

### Agent View

- Can only see and manage leads/campaigns they created
- Cannot see other agents or their data

### Admin View

- Can see and manage their own leads/campaigns
- Can see and manage ALL leads/campaigns created by their agents
- Can manage their agents (create, edit, delete)
- Cannot see other admins or their agents

## Database Migration

### Breaking Changes

The database schema has changed significantly:

1. **User table added** with authentication fields
2. **Leads table**:
   - Removed `lead_type` column (Enum)
   - Added `owner_id` column (FK to users)
3. **Campaigns table**:
   - Added `owner_id` column (FK to users)

### Migration Steps Required

1. Backup existing database: `voice_agent_crm.db`
2. The new schema will be created automatically on first run
3. **Data migration script needed** if you have existing leads/campaigns

## Setup Instructions

### Backend

1. Install new dependencies:

   ```bash
   pip install pyjwt passlib[bcrypt]
   ```

2. Set JWT secret key (optional, defaults to a dev key):

   ```bash
   export JWT_SECRET_KEY="your-secret-key-here"
   ```

3. Run the voice agent server:
   ```bash
   python windows_voice_agent.py
   ```

### Frontend

No additional dependencies needed - all using existing packages.

1. Start the React app:
   ```bash
   cd crm-frontend
   npm start
   ```

## First Time Use

1. Navigate to `http://localhost:3000`
2. You'll be redirected to the login page
3. Click "Register here" to create your first admin account
4. After registration, you'll be logged in automatically
5. As an admin, go to "Agents" menu to create agents
6. Each agent will have their own login credentials

## API Endpoints Summary

### Authentication

- `POST /api/auth/register` - Register admin
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user
- `POST /api/auth/verify-token` - Verify token

### User Management (Admin Only)

- `GET /api/users/agents` - List agents
- `POST /api/users/agents` - Create agent
- `GET /api/users/agents/{id}` - Get agent
- `PUT /api/users/agents/{id}` - Update agent
- `DELETE /api/users/agents/{id}` - Delete agent

### CRM (All Authenticated Users)

All existing CRM endpoints now:

- Require authentication (Bearer token)
- Filter data by ownership
- Return owner information in responses

## Security Features

1. **Password Hashing**: Using bcrypt for secure password storage
2. **JWT Tokens**: Stateless authentication with 24-hour expiration
3. **Role-Based Access**: Admins and agents have different permissions
4. **Data Isolation**: Users can only access their own data (or their team's data for admins)
5. **Protected Routes**: All CRM pages require authentication

## CSV Import Format

The CSV import format has been updated to remove `lead_type`:

```csv
first_name,last_name,email,phone,country,prefix,gender,address
John,Doe,john.doe@example.com,555-1234,USA,+1,male,123 Main St
Jane,Smith,jane.smith@example.com,555-5678,UK,+44,female,456 High St
```

## Notes

- All leads imported by an agent are automatically assigned to that agent
- When an admin deletes an agent, all their leads and campaigns are also deleted (CASCADE)
- Admins and their agents form isolated teams - admins cannot see other admins' teams
- The system supports multiple admin teams working independently

## Troubleshooting

### Token Issues

- Tokens are stored in localStorage under the key "token"
- Clear browser localStorage if experiencing auth issues

### Database Issues

- If you get database errors, the schema might need recreation
- Backup and delete `voice_agent_crm.db` to start fresh

### API Issues

- Check browser console for API errors
- Verify the backend is running on http://localhost:8000
- Check that JWT_SECRET_KEY matches between registration and login sessions
