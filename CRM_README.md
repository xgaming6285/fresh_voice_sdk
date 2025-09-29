# Voice Agent CRM System

A comprehensive Customer Relationship Management (CRM) system for the Voice Agent, enabling automated outbound calling campaigns with lead management, campaign orchestration, and call session review capabilities.

## Features

### 1. **Dashboard**

- Real-time system status monitoring
- Call statistics and success rates
- Recent call sessions overview
- Visual charts for call trends and outcomes

### 2. **Leads Management**

- **Lead Types**: FTD (First Time Deposit), Cold, Filler, Live
- **Import/Export**: Bulk CSV import with template download
- **Lead Fields**:
  - Personal: First name, Last name, Gender
  - Contact: Email, Phone (with country code separation)
  - Location: Country, Address
  - Tracking: Call count, Last called timestamp
- **Quick Actions**: Direct calling, editing, deletion
- **Filtering**: By country, lead type, call history

### 3. **Campaign Management**

- **Campaign Creation**: Name, description, bot configuration
- **Lead Selection**:
  - Manual selection from lead list
  - Filter-based selection (country, type, call history)
- **Campaign Controls**: Start, Pause, Resume, Stop
- **Real-time Monitoring**:
  - Progress tracking
  - Live call statistics
  - Success/failure rates
- **Configuration Options**:
  - Concurrent calls limit
  - Wait time between calls
  - Retry attempts
  - Greeting delay

### 4. **Session Review**

- **Audio Playback**: Listen to recorded calls
  - Incoming audio (caller)
  - Outgoing audio (AI agent)
  - Mixed audio (full conversation)
- **Transcripts**:
  - Automatic transcription in multiple languages
  - Language detection based on caller location
  - Confidence scores
  - Re-transcription capability
- **Session Analytics**:
  - Call duration and talk time
  - AI-generated insights
  - Sentiment analysis
  - Follow-up recommendations

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- SQLite (included with Python)
- Windows/Linux/macOS

### Backend Setup

1. **Install Python dependencies**:

```bash
pip install sqlalchemy fastapi uvicorn
```

2. **Initialize the database**:

```bash
python crm_database.py
```

3. **Update windows_voice_agent.py** (already done if using the provided files)

### Frontend Setup

1. **Navigate to frontend directory**:

```bash
cd crm-frontend
```

2. **Install dependencies**:

```bash
npm install
```

3. **Start the development server**:

```bash
npm start
```

## Usage

### Starting the System

1. **Start the Voice Agent with CRM**:

```bash
python windows_voice_agent.py
```

2. **Start the CRM Frontend** (in a separate terminal):

```bash
cd crm-frontend
npm start
```

3. **Access the CRM**:
   Open your browser and navigate to `http://localhost:3000`

### Workflow

#### 1. Import Leads

- Go to **Leads** page
- Click **Download Template** to get CSV format
- Fill in lead data in the CSV file
- Click **Import CSV** and upload your file

#### 2. Create a Campaign

- Go to **Campaigns** page
- Click **Create Campaign**
- Enter campaign name and configure settings
- Save the campaign

#### 3. Add Leads to Campaign

- Open your campaign
- Click **Add Selected Leads** or **Add Filtered Leads**
- Select leads based on your criteria
- Confirm addition

#### 4. Start Campaign

- Ensure leads are added to the campaign
- Click **Start Campaign**
- Monitor progress in real-time
- Pause/Resume as needed

#### 5. Review Sessions

- Go to **Call Sessions** page
- Click **Review** on any session
- Listen to recordings
- View transcripts
- Analyze AI insights

## API Endpoints

### CRM API (`/api/crm/`)

#### Leads

- `GET /api/crm/leads` - List leads with pagination
- `POST /api/crm/leads` - Create new lead
- `GET /api/crm/leads/{id}` - Get lead details
- `PUT /api/crm/leads/{id}` - Update lead
- `DELETE /api/crm/leads/{id}` - Delete lead
- `POST /api/crm/leads/import` - Import leads from CSV

#### Campaigns

- `GET /api/crm/campaigns` - List campaigns
- `POST /api/crm/campaigns` - Create campaign
- `GET /api/crm/campaigns/{id}` - Get campaign details
- `PUT /api/crm/campaigns/{id}` - Update campaign
- `POST /api/crm/campaigns/{id}/leads` - Add selected leads
- `POST /api/crm/campaigns/{id}/leads/filter` - Add filtered leads
- `GET /api/crm/campaigns/{id}/leads` - Get campaign leads
- `POST /api/crm/campaigns/{id}/start` - Start campaign
- `POST /api/crm/campaigns/{id}/pause` - Pause campaign
- `POST /api/crm/campaigns/{id}/stop` - Stop campaign

#### Sessions

- `GET /api/crm/sessions` - List call sessions
- `GET /api/crm/sessions/{session_id}` - Get session details

### Voice Agent API (existing)

- `GET /health` - System health check
- `GET /api/sessions` - Active voice sessions
- `GET /api/recordings` - Call recordings
- `GET /api/transcripts/{session_id}` - Session transcripts
- `POST /api/make_call` - Make outbound call

## Database Schema

### Tables

1. **leads**

   - Lead information and contact details
   - Call history tracking

2. **campaigns**

   - Campaign configuration and status
   - Call statistics

3. **campaign_leads**

   - Many-to-many relationship
   - Individual lead status in campaign

4. **call_sessions**
   - Detailed call records
   - Links to voice agent sessions

## Configuration

### Bot Configuration (per campaign)

```json
{
  "greeting_delay": 1,
  "max_call_duration": 300,
  "voice_speed": 1.0
}
```

### Dialing Configuration

```json
{
  "concurrent_calls": 1,
  "wait_between_calls": 5,
  "retry_attempts": 2
}
```

## Best Practices

1. **Lead Management**

   - Keep lead data up-to-date
   - Use meaningful lead types
   - Add notes for special cases

2. **Campaign Planning**

   - Start with small test campaigns
   - Monitor and adjust configurations
   - Use appropriate concurrent call limits

3. **Session Review**
   - Review failed calls for improvements
   - Use transcripts for quality assurance
   - Act on AI-generated insights

## Troubleshooting

### Common Issues

1. **Transcription not working**

   - Check if Whisper is installed
   - Verify audio files exist
   - Check language detection

2. **Campaign not starting**

   - Ensure leads are added
   - Check voice agent status
   - Verify SIP registration

3. **Frontend connection issues**
   - Check backend is running on port 8000
   - Verify CORS settings
   - Check API endpoints

## Security Considerations

- Implement authentication before production use
- Secure API endpoints
- Encrypt sensitive lead data
- Regular backups of SQLite database
- Audit trail for campaign actions

## Future Enhancements

- Real-time call monitoring
- Advanced analytics dashboard
- Email/SMS integration
- Multi-user support with roles
- Scheduled campaigns
- A/B testing for scripts
- Integration with external CRMs
- Webhook notifications
