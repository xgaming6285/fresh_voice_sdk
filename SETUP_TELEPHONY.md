# Voice Agent Telephony Integration Setup Guide

This guide will help you integrate your Google Gemini voice agent with your SIM gateway through Asterisk for phone call functionality.

## Overview

The integration consists of several components:

1. **AGI Voice Server** (`agi_voice_server.py`) - FastAPI server that bridges Asterisk with Google Gemini
2. **AGI Script** (`voice_agent_agi.py`) - Python script called by Asterisk for each call
3. **Configuration files** - Asterisk dialplan and JSON configuration
4. **Your existing voice agent** - Enhanced with telephony capabilities

## Prerequisites

### System Requirements

- Linux server (Ubuntu 18.04+ or CentOS 7+)
- Asterisk 16+ installed and configured
- Python 3.8+
- Network access to your SIM gateway
- Google AI API key

### SIM Gateway Information Needed

From your VoIP panel screenshots, you'll need:

- SIM gateway IP address
- Admin username/password
- SIP configuration details
- DID numbers (if any)

## Step 1: Install Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv asterisk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Configure Asterisk SIP Connection

### 2.1 Configure SIP Trunk to Your SIM Gateway

Edit `/etc/asterisk/sip.conf` (or `pjsip.conf` for newer Asterisk):

```ini
; Add to sip.conf
[sim-gateway]
type=peer
host=YOUR_SIM_GATEWAY_IP
port=5060
username=YOUR_SIP_USERNAME
secret=YOUR_SIP_PASSWORD
context=sip-trunk-incoming
dtmfmode=rfc2833
canreinvite=no
nat=force_rport,comedia
qualify=yes
disallow=all
allow=ulaw,alaw,gsm

; If you have specific DIDs/phone numbers
[your-phone-number]
type=peer
host=YOUR_SIM_GATEWAY_IP
username=YOUR_USERNAME
secret=YOUR_PASSWORD
context=sip-trunk-incoming
```

### 2.2 Configure Dialplan

Copy the provided dialplan configuration to `/etc/asterisk/extensions.conf`:

```bash
sudo cp asterisk_dialplan.conf /etc/asterisk/extensions_voice_agent.conf

# Include in main extensions.conf
echo "#include extensions_voice_agent.conf" | sudo tee -a /etc/asterisk/extensions.conf
```

### 2.3 Install AGI Script

```bash
# Copy AGI script to Asterisk AGI directory
sudo cp voice_agent_agi.py /var/lib/asterisk/agi-bin/
sudo chmod +x /var/lib/asterisk/agi-bin/voice_agent_agi.py

# Ensure Python path is correct in the script
sudo sed -i '1s|.*|#!/usr/bin/env python3|' /var/lib/asterisk/agi-bin/voice_agent_agi.py
```

## Step 3: Configure Your Voice Agent

### 3.1 Update Configuration

Edit `asterisk_config.json` with your SIM gateway details:

```json
{
  "host": "192.168.1.100", // Your SIM gateway IP
  "username": "admin",
  "password": "your_password_here",
  "sip_port": 5060,
  "ari_port": 8088,
  "ari_username": "asterisk",
  "ari_password": "asterisk",
  "context": "voice-agent"
}
```

### 3.2 Set Environment Variables

Create `.env` file if it doesn't exist:

```bash
# Google AI API Key
GOOGLE_API_KEY=your_google_ai_api_key_here

# MongoDB (optional)
MONGODB_HOST=localhost
MONGODB_PORT=27017

# Voice Agent Settings
VOICE_AGENT_HOST=0.0.0.0
VOICE_AGENT_PORT=8000
```

## Step 4: Start Services

### 4.1 Start the AGI Voice Server

```bash
# In your project directory
source venv/bin/activate
python agi_voice_server.py --host 0.0.0.0 --port 8000
```

### 4.2 Restart Asterisk

```bash
sudo systemctl restart asterisk

# Or reload configuration
sudo asterisk -rx "core reload"
sudo asterisk -rx "sip reload"
```

## Step 5: Configure SIM Gateway Routing

Based on your VoIP panel interface:

### 5.1 Access Your VoIP Panel

1. Log into your VoIP panel using the credentials your boss provided
2. Navigate to the telephony configuration section

### 5.2 Configure Routing

1. **Incoming Calls**: Set up routing rules to send incoming calls to your Asterisk server
2. **SIP Trunk**: Configure SIP trunk pointing to your Asterisk server IP
3. **Context**: Set the context to route calls to your voice agent

### 5.3 Example Routing Configuration

In your VoIP panel, look for:

- **Routing/Рутиране**: Configure call routing
- **SIP Trunk/VoIP оператори**: Add your Asterisk server as a SIP endpoint
- **Incoming Rules/Входящи правила**: Route calls to Asterisk

## Step 6: Testing

### 6.1 Test Asterisk Configuration

```bash
# Check SIP registration
sudo asterisk -rx "sip show peers"
sudo asterisk -rx "sip show registry"

# Test dialplan
sudo asterisk -rx "dialplan show voice-agent-context"
```

### 6.2 Test Voice Agent Server

```bash
# Health check
curl http://localhost:8000/health

# Check active sessions
curl http://localhost:8000/api/sessions
```

### 6.3 Make a Test Call

1. **Incoming Call**: Call one of your SIM card numbers
2. **Outgoing Call**: Use the API endpoint:

```bash
curl -X POST http://localhost:8000/api/make_call \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+1234567890", "caller_id": "VoiceAgent"}'
```

## Troubleshooting

### Common Issues

#### 1. Asterisk Can't Connect to SIM Gateway

- Check firewall settings
- Verify SIP credentials
- Check network connectivity: `telnet SIM_GATEWAY_IP 5060`

#### 2. AGI Script Errors

- Check logs: `sudo tail -f /var/log/asterisk/voice_agent_agi.log`
- Verify Python environment: `which python3`
- Check script permissions: `ls -la /var/lib/asterisk/agi-bin/`

#### 3. Voice Server Connection Issues

- Verify server is running: `ps aux | grep agi_voice_server`
- Check port availability: `netstat -tulpn | grep :8000`
- Review server logs

#### 4. Audio Quality Issues

- Check codec configuration (prefer ulaw/alaw for telephony)
- Verify sample rate conversion
- Monitor network latency

### Debug Commands

```bash
# Asterisk console with debug
sudo asterisk -cvvv

# Enable SIP debug
sudo asterisk -rx "sip set debug on"

# Enable AGI debug
sudo asterisk -rx "agi set debug on"

# Monitor calls in real-time
sudo asterisk -rx "core show channels"
```

## Advanced Configuration

### Call Recording

The system automatically logs conversations. To enable audio recording:

```json
{
  "voice_agent_settings": {
    "enable_call_recording": true
  }
}
```

### Multiple SIM Cards

Configure multiple SIM card endpoints in `sip.conf`:

```ini
[sim-card-1]
type=peer
host=SIM_GATEWAY_IP
username=sim1_user
secret=sim1_pass
context=sip-trunk-incoming

[sim-card-2]
type=peer
host=SIM_GATEWAY_IP
username=sim2_user
secret=sim2_pass
context=sip-trunk-incoming
```

### Load Balancing

For high call volumes, run multiple AGI voice server instances:

```bash
# Start multiple servers on different ports
python agi_voice_server.py --port 8000 &
python agi_voice_server.py --port 8001 &
python agi_voice_server.py --port 8002 &
```

### Security Considerations

1. **Firewall**: Limit SIP access to your SIM gateway IP
2. **Authentication**: Use strong SIP passwords
3. **Encryption**: Consider SIP over TLS (SIPS) if supported
4. **API Security**: Add authentication to AGI voice server endpoints

## Monitoring and Maintenance

### Log Locations

- Asterisk logs: `/var/log/asterisk/`
- AGI script logs: `/var/log/asterisk/voice_agent_agi.log`
- Voice server logs: Check console output
- Session data: `sessions/` directory

### Performance Monitoring

```bash
# Monitor active calls
sudo asterisk -rx "core show channels"

# Check system resources
htop
df -h
```

### Backup Important Files

```bash
# Configuration files
/etc/asterisk/sip.conf
/etc/asterisk/extensions.conf
/var/lib/asterisk/agi-bin/voice_agent_agi.py
asterisk_config.json

# Session data
sessions/
```

## Next Steps

Once basic integration is working:

1. **Enhance conversation handling** - Add more sophisticated dialog management
2. **Implement transfer capabilities** - Route specific requests to human operators
3. **Add call analytics** - Track call success rates, conversation topics
4. **Scale the system** - Add load balancing and redundancy
5. **Integrate with CRM** - Connect caller information with customer databases

## Support

For issues with this integration:

1. Check logs in `/var/log/asterisk/` and console output
2. Verify all services are running
3. Test each component individually
4. Review the configuration files for typos or incorrect values

The voice agent should now be able to receive and make phone calls through your SIM gateway, providing AI-powered voice assistance to callers!
