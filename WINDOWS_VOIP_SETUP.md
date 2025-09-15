# 🖥️ Windows VoIP Voice Agent Setup Guide

## Complete Setup for Gate VoIP System Integration

This guide will set up the Gemini AI voice agent to work directly on Windows with your Gate VoIP system without requiring Linux or WSL.

### 📋 Prerequisites

- **Windows 10/11** with Python 3.8+
- **Gate VoIP System** with SIM card (+359898995151)
- **Google AI Studio API Key**
- **Network connectivity** to Gate VoIP system

### 🚀 Quick Setup (5 minutes)

#### 1. Install Python Dependencies

```powershell
# Install Windows-specific requirements
pip install -r requirements_windows.txt
```

#### 2. Run Windows Setup

```powershell
# Automated setup for Windows
python setup_windows.py
```

This will:

- ✅ Detect your Windows IP address
- ✅ Create `.env` configuration file
- ✅ Test audio devices
- ✅ Verify Gate VoIP connectivity
- ✅ Create startup scripts

#### 3. Configure API Key

Edit the `.env` file and add your Google API key:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

#### 4. Start the Voice Agent

```powershell
# Start the Windows voice agent
python windows_voice_agent.py

# Or use the batch file
start_voice_agent.bat
```

#### 5. Test Everything

```powershell
# Run comprehensive tests
python test_windows_voip.py

# Quick tests only
python test_windows_voip.py --quick
```

### 📞 Your Gate VoIP Configuration

Based on your screenshots, your system is configured as:

- **Phone Number**: +359898995151 (SIM card)
- **Gate VoIP IP**: 192.168.50.127
- **Your Windows IP**: 192.168.50.158
- **SIP Port**: 5060
- **Voice Agent**: Configured as "voice-agent" operator

### 🔧 Network Configuration

The system is automatically configured for your network:

```json
{
  "host": "192.168.50.127", // Gate VoIP IP
  "username": "voice-agent", // As shown in your VoIP panel
  "password": "5060", // SIP port
  "sip_port": 5060,
  "phone_number": "+359898995151", // Your SIM number
  "local_ip": "192.168.50.158" // Your Windows IP
}
```

### 🎯 How It Works

1. **Incoming Calls**:

   - Gate VoIP routes calls to "voice-agent" (192.168.50.158:5060)
   - Windows voice agent answers via SIP protocol
   - Audio is processed through Google Gemini AI
   - Response is sent back through Gate VoIP to caller

2. **Outgoing Calls**:
   - API call to `/api/make_call` with phone number
   - SIP INVITE sent to Gate VoIP system
   - Gate VoIP routes through SIM card (+359898995151)
   - Gemini AI handles the conversation

### 📡 API Endpoints

Once running, these endpoints are available:

```http
# Health check
GET http://localhost:8000/health

# View current configuration
GET http://localhost:8000/api/config

# Active call sessions
GET http://localhost:8000/api/sessions

# Make outbound call
POST http://localhost:8000/api/make_call
Content-Type: application/json
{
  "phone_number": "+1234567890"
}
```

### 🧪 Testing Your Setup

#### Test 1: System Health

```powershell
curl http://localhost:8000/health
```

Should return:

```json
{
  "status": "healthy",
  "phone_number": "+359898995151",
  "local_ip": "192.168.50.158",
  "gate_voip": "192.168.50.127"
}
```

#### Test 2: Make Test Call

```powershell
curl -X POST http://localhost:8000/api/make_call ^
  -H "Content-Type: application/json" ^
  -d "{\"phone_number\": \"+359898995151\"}"
```

#### Test 3: Receive Call

Call your SIM number (+359898995151) from any phone:

- Gate VoIP should route to your Windows system
- Gemini AI should answer and have a conversation

### 🔧 Troubleshooting

#### Can't Connect to Gate VoIP

```powershell
# Test network connectivity
ping 192.168.50.127

# Check if port is accessible
telnet 192.168.50.127 5060
```

#### Voice Agent Won't Start

```powershell
# Check Python environment
python test_windows_voip.py --quick

# Check logs
python windows_voice_agent.py --reload
```

#### No Audio or Poor Quality

1. Run audio device test:

```powershell
python -c "
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f'Input {i}: {info[\"name\"]}')
"
```

2. Update audio device in `.env`:

```env
AUDIO_INPUT_DEVICE_INDEX=1
AUDIO_OUTPUT_DEVICE_INDEX=2
```

#### Firewall Issues

```powershell
# Allow voice agent through Windows Firewall (as Administrator)
netsh advfirewall firewall add rule name="VoIP Voice Agent" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="SIP Protocol" dir=in action=allow protocol=UDP localport=5060
```

### 🎉 What You Can Do Now

✅ **Answer incoming calls** on +359898995151 with AI  
✅ **Make outbound calls** via API  
✅ **Full conversation handling** with Google Gemini  
✅ **Call logging and recording**  
✅ **DTMF (keypad) support**  
✅ **Session management**

### 🔄 Daily Operations

#### Start Voice Agent

```powershell
# Method 1: Direct
python windows_voice_agent.py

# Method 2: Batch file
start_voice_agent.bat

# Method 3: Background service
python windows_voice_agent.py --host 0.0.0.0 --port 8000
```

#### Monitor Calls

```powershell
# Check active sessions
curl http://localhost:8000/api/sessions

# View logs
# Logs appear in console where you started the agent
```

#### Make Outbound Calls

```powershell
# Via curl
curl -X POST http://localhost:8000/api/make_call ^
  -H "Content-Type: application/json" ^
  -d "{\"phone_number\": \"DESTINATION_NUMBER\"}"

# Via browser (visit http://localhost:8000/docs for interactive API)
```

### 📊 Performance Expectations

With this Windows setup you should achieve:

- **Call Answer Time**: < 3 seconds
- **Audio Latency**: 500-1000ms (typical for VoIP + AI)
- **Concurrent Calls**: 5-10 (depends on hardware)
- **Uptime**: 99%+ (restart if needed)

### 🚀 Production Deployment

For production use:

1. **Windows Service**: Convert to Windows service for auto-start
2. **Monitoring**: Set up health check monitoring
3. **Logging**: Configure structured logging
4. **Backup**: Regular backup of call logs and sessions
5. **Security**: Configure proper authentication if exposing APIs

### 🆘 Support

If you encounter issues:

1. **Run full tests**: `python test_windows_voip.py`
2. **Check logs**: Look at console output when running voice agent
3. **Verify network**: Ensure Gate VoIP system is accessible
4. **Test API key**: Make sure Google AI key is valid

### 📝 Configuration Files Summary

- **`asterisk_config.json`**: VoIP system configuration
- **`.env`**: Environment variables and API keys
- **`windows_voice_agent.py`**: Main voice agent application
- **`setup_windows.py`**: Automated setup script
- **`test_windows_voip.py`**: Comprehensive test suite
- **`start_voice_agent.bat`**: Windows startup script

---

**🎯 That's it! Your Windows system is now a complete AI telephone system that can handle calls through your Gate VoIP and SIM card setup!**
