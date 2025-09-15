# 🎯 Windows VoIP Voice Agent - Quick Start

## Your Gemini AI + Gate VoIP Integration

This is a **Windows-native** solution that connects your **Gate VoIP system** (+359898995151) with **Google Gemini AI** for intelligent phone call handling.

### ⚡ 5-Minute Setup

```powershell
# 1. Install dependencies
pip install -r requirements_windows.txt

# 2. Run setup
python setup_windows.py

# 3. Add your Google API key to .env file
notepad .env

# 4. Start the voice agent
python windows_voice_agent.py
```

### 📞 Your System Configuration

From your Gate VoIP screenshots:

- **📱 Phone Number**: +359898995151 (your SIM card)
- **🖥️ Windows IP**: 192.168.50.158
- **📡 Gate VoIP IP**: 192.168.50.127
- **🔌 SIP Port**: 5060
- **👤 Voice Operator**: "voice-agent"

### ✨ What It Does

🔄 **Incoming Calls → AI Assistant**  
When someone calls +359898995151:

1. Gate VoIP routes call to your Windows PC
2. Gemini AI answers and has natural conversation
3. Call logs are saved automatically

🚀 **Outgoing Calls via API**

```powershell
curl -X POST http://localhost:8000/api/make_call ^
  -H "Content-Type: application/json" ^
  -d "{\"phone_number\": \"+1234567890\"}"
```

### 🧪 Test Your Setup

```powershell
# Run all tests
python test_windows_voip.py

# Quick health check
curl http://localhost:8000/health
```

### 🎮 Daily Usage

**Start Voice Agent:**

```powershell
python windows_voice_agent.py
# or
start_voice_agent.bat
```

**Monitor Active Calls:**

```powershell
curl http://localhost:8000/api/sessions
```

**View API Documentation:**  
Open: http://localhost:8000/docs

### 📂 Files Overview

- **`windows_voice_agent.py`** - Main voice agent (runs on Windows)
- **`asterisk_config.json`** - Your Gate VoIP configuration
- **`setup_windows.py`** - Automated setup script
- **`test_windows_voip.py`** - Test everything
- **`.env`** - API keys and settings
- **`WINDOWS_VOIP_SETUP.md`** - Detailed setup guide

### 🆘 Common Issues

**Can't connect to Gate VoIP?**

```powershell
ping 192.168.50.127
```

**API key not working?**

```powershell
# Edit .env file and add:
# GOOGLE_API_KEY=your_actual_key_here
```

**Firewall blocking?**

```powershell
# As Administrator:
netsh advfirewall firewall add rule name="VoIP Agent" dir=in action=allow protocol=TCP localport=8000
```

### 🎉 Success Indicators

✅ **Health endpoint returns "healthy"**  
✅ **Can ping Gate VoIP system**  
✅ **Audio devices detected**  
✅ **Google AI key works**  
✅ **SIP port 5060 accessible**

### 🚀 Ready to Go!

Your Windows system is now a complete **AI telephone system** that:

- Answers calls on +359898995151 with Gemini AI
- Makes outbound calls via REST API
- Logs all conversations
- Works with your existing Gate VoIP hardware

**Test it:** Call +359898995151 and talk to your AI assistant! 🤖📞
