# 🖥️ Windows 11 Quick Start Guide

## Voice Agent Telephony on Windows 11 in 10 Minutes

This guide will get your voice agent working with phone calls on Windows 11 using WSL.

### ⚡ Quick Setup Steps

#### 1. Install WSL (5 minutes)

Open **PowerShell as Administrator**:

```powershell
# Install WSL with Ubuntu
wsl --install

# Reboot your computer when prompted
```

After reboot, Ubuntu will start automatically:

- Create username/password when prompted
- Wait for installation to complete

#### 2. Setup Voice Agent (3 minutes)

In **PowerShell** (regular user, not administrator):

```powershell
# Navigate to your project
cd C:\Users\dani0\GitProjects\fresh_voice_sdk

# Copy project to WSL and run setup
wsl cp -r . ~/fresh_voice_sdk/
wsl -e bash -c "cd ~/fresh_voice_sdk && chmod +x setup_telephony.sh && ./setup_telephony.sh"
```

#### 3. Configure Settings (2 minutes)

Edit configuration files in WSL:

```bash
# In WSL terminal
cd ~/fresh_voice_sdk

# Edit SIM gateway settings
nano asterisk_config.json
# Update: host, username, password with your SIM gateway details

# Edit Google API key
nano .env
# Add: GOOGLE_API_KEY=your_google_ai_api_key_here
```

#### 4. Start Services (1 minute)

```bash
# In WSL terminal
cd ~/fresh_voice_sdk

# Start Asterisk
sudo systemctl start asterisk

# Start voice agent
source venv/bin/activate
python agi_voice_server.py --host 0.0.0.0 --port 8000
```

#### 5. Test Everything

```bash
# In another WSL terminal
cd ~/fresh_voice_sdk
source venv/bin/activate
python test_telephony.py --system-checks
```

### 🎯 You're Done!

Your voice agent can now:

- **Answer phone calls** through your SIM gateway
- **Make outbound calls** via API
- **Process conversations** with Google Gemini AI
- **Log everything** with your existing system

### ⚙️ Configure Your SIM Gateway

In your **VoIP panel** (the one your boss gave you access to):

1. **Find your Windows IP**: In PowerShell: `ipconfig`
2. **Add SIP trunk**: Point to your Windows IP, port 5060
3. **Route incoming calls**: Set context to route to your system

### 🔧 Quick Commands

**Start services:**

```bash
# WSL terminal
cd ~/fresh_voice_sdk
sudo systemctl start asterisk
source venv/bin/activate
python agi_voice_server.py
```

**Test call via API:**

```powershell
# Windows PowerShell
curl -X POST http://localhost:8000/api/make_call -H "Content-Type: application/json" -d '{"phone_number": "+1234567890"}'
```

**Check system health:**

```bash
# WSL terminal
curl http://localhost:8000/health
```

### 📁 File Access

- **Edit in Windows**: Use VS Code/your editor on Windows files
- **Run in WSL**: Execute commands in WSL terminal
- **Access WSL files**: `\\wsl$\Ubuntu\home\username\fresh_voice_sdk`

### 🆘 Troubleshooting

**WSL not working?**

```powershell
wsl --status
wsl --version  # Should show WSL 2
```

**Can't connect to voice server from Windows?**

```powershell
# Add firewall rule (as Administrator)
New-NetFirewallRule -DisplayName "WSL Voice Agent" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

**Asterisk issues?**

```bash
# Check Asterisk status
sudo systemctl status asterisk
sudo asterisk -rx "core show version"
```

### 🎉 What's Next?

- **Call your SIM number** - AI should answer!
- **Use the API** to make outbound calls
- **Monitor conversations** in the sessions/ directory
- **Scale up** for production use

For detailed configuration, see `SETUP_WINDOWS_WSL.md`

---

**That's it!** Your Windows 11 system is now a full-featured AI telephone system! 📞🤖
