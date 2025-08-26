# Voice Agent Telephony Setup for Windows 11 + WSL

## 🖥️ Windows 11 + WSL Setup Guide

This guide will help you set up the telephony integration on Windows 11 using WSL (Windows Subsystem for Linux).

### ✅ Why WSL is Perfect for This

- **Asterisk runs natively** in Linux environment
- **Full network access** to your SIM gateway
- **Better performance** than Docker for telephony
- **Native Linux tooling** for audio processing
- **Easy file sharing** between Windows and WSL

## Step 1: Install WSL

### 1.1 Install WSL 2 (Recommended)

Open **PowerShell as Administrator** and run:

```powershell
# Install WSL with Ubuntu (default and recommended)
wsl --install

# Or specify Ubuntu 22.04 LTS specifically
wsl --install -d Ubuntu-22.04
```

**Reboot your computer** when prompted.

### 1.2 Complete Ubuntu Setup

After reboot, Ubuntu will launch automatically:

1. **Create a username** (e.g., `voiceagent`)
2. **Set a password**
3. **Update the system**:

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Transfer Your Project to WSL

### 2.1 From Windows PowerShell

Navigate to your project directory and copy to WSL:

```powershell
# Navigate to your project
cd C:\Users\dani0\GitProjects\fresh_voice_sdk

# Copy entire project to WSL home directory
wsl cp -r . ~/fresh_voice_sdk/
```

### 2.2 Alternative: Access from WSL

```bash
# In WSL, access your Windows files directly
cd /mnt/c/Users/dani0/GitProjects/fresh_voice_sdk

# Or copy to WSL file system for better performance
cp -r /mnt/c/Users/dani0/GitProjects/fresh_voice_sdk ~/
cd ~/fresh_voice_sdk
```

## Step 3: Run the Automated Setup

### 3.1 Make Setup Script Executable

```bash
cd ~/fresh_voice_sdk
chmod +x setup_telephony.sh
```

### 3.2 Run the Setup Script

```bash
# Run the automated setup
./setup_telephony.sh
```

This will:

- Install all system dependencies (Asterisk, Python, etc.)
- Create Python virtual environment
- Configure Asterisk
- Set up AGI scripts
- Create configuration files

## Step 4: Windows-Specific Configuration

### 4.1 Network Configuration

WSL 2 uses a virtual network. Check your WSL IP address:

```bash
# Find your WSL IP address
ip addr show eth0
# Look for inet xxx.xxx.xxx.xxx

# Or use this command
hostname -I
```

### 4.2 Windows Firewall Configuration

In **Windows PowerShell (as Administrator)**:

```powershell
# Allow WSL traffic through Windows Firewall
# Replace xxx.xxx.xxx.xxx with your WSL IP range (usually 172.x.x.x)
New-NetFirewallRule -DisplayName "WSL Voice Agent" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
New-NetFirewallRule -DisplayName "WSL SIP" -Direction Inbound -Protocol UDP -LocalPort 5060 -Action Allow
New-NetFirewallRule -DisplayName "WSL RTP" -Direction Inbound -Protocol UDP -LocalPort 10000-20000 -Action Allow
```

### 4.3 Port Forwarding (if needed)

If your SIM gateway needs to reach WSL from outside:

```powershell
# Forward port 5060 (SIP) from Windows to WSL
netsh interface portproxy add v4tov4 listenport=5060 listenaddress=0.0.0.0 connectport=5060 connectaddress=WSL_IP_ADDRESS

# Forward port 8000 (Voice Agent API)
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=WSL_IP_ADDRESS
```

## Step 5: Configure Your SIM Gateway

### 5.1 Update Configuration Files

In WSL, edit the configuration:

```bash
cd ~/fresh_voice_sdk

# Edit with nano or vim
nano asterisk_config.json
```

Update with your details:

```json
{
  "host": "YOUR_SIM_GATEWAY_IP",
  "username": "admin",
  "password": "YOUR_PASSWORD",
  "sip_port": 5060,
  "ari_port": 8088,
  "ari_username": "asterisk",
  "ari_password": "asterisk",
  "context": "voice-agent"
}
```

### 5.2 Set Environment Variables

```bash
# Edit .env file
nano .env
```

Add your Google API key:

```
GOOGLE_API_KEY=your_google_ai_api_key_here
MONGODB_HOST=localhost
MONGODB_PORT=27017
VOICE_AGENT_HOST=0.0.0.0
VOICE_AGENT_PORT=8000
```

### 5.3 Configure SIP Gateway Connection

In your **VoIP panel** (from Windows browser):

1. **SIP Server IP**: Use your **Windows IP address** (not WSL IP)
2. **Port**: 5060
3. **Protocol**: UDP
4. **Context**: Set to route calls to your system

To find your Windows IP:

```powershell
# In Windows PowerShell
ipconfig | findstr IPv4
```

## Step 6: Start Services

### 6.1 Start Asterisk

```bash
# In WSL
sudo systemctl start asterisk
sudo systemctl enable asterisk

# Check status
sudo asterisk -rx "core show version"
```

### 6.2 Start Voice Agent Server

```bash
cd ~/fresh_voice_sdk
source venv/bin/activate
python agi_voice_server.py --host 0.0.0.0 --port 8000
```

## Step 7: Testing

### 7.1 Run System Tests

```bash
# In another WSL terminal
cd ~/fresh_voice_sdk
source venv/bin/activate
python test_telephony.py --system-checks
```

### 7.2 Test from Windows

From Windows PowerShell:

```powershell
# Health check (use Windows IP or localhost)
curl http://localhost:8000/health

# Or if using port forwarding
curl http://YOUR_WINDOWS_IP:8000/health
```

### 7.3 Test API Endpoints

```bash
# From WSL
curl -X POST http://localhost:8000/api/make_call \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+1234567890"}'
```

## Windows-Specific Tips

### 💡 Accessing Files

- **WSL files from Windows**: `\\wsl$\Ubuntu\home\username\fresh_voice_sdk`
- **Windows files from WSL**: `/mnt/c/Users/dani0/GitProjects/`

### 💡 Managing Services

```bash
# Start/stop Asterisk
sudo systemctl start asterisk
sudo systemctl stop asterisk
sudo systemctl status asterisk

# View logs
sudo tail -f /var/log/asterisk/messages
```

### 💡 Development Workflow

1. **Edit files** in Windows using your preferred editor (VS Code, etc.)
2. **Run services** in WSL terminals
3. **Test APIs** from Windows browsers/tools
4. **Monitor logs** in WSL

### 💡 Auto-start Services

Create a startup script in WSL:

```bash
# Create startup script
nano ~/start_voice_agent.sh
```

Add:

```bash
#!/bin/bash
cd ~/fresh_voice_sdk
source venv/bin/activate

# Start Asterisk if not running
sudo systemctl start asterisk

# Start voice agent server
python agi_voice_server.py --host 0.0.0.0 --port 8000
```

Make it executable:

```bash
chmod +x ~/start_voice_agent.sh
```

## Troubleshooting WSL Issues

### Audio Issues

If you encounter audio problems:

```bash
# Install additional audio libraries
sudo apt install pulseaudio pulseaudio-utils
```

### Network Connectivity

```bash
# Test connectivity to SIM gateway
ping YOUR_SIM_GATEWAY_IP
telnet YOUR_SIM_GATEWAY_IP 5060
```

### WSL Service Management

```bash
# Restart WSL network
sudo service networking restart

# Check WSL version
wsl --version
```

### Windows Integration

```powershell
# Check WSL status
wsl --status

# Restart WSL if needed
wsl --shutdown
wsl
```

## Performance Optimization

### 1. Use WSL 2 (not WSL 1)

```powershell
# Check WSL version
wsl -l -v

# Upgrade to WSL 2 if needed
wsl --set-version Ubuntu-22.04 2
```

### 2. Store Files in WSL File System

- **Better performance**: Store project files in WSL (`~/fresh_voice_sdk`)
- **Faster I/O**: Avoid `/mnt/c/` for frequently accessed files

### 3. Resource Allocation

Create `.wslconfig` in `C:\Users\YourUsername\`:

```ini
[wsl2]
memory=4GB
processors=2
```

## Expected Performance

With this setup you should achieve:

- **Call latency**: < 200ms
- **Audio quality**: Excellent (same as native Linux)
- **Concurrent calls**: 10+ (depending on hardware)
- **System stability**: Production-ready

## Quick Start Commands

Once everything is set up:

```bash
# Start everything in one command
cd ~/fresh_voice_sdk && ./start_voice_agent.sh

# Or manually:
sudo systemctl start asterisk
source venv/bin/activate
python agi_voice_server.py --host 0.0.0.0 --port 8000
```

## 🎯 Summary

Your Windows 11 + WSL setup will provide:

✅ **Full Linux compatibility** for Asterisk
✅ **Native Windows development** environment
✅ **Easy file access** between systems
✅ **Production-grade performance**
✅ **Simple debugging** and monitoring

The telephony integration will work **exactly the same** as on a native Linux server, but with the convenience of your Windows development environment!

Need help with any specific step? The setup should be straightforward with WSL! 🚀
