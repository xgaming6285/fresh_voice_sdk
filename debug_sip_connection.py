#!/usr/bin/env python3
"""
Debug script for SIP connection issues
Helps identify IP address mismatches and connectivity problems
"""

import socket
import json
import requests
from pathlib import Path

def load_config():
    """Load asterisk configuration"""
    try:
        with open('asterisk_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ asterisk_config.json not found")
        return None
    except json.JSONDecodeError:
        print("❌ Invalid JSON in asterisk_config.json")
        return None

def check_network_connectivity():
    """Check network connectivity and IP addresses"""
    print("🔍 Network Connectivity Check")
    print("=" * 50)
    
    # Get local IP addresses
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"📍 Hostname: {hostname}")
        print(f"📍 Local IP (gethostbyname): {local_ip}")
        
        # Get all network interfaces
        import subprocess
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        print(f"\n📍 Network Interfaces:")
        for line in result.stdout.split('\n'):
            if 'IPv4 Address' in line or 'IP Address' in line:
                print(f"   {line.strip()}")
                
    except Exception as e:
        print(f"❌ Error getting local IP: {e}")
    
    print()

def check_config_ips():
    """Check IP addresses in configuration"""
    print("🔍 Configuration Check")
    print("=" * 50)
    
    config = load_config()
    if not config:
        return False
        
    print(f"📍 Asterisk Host (Gate VoIP): {config.get('host', 'NOT SET')}")
    print(f"📍 Voice Agent IP: {config.get('voice_agent_ip', 'NOT SET')}")
    print(f"📍 Local IP: {config.get('local_ip', 'NOT SET')}")
    print(f"📍 Gateway IP: {config.get('gateway_ip', 'NOT SET')}")
    print(f"📍 Public IP: {config.get('public_ip', 'NOT SET')}")
    print()
    
    return True

def test_sip_port():
    """Test SIP port accessibility"""
    print("🔍 SIP Port Test")
    print("=" * 50)
    
    config = load_config()
    if not config:
        return
    
    local_ip = config.get('local_ip', '192.168.50.128')
    sip_port = config.get('sip_port', 5060)
    
    try:
        # Try to bind to SIP port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((local_ip, sip_port))
        print(f"✅ Can bind to {local_ip}:{sip_port}")
        sock.close()
    except socket.error as e:
        print(f"❌ Cannot bind to {local_ip}:{sip_port} - {e}")
        print("💡 The port might already be in use by the voice agent")
    
    print()

def test_gate_connectivity():
    """Test connectivity to Gate VoIP"""
    print("🔍 Gate VoIP Connectivity Test")
    print("=" * 50)
    
    config = load_config()
    if not config:
        return
    
    gate_ip = config.get('host', '192.168.50.50')
    sip_port = config.get('sip_port', 5060)
    
    try:
        # Test UDP connectivity to Gate VoIP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2)
        
        # Send a simple packet
        test_message = b"test"
        sock.sendto(test_message, (gate_ip, sip_port))
        print(f"✅ Can send UDP packets to {gate_ip}:{sip_port}")
        sock.close()
        
    except socket.error as e:
        print(f"❌ Cannot reach {gate_ip}:{sip_port} - {e}")
    
    print()

def check_voice_agent_status():
    """Check if voice agent is running"""
    print("🔍 Voice Agent Status")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Voice agent is running and healthy")
            data = response.json()
            print(f"📍 Status: {data.get('status', 'unknown')}")
            print(f"📍 Local IP: {data.get('local_ip', 'unknown')}")
        else:
            print(f"⚠️ Voice agent responded with status {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Voice agent is not running or not accessible: {e}")
        print("💡 Make sure to start the voice agent with: python windows_voice_agent.py --port 8001")
    
    print()

def identify_ip_mismatch():
    """Identify potential IP mismatches"""
    print("🔍 IP Mismatch Analysis")
    print("=" * 50)
    
    config = load_config()
    if not config:
        return
    
    voice_agent_ip = config.get('voice_agent_ip', 'NOT SET')
    local_ip = config.get('local_ip', 'NOT SET')
    
    # Common issues
    issues = []
    
    if voice_agent_ip != local_ip:
        issues.append(f"voice_agent_ip ({voice_agent_ip}) != local_ip ({local_ip})")
    
    if voice_agent_ip == 'NOT SET':
        issues.append("voice_agent_ip is not configured")
    
    if local_ip == 'NOT SET':
        issues.append("local_ip is not configured")
    
    # Check for common IP patterns
    if '192.168.2.' in voice_agent_ip or '192.168.2.' in local_ip:
        issues.append("Detected 192.168.2.x IP which might be incorrect (should be 192.168.50.x?)")
    
    if issues:
        print("⚠️ Potential issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print()
        print("💡 Recommendations:")
        print("   1. Ensure voice_agent_ip and local_ip are the same")
        print("   2. Verify the IP address matches your actual network interface")
        print("   3. Check your Gate VoIP configuration to ensure it points to the correct IP")
    else:
        print("✅ No obvious IP configuration issues detected")
    
    print()

def main():
    """Main debug function"""
    print("🔧 SIP Connection Debug Tool")
    print("=" * 50)
    print()
    
    check_network_connectivity()
    check_config_ips()
    test_sip_port()
    test_gate_connectivity()
    check_voice_agent_status()
    identify_ip_mismatch()
    
    print("🔧 Debug complete!")
    print()
    print("📋 Next steps if issues found:")
    print("1. Fix IP address configuration in asterisk_config.json")
    print("2. Check your Gate VoIP SIP trunk configuration")
    print("3. Verify Windows firewall allows UDP traffic on port 5060")
    print("4. Restart the voice agent after configuration changes")

if __name__ == "__main__":
    main()
