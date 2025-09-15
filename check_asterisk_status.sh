#!/bin/bash

# Asterisk Status Checker Script
# This script checks active connections, SIP peers, and system status

echo "=================================================="
echo "🎯 ASTERISK STATUS CHECKER"
echo "=================================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "=================================================="
echo

# Function to run Asterisk CLI command and format output
run_cli_command() {
    local command="$1"
    local description="$2"
    
    echo "📋 $description"
    echo "Command: asterisk -rx '$command'"
    echo "--------------------------------------------------"
    
    # Run the command and capture output
    output=$(sudo asterisk -rx "$command" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        if [ -n "$output" ]; then
            echo "$output"
        else
            echo "   (No output - command executed successfully)"
        fi
    else
        echo "   ❌ Error executing command"
    fi
    
    echo "--------------------------------------------------"
    echo
}

# Check Asterisk core status
run_cli_command "core show version" "ASTERISK VERSION & STATUS"

# Check active channels (ongoing calls)
run_cli_command "core show channels" "ACTIVE CHANNELS (ONGOING CALLS)"

# Check SIP peers (registered devices/trunks)
run_cli_command "sip show peers" "SIP PEERS (REGISTERED DEVICES)"

# Check SIP registrations
run_cli_command "sip show registry" "SIP REGISTRATIONS (OUTBOUND)"

# Check specific peer details (voice agent)
run_cli_command "sip show peer voice-agent" "VOICE AGENT TRUNK STATUS"

# Check dialplan
run_cli_command "dialplan show" "DIALPLAN CONFIGURATION"

# Check active calls details
run_cli_command "core show calls" "ACTIVE CALLS SUMMARY"

# Check system uptime
run_cli_command "core show uptime" "ASTERISK UPTIME"

# Check recent call logs
echo "📋 RECENT CALL ATTEMPTS (last 10 lines)"
echo "Command: tail -n 10 /var/log/asterisk/full"
echo "--------------------------------------------------"
if [ -f "/var/log/asterisk/full" ]; then
    tail -n 10 /var/log/asterisk/full 2>/dev/null | grep -E "(INVITE|SIP|Call|DTMF)" || echo "   (No recent call logs found)"
else
    echo "   (Log file not found)"
fi
echo "--------------------------------------------------"
echo

# Check if voice agent is reachable
echo "📋 VOICE AGENT CONNECTIVITY TEST"
echo "Command: ping -c 3 192.168.50.159"
echo "--------------------------------------------------"
ping -c 3 192.168.50.159 2>/dev/null | tail -n 2 || echo "   ❌ Voice agent not reachable"
echo "--------------------------------------------------"
echo

# Summary
echo "=================================================="
echo "🎯 STATUS SUMMARY"
echo "=================================================="

# Count active channels
channels=$(sudo asterisk -rx "core show channels" 2>/dev/null | grep "active channels" | head -1)
echo "Active Channels: ${channels:-Unknown}"

# Check if voice-agent peer is registered
voice_agent_status=$(sudo asterisk -rx "sip show peer voice-agent" 2>/dev/null | grep "Status" | head -1)
echo "Voice Agent Status: ${voice_agent_status:-Not configured}"

# Check SIP peer count
peer_count=$(sudo asterisk -rx "sip show peers" 2>/dev/null | tail -1)
echo "SIP Peers: ${peer_count:-Unknown}"

echo "=================================================="
echo "✅ Status check completed!"
echo "=================================================="

# Instructions
echo
echo "📝 MANUAL CLI COMMANDS TO TRY:"
echo
echo "Connect to CLI:           sudo asterisk -r"
echo "Show active calls:        core show calls"
echo "Show SIP peers:           sip show peers"
echo "Show voice agent:         sip show peer voice-agent"
echo "Reload SIP:               sip reload"
echo "Exit CLI:                 exit"
echo
echo "🎯 TO TEST VOICE AGENT:"
echo
echo "1. Run this script:       ./check_asterisk_status.sh"
echo "2. Call +359898995151 from external phone"
echo "3. Run this script again to see active calls"
echo "4. Check voice agent logs for RTP/SIP activity"
echo

