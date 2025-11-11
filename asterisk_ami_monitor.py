"""
Asterisk AMI Monitor
Connects to Asterisk Manager Interface to capture linkedid for calls
"""
import asyncio
import logging
from panoramisk import Manager

logger = logging.getLogger(__name__)

class AsteriskAMIMonitor:
    """Monitor Asterisk AMI for linkedid and call events"""
    
    def __init__(self, host='192.168.50.50', port=5038, username='voipsystems', password='asccyber@1'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.manager = None
        self.linkedids = {}  # channel -> linkedid mapping
        self.session_linkedids = {}  # session_id -> linkedid mapping
        self.running = False
        
    async def connect(self):
        """Connect to Asterisk AMI"""
        try:
            self.manager = Manager(
                host=self.host,
                port=self.port,
                username=self.username,
                secret=self.password,
                ping_delay=10,
                ping_tries=3
            )
            
            # Register event handlers
            self.manager.register_event('Newchannel', self.on_new_channel)
            self.manager.register_event('Newstate', self.on_new_state)
            self.manager.register_event('DialBegin', self.on_dial_begin)
            self.manager.register_event('DialEnd', self.on_dial_end)
            self.manager.register_event('Hangup', self.on_hangup)
            
            await self.manager.connect()
            self.running = True
            logger.info("✅ Connected to Asterisk AMI")
            logger.info(f"🔗 Monitoring calls from Extension 200 for linkedid")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Asterisk AMI: {e}")
            logger.error(f"   Host: {self.host}:{self.port}")
            logger.error(f"   Username: {self.username}")
            import traceback
            logger.error(traceback.format_exc())
            
    async def on_new_channel(self, manager, event):
        """Handle new channel creation - capture linkedid"""
        try:
            channel = event.get('Channel', '')
            linkedid = event.get('Linkedid', '')
            uniqueid = event.get('Uniqueid', '')
            callerid = event.get('CallerIDNum', '')
            exten = event.get('Exten', '')
            
            # Check if this is from Extension 200 (our voice agent)
            if callerid == '200' or 'SIP/200-' in channel:
                if linkedid:
                    self.linkedids[channel] = linkedid
                    logger.info(f"🔗 NEW CALL - Asterisk Linked ID: {linkedid}")
                    logger.info(f"   Channel: {channel}")
                    logger.info(f"   UniqueID: {uniqueid}")
                    logger.info(f"   Calling: {exten}")
                    
        except Exception as e:
            logger.error(f"Error in on_new_channel: {e}")
    
    async def on_new_state(self, manager, event):
        """Handle channel state changes"""
        try:
            channel = event.get('Channel', '')
            linkedid = event.get('Linkedid', '')
            channelstate = event.get('ChannelState', '')
            channelstatedesc = event.get('ChannelStateDesc', '')
            
            # Update linkedid if we see it
            if linkedid and channel:
                if '200' in channel or 'SIP/200-' in channel:
                    if channel not in self.linkedids:
                        self.linkedids[channel] = linkedid
                        logger.info(f"🔗 Asterisk Linked ID: {linkedid} (State: {channelstatedesc})")
                        
        except Exception as e:
            logger.error(f"Error in on_new_state: {e}")
    
    async def on_dial_begin(self, manager, event):
        """Handle dial begin - this is when outbound call starts"""
        try:
            channel = event.get('Channel', '')
            destchannel = event.get('DestChannel', '')
            linkedid = event.get('Linkedid', '')
            dialstring = event.get('DialString', '')
            
            if linkedid and ('200' in channel or 'SIP/200-' in channel):
                self.linkedids[channel] = linkedid
                logger.info(f"📞 OUTBOUND CALL STARTED")
                logger.info(f"🔗 Asterisk Linked ID: {linkedid}")
                logger.info(f"   Dialing: {dialstring}")
                
        except Exception as e:
            logger.error(f"Error in on_dial_begin: {e}")
    
    async def on_dial_end(self, manager, event):
        """Handle dial end - call answered or failed"""
        try:
            channel = event.get('Channel', '')
            linkedid = event.get('Linkedid', '')
            dialstatus = event.get('DialStatus', '')
            
            if linkedid and ('200' in channel or 'SIP/200-' in channel):
                logger.info(f"📞 DIAL ENDED - Status: {dialstatus}")
                logger.info(f"🔗 Linked ID: {linkedid}")
                
        except Exception as e:
            logger.error(f"Error in on_dial_end: {e}")
    
    async def on_hangup(self, manager, event):
        """Clean up linkedid when channel hangs up"""
        try:
            channel = event.get('Channel', '')
            linkedid = event.get('Linkedid', '')
            cause = event.get('Cause', '')
            causetxt = event.get('Cause-txt', '')
            
            if channel in self.linkedids:
                stored_linkedid = self.linkedids[channel]
                logger.info(f"📞 CALL ENDED")
                logger.info(f"🔗 Linked ID: {stored_linkedid}")
                logger.info(f"   Cause: {causetxt} ({cause})")
                del self.linkedids[channel]
                
        except Exception as e:
            logger.error(f"Error in on_hangup: {e}")
    
    def get_linkedid_for_extension_200(self):
        """Get the most recent linkedid for Extension 200"""
        for channel, linkedid in self.linkedids.items():
            if 'SIP/200-' in channel or '/200-' in channel:
                return linkedid
        return None
    
    def get_all_linkedids(self):
        """Get all active linkedids"""
        return dict(self.linkedids)
    
    async def run(self):
        """Main AMI event loop"""
        await self.connect()
        
        # Keep running to receive events
        while self.running:
            try:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"AMI loop error: {e}")
                break
    
    async def stop(self):
        """Stop AMI monitoring"""
        self.running = False
        if self.manager:
            await self.manager.close()
        logger.info("🔗 Asterisk AMI monitor stopped")

# Global instance
ami_monitor = None

async def start_ami_monitoring(host='192.168.50.50', port=5038, username='voipsystems', password='asccyber@1'):
    """Start AMI monitoring"""
    global ami_monitor
    ami_monitor = AsteriskAMIMonitor(host, port, username, password)
    await ami_monitor.run()

def get_current_linkedid():
    """Get the current linkedid for Extension 200 calls"""
    if ami_monitor:
        return ami_monitor.get_linkedid_for_extension_200()
    return None