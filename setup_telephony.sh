#!/bin/bash
# Voice Agent Telephony Setup Script
# This script helps automate the installation and configuration process

set -e  # Exit on any error

echo "=========================================="
echo "Voice Agent Telephony Integration Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions for colored output
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Check if running as root (needed for some operations)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root. Some operations may require regular user privileges."
        SUDO=""
    else
        SUDO="sudo"
    fi
}

# Check system requirements
check_requirements() {
    info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        success "Linux detected"
        if command -v apt-get &> /dev/null; then
            PACKAGE_MANAGER="apt-get"
        elif command -v yum &> /dev/null; then
            PACKAGE_MANAGER="yum"
        else
            error "Unsupported package manager. Please install dependencies manually."
            exit 1
        fi
    else
        warning "Non-Linux OS detected. Manual configuration may be required."
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [[ $PYTHON_MAJOR -gt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            success "Python $PYTHON_VERSION detected"
        else
            error "Python 3.8+ required, found Python $PYTHON_VERSION"
            exit 1
        fi
    else
        error "Python 3 not found"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    info "Installing system dependencies..."
    
    if [[ "$PACKAGE_MANAGER" == "apt-get" ]]; then
        $SUDO apt-get update
        $SUDO apt-get install -y \
            python3-pip \
            python3-venv \
            python3-dev \
            build-essential \
            portaudio19-dev \
            libasound2-dev \
            asterisk \
            curl \
            wget
    elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        $SUDO yum update -y
        $SUDO yum install -y \
            python3-pip \
            python3-devel \
            gcc \
            gcc-c++ \
            portaudio-devel \
            alsa-lib-devel \
            asterisk \
            curl \
            wget
    fi
    
    success "System dependencies installed"
}

# Create Python virtual environment
setup_python_env() {
    info "Setting up Python virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        success "Virtual environment created"
    else
        info "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        success "Python dependencies installed"
    else
        warning "requirements.txt not found. Installing basic dependencies..."
        pip install fastapi uvicorn requests websockets pydub google-genai pyaudio pymongo opencv-python pillow mss python-dotenv
    fi
}

# Configure Asterisk
configure_asterisk() {
    info "Configuring Asterisk..."
    
    # Backup existing configuration
    if [[ -f "/etc/asterisk/extensions.conf" ]]; then
        $SUDO cp /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.backup.$(date +%Y%m%d_%H%M%S)
        info "Backed up existing extensions.conf"
    fi
    
    # Copy voice agent dialplan
    if [[ -f "asterisk_dialplan.conf" ]]; then
        $SUDO cp asterisk_dialplan.conf /etc/asterisk/extensions_voice_agent.conf
        
        # Add include to main extensions.conf if not already present
        if ! grep -q "extensions_voice_agent.conf" /etc/asterisk/extensions.conf; then
            echo "#include extensions_voice_agent.conf" | $SUDO tee -a /etc/asterisk/extensions.conf > /dev/null
            success "Voice agent dialplan added to Asterisk configuration"
        else
            info "Voice agent dialplan already included in extensions.conf"
        fi
    else
        warning "asterisk_dialplan.conf not found. Please configure manually."
    fi
    
    # Install AGI script
    if [[ -f "voice_agent_agi.py" ]]; then
        $SUDO cp voice_agent_agi.py /var/lib/asterisk/agi-bin/
        $SUDO chmod +x /var/lib/asterisk/agi-bin/voice_agent_agi.py
        $SUDO chown asterisk:asterisk /var/lib/asterisk/agi-bin/voice_agent_agi.py
        success "AGI script installed"
    else
        warning "voice_agent_agi.py not found"
    fi
    
    # Create log directory
    $SUDO mkdir -p /var/log/asterisk
    $SUDO chown asterisk:asterisk /var/log/asterisk
    $SUDO chmod 755 /var/log/asterisk
}

# Configure voice agent
configure_voice_agent() {
    info "Configuring voice agent..."
    
    # Create configuration file if it doesn't exist
    if [[ ! -f "asterisk_config.json" ]]; then
        cat > asterisk_config.json << EOF
{
  "host": "YOUR_SIM_GATEWAY_IP_HERE",
  "username": "admin",
  "password": "YOUR_PASSWORD_HERE",
  "sip_port": 5060,
  "ari_port": 8088,
  "ari_username": "asterisk",
  "ari_password": "asterisk",
  "context": "voice-agent"
}
EOF
        warning "Created default asterisk_config.json - please update with your SIM gateway details"
    else
        info "Configuration file already exists"
    fi
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Google AI API Key
GOOGLE_API_KEY=your_google_ai_api_key_here

# MongoDB (optional)
MONGODB_HOST=localhost
MONGODB_PORT=27017

# Voice Agent Settings
VOICE_AGENT_HOST=0.0.0.0
VOICE_AGENT_PORT=8000
EOF
        warning "Created default .env file - please update with your API keys"
    else
        info ".env file already exists"
    fi
    
    # Create sessions directory
    mkdir -p sessions
    success "Voice agent configuration completed"
}

# Start services
start_services() {
    info "Starting services..."
    
    # Reload Asterisk configuration
    if $SUDO systemctl is-active --quiet asterisk; then
        $SUDO asterisk -rx "core reload"
        $SUDO asterisk -rx "sip reload"
        success "Asterisk configuration reloaded"
    else
        $SUDO systemctl start asterisk
        $SUDO systemctl enable asterisk
        success "Asterisk started and enabled"
    fi
    
    # Check if voice server should be started
    echo
    info "To start the voice agent server, run:"
    echo "source venv/bin/activate"
    echo "python agi_voice_server.py --host 0.0.0.0 --port 8000"
    echo
    warning "Remember to update your configuration files before starting!"
}

# Run tests
run_tests() {
    info "Running basic tests..."
    
    source venv/bin/activate
    
    if [[ -f "test_telephony.py" ]]; then
        python test_telephony.py --system-checks
    else
        warning "Test file not found. Skipping tests."
    fi
}

# Display configuration instructions
show_next_steps() {
    echo
    info "=========================================="
    info "SETUP COMPLETED - NEXT STEPS"
    info "=========================================="
    echo
    warning "Before using the system, you need to:"
    echo
    echo "1. Update asterisk_config.json with your SIM gateway details:"
    echo "   - host: Your SIM gateway IP address"
    echo "   - username/password: Your VoIP panel credentials"
    echo
    echo "2. Update .env file with your Google AI API key:"
    echo "   - Get your API key from Google AI Studio"
    echo "   - Set GOOGLE_API_KEY in the .env file"
    echo
    echo "3. Configure your SIM gateway to route calls to this server:"
    echo "   - Set up SIP trunk pointing to your Asterisk server"
    echo "   - Configure incoming call routing"
    echo
    echo "4. Start the voice agent server:"
    echo "   source venv/bin/activate"
    echo "   python agi_voice_server.py"
    echo
    echo "5. Test the setup:"
    echo "   python test_telephony.py"
    echo
    info "For detailed configuration instructions, see SETUP_TELEPHONY.md"
    echo
}

# Main installation flow
main() {
    echo
    info "Starting Voice Agent Telephony Integration setup..."
    echo
    
    check_root
    check_requirements
    
    # Ask for confirmation
    read -p "Do you want to continue with the installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Installation cancelled by user"
        exit 0
    fi
    
    install_system_deps
    setup_python_env
    configure_asterisk
    configure_voice_agent
    start_services
    run_tests
    show_next_steps
    
    success "Setup completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "--help"|"-h")
        echo "Voice Agent Telephony Setup Script"
        echo
        echo "Usage: $0 [OPTION]"
        echo
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --deps-only    Install system dependencies only"
        echo "  --python-only  Set up Python environment only"
        echo "  --config-only  Configure services only"
        echo "  --test         Run tests only"
        echo
        exit 0
        ;;
    "--deps-only")
        check_root
        check_requirements
        install_system_deps
        exit 0
        ;;
    "--python-only")
        check_requirements
        setup_python_env
        exit 0
        ;;
    "--config-only")
        check_root
        configure_asterisk
        configure_voice_agent
        exit 0
        ;;
    "--test")
        run_tests
        exit 0
        ;;
    "")
        main
        ;;
    *)
        error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac