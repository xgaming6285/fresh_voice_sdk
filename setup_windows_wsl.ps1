# Voice Agent Telephony Setup for Windows 11 + WSL
# PowerShell script to automate WSL setup and project transfer

param(
    [switch]$InstallWSL,
    [switch]$SetupProject,
    [switch]$ConfigureFirewall,
    [string]$WSLDistro = "Ubuntu-22.04"
)

# Colors for output
$ErrorColor = "Red"
$SuccessColor = "Green"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

function Write-Info($message) {
    Write-Host "ℹ️  $message" -ForegroundColor $InfoColor
}

function Write-Success($message) {
    Write-Host "✅ $message" -ForegroundColor $SuccessColor
}

function Write-Warning($message) {
    Write-Host "⚠️  $message" -ForegroundColor $WarningColor
}

function Write-Error($message) {
    Write-Host "❌ $message" -ForegroundColor $ErrorColor
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-WSL {
    Write-Info "Installing WSL with $WSLDistro..."
    
    if (!(Test-Administrator)) {
        Write-Error "Administrator privileges required for WSL installation."
        Write-Info "Please run this script as Administrator or install WSL manually:"
        Write-Info "  wsl --install -d $WSLDistro"
        return $false
    }
    
    try {
        # Install WSL
        wsl --install -d $WSLDistro
        Write-Success "WSL installation initiated"
        Write-Warning "Please reboot your computer and run this script again with -SetupProject"
        return $true
    }
    catch {
        Write-Error "Failed to install WSL: $_"
        return $false
    }
}

function Test-WSLInstalled {
    try {
        $wslList = wsl -l -v 2>$null
        return $wslList -match $WSLDistro
    }
    catch {
        return $false
    }
}

function Setup-Project {
    Write-Info "Setting up Voice Agent project in WSL..."
    
    if (!(Test-WSLInstalled)) {
        Write-Error "WSL with $WSLDistro is not installed."
        Write-Info "Run this script with -InstallWSL first"
        return $false
    }
    
    # Get current directory (should be the project root)
    $projectPath = Get-Location
    $projectName = Split-Path $projectPath -Leaf
    
    Write-Info "Project path: $projectPath"
    Write-Info "Copying project to WSL..."
    
    try {
        # Copy project to WSL
        wsl -d $WSLDistro -e bash -c "mkdir -p ~/$projectName"
        
        # Copy all files
        $files = Get-ChildItem -Path $projectPath -Recurse | Where-Object { !$_.PSIsContainer }
        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring($projectPath.Path.Length + 1)
            $wslPath = $relativePath -replace '\\', '/'
            $wslDir = Split-Path "~/$projectName/$wslPath" -Parent
            
            # Create directory structure in WSL
            wsl -d $WSLDistro -e bash -c "mkdir -p '$wslDir'"
            
            # Copy file
            $windowsPath = "/mnt/" + $file.FullName.Substring(0,1).ToLower() + $file.FullName.Substring(2) -replace '\\', '/'
            wsl -d $WSLDistro -e bash -c "cp '$windowsPath' '~/$projectName/$wslPath'"
        }
        
        Write-Success "Project copied to WSL successfully"
        
        # Run the Linux setup script
        Write-Info "Running setup script in WSL..."
        wsl -d $WSLDistro -e bash -c "cd ~/$projectName && chmod +x setup_telephony.sh && ./setup_telephony.sh"
        
        Write-Success "WSL project setup completed"
        return $true
    }
    catch {
        Write-Error "Failed to setup project in WSL: $_"
        return $false
    }
}

function Configure-WindowsFirewall {
    Write-Info "Configuring Windows Firewall for Voice Agent..."
    
    if (!(Test-Administrator)) {
        Write-Error "Administrator privileges required for firewall configuration."
        return $false
    }
    
    try {
        # Voice Agent API port
        New-NetFirewallRule -DisplayName "WSL Voice Agent API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -ErrorAction SilentlyContinue
        Write-Success "Firewall rule added for port 8000 (Voice Agent API)"
        
        # SIP port
        New-NetFirewallRule -DisplayName "WSL SIP" -Direction Inbound -Protocol UDP -LocalPort 5060 -Action Allow -ErrorAction SilentlyContinue  
        Write-Success "Firewall rule added for port 5060 (SIP)"
        
        # RTP ports range
        New-NetFirewallRule -DisplayName "WSL RTP" -Direction Inbound -Protocol UDP -LocalPort "10000-20000" -Action Allow -ErrorAction SilentlyContinue
        Write-Success "Firewall rule added for ports 10000-20000 (RTP)"
        
        Write-Success "Windows Firewall configured successfully"
        return $true
    }
    catch {
        Write-Error "Failed to configure firewall: $_"
        return $false
    }
}

function Show-NextSteps {
    Write-Info "=========================================="
    Write-Info "NEXT STEPS"
    Write-Info "=========================================="
    Write-Success "WSL setup completed successfully!"
    Write-Info ""
    Write-Info "To complete the setup:"
    Write-Info "1. Open WSL terminal: wsl -d $WSLDistro"
    Write-Info "2. Navigate to project: cd ~/fresh_voice_sdk"
    Write-Info "3. Edit configuration:"
    Write-Info "   nano asterisk_config.json  # Add your SIM gateway details"
    Write-Info "   nano .env                   # Add your Google API key"
    Write-Info "4. Start services:"
    Write-Info "   sudo systemctl start asterisk"
    Write-Info "   source venv/bin/activate"
    Write-Info "   python agi_voice_server.py --host 0.0.0.0 --port 8000"
    Write-Info "5. Test the setup:"
    Write-Info "   python test_telephony.py"
    Write-Info ""
    Write-Warning "Don't forget to configure your SIM gateway to route calls to your Windows IP!"
    Write-Info ""
    Write-Info "For detailed instructions, see SETUP_WINDOWS_WSL.md"
}

function Get-WindowsIP {
    $ip = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch "Loopback" -and $_.InterfaceAlias -notmatch "vEthernet" } | Select-Object -First 1
    return $ip.IPAddress
}

function Main {
    Write-Info "=========================================="
    Write-Info "Voice Agent Telephony - Windows 11 + WSL Setup"
    Write-Info "=========================================="
    Write-Info ""
    
    # Check Windows version
    $osVersion = [Environment]::OSVersion.Version
    if ($osVersion.Major -lt 10) {
        Write-Error "Windows 10 version 1903 or higher is required for WSL 2"
        exit 1
    }
    
    Write-Success "Windows version check passed"
    
    if ($InstallWSL) {
        $result = Install-WSL
        if ($result) {
            Write-Warning "WSL installation completed. Please reboot and run with -SetupProject"
        }
        return
    }
    
    if ($SetupProject) {
        $result = Setup-Project
        if (!$result) {
            exit 1
        }
    }
    
    if ($ConfigureFirewall) {
        Configure-WindowsFirewall
    }
    
    # Show Windows IP for SIM gateway configuration
    $windowsIP = Get-WindowsIP
    Write-Info "Your Windows IP address: $windowsIP"
    Write-Info "Configure your SIM gateway to route calls to this IP"
    
    if ($SetupProject -or $ConfigureFirewall) {
        Show-NextSteps
    }
    
    if (!$InstallWSL -and !$SetupProject -and !$ConfigureFirewall) {
        Write-Info "Usage:"
        Write-Info "  .\setup_windows_wsl.ps1 -InstallWSL          # Install WSL (run as Administrator)"
        Write-Info "  .\setup_windows_wsl.ps1 -SetupProject        # Setup project in existing WSL"
        Write-Info "  .\setup_windows_wsl.ps1 -ConfigureFirewall   # Configure Windows Firewall (run as Administrator)"
        Write-Info ""
        Write-Info "For first-time setup:"
        Write-Info "  1. .\setup_windows_wsl.ps1 -InstallWSL"
        Write-Info "  2. Reboot computer"
        Write-Info "  3. .\setup_windows_wsl.ps1 -SetupProject -ConfigureFirewall"
    }
}

# Run main function
Main