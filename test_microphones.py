#!/usr/bin/env python3
"""
Simple script to list all available audio input devices
"""
import pyaudio

def list_audio_devices():
    pya = pyaudio.PyAudio()
    
    print("🎤 Available Audio Input Devices:")
    print("=" * 50)
    
    for i in range(pya.get_device_count()):
        try:
            device_info = pya.get_device_info_by_index(i)
            device_name = device_info['name']
            input_channels = device_info['maxInputChannels']
            output_channels = device_info['maxOutputChannels']
            
            if input_channels > 0:  # Only show input devices
                print(f"Device {i}: {device_name}")
                print(f"  📥 Input channels: {input_channels}")
                print(f"  📤 Output channels: {output_channels}")
                print(f"  📊 Sample rate: {device_info['defaultSampleRate']}")
                
                # Check if this looks like a headphone mic
                if any(keyword in device_name.lower() for keyword in ['headphone', 'headset', 'bluetooth', 'bt', 'wireless']):
                    print(f"  🎧 ** HEADPHONE MICROPHONE DETECTED **")
                
                print()
                
        except Exception as e:
            print(f"Error checking device {i}: {e}")
    
    # Also show the default device
    try:
        default_input = pya.get_default_input_device_info()
        print(f"🎯 Default Input Device: {default_input['name']} (Device {default_input['index']})")
    except:
        print("❌ No default input device found")
    
    pya.terminate()

if __name__ == "__main__":
    list_audio_devices()
