#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone Audio Transcription Script
Transcribe existing WAV files using Gemini API or Whisper

Usage:
    python transcribe_audio.py <audio_file.wav>
    python transcribe_audio.py <session_directory>
    python transcribe_audio.py --all-sessions
    python transcribe_audio.py <file.wav> --language bg
"""

import sys
import os
import json
import argparse
import base64
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        import codecs
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        else:
            # Fallback for older Python versions
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        # If reconfiguration fails, continue without UTF-8 (emojis may not display)
        pass

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
        return True
    return False

# Load .env file
env_loaded = load_env_file()
if env_loaded:
    print("✅ Loaded environment variables from .env file")

# Check for Gemini API
TRANSCRIPTION_METHOD = None
TRANSCRIPTION_AVAILABLE = False
GEMINI_API_AVAILABLE = False

try:
    from google import genai
    if os.getenv('GEMINI_API_KEY'):
        GEMINI_API_AVAILABLE = True
        TRANSCRIPTION_METHOD = "gemini_api"
        TRANSCRIPTION_AVAILABLE = True
except ImportError:
    pass

# Simplified AudioTranscriber for standalone use
class AudioTranscriber:
    """Simplified audio transcriber using Gemini API"""
    
    def __init__(self):
        """Initialize the transcriber"""
        self.transcription_method = TRANSCRIPTION_METHOD
        self.available = TRANSCRIPTION_AVAILABLE
        
        if self.transcription_method == "gemini_api":
            try:
                from google import genai
                self.gemini_client = genai.Client(
                    api_key=os.getenv('GEMINI_API_KEY'),
                    http_options={"api_version": "v1"}  # Explicitly use v1 API for gemini-1.5-flash
                )
            except Exception as e:
                print(f"❌ Failed to initialize Gemini client: {e}")
                self.available = False
    
    def transcribe_audio_file(self, audio_path: str, language: str = None, is_conversation: bool = False) -> dict:
        """Transcribe audio file using Gemini API
        
        Args:
            audio_path: Path to the audio file (WAV or MP3)
            language: Optional language hint (e.g., 'bg', 'en', 'ro')
            is_conversation: If True, indicates this is a full conversation between agent and user
        
        Returns:
            dict with keys: text, success, error (if any)
        """
        if not self.available:
            return {
                "text": "",
                "error": "No transcription method available",
                "success": False
            }
        
        try:
            if not Path(audio_path).exists():
                return {"text": "", "error": "File not found", "success": False}
            
            # Read audio file
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Determine MIME type based on file extension
            file_ext = Path(audio_path).suffix.lower()
            mime_type = "audio/wav" if file_ext == ".wav" else "audio/mp3"
            
            # Prepare audio for Gemini
            audio_file_part = {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(audio_data).decode('utf-8')
                }
            }
            
            # Create prompt based on context
            if is_conversation:
                # For full conversation recordings (PBX)
                if language:
                    lang_names = {
                        'bg': 'Bulgarian', 'en': 'English', 'ro': 'Romanian',
                        'el': 'Greek', 'de': 'German', 'fr': 'French',
                        'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian'
                    }
                    lang_name = lang_names.get(language, language)
                    prompt = f"""We are passing 1 audio file and this is a conversation between agent and user. The language is {lang_name}. 

Please transcribe the call and return it as a JSON array where each element represents a turn in the conversation with the speaker identified.

Return ONLY a JSON array in this exact format:
[
  {{"speaker": "agent", "text": "what the agent said"}},
  {{"speaker": "user", "text": "what the user said"}},
  {{"speaker": "agent", "text": "agent's next message"}},
  ...
]

Important:
- "speaker" must be either "agent" or "user"
- "text" is what was said
- Return ONLY the JSON array, nothing else
- The agent is the AI assistant making the call
- The user is the person receiving the call"""
                else:
                    prompt = """We are passing 1 audio file and this is a conversation between agent and user.

Please transcribe the call and return it as a JSON array where each element represents a turn in the conversation with the speaker identified.

Return ONLY a JSON array in this exact format:
[
  {"speaker": "agent", "text": "what the agent said"},
  {"speaker": "user", "text": "what the user said"},
  {"speaker": "agent", "text": "agent's next message"},
  ...
]

Important:
- "speaker" must be either "agent" or "user"
- "text" is what was said
- Return ONLY the JSON array, nothing else
- The agent is the AI assistant making the call
- The user is the person receiving the call"""
            else:
                # For individual channel recordings (incoming/outgoing/mixed)
                if language:
                    lang_names = {
                        'bg': 'Bulgarian', 'en': 'English', 'ro': 'Romanian',
                        'el': 'Greek', 'de': 'German', 'fr': 'French',
                        'es': 'Spanish', 'it': 'Italian', 'ru': 'Russian'
                    }
                    lang_name = lang_names.get(language, language)
                    prompt = f"Please transcribe this audio in {lang_name}. Provide only the transcription text, nothing else."
                else:
                    prompt = "Please transcribe this audio. Provide only the transcription text, nothing else."
            
            # Use Gemini 2.5 Flash model (paid, supports audio transcription)
            model_id = "gemini-2.5-flash"
            
            # Generate transcription
            response = self.gemini_client.models.generate_content(
                model=model_id,
                contents=[prompt, audio_file_part]
            )
            
            # Extract text
            transcript_text = ""
            conversation_data = None
            
            if hasattr(response, 'text'):
                transcript_text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, 'text'):
                                transcript_text += part.text
            
            transcript_text = transcript_text.strip()
            
            # For conversation transcripts, try to parse JSON structure
            if is_conversation and transcript_text:
                try:
                    # Remove markdown code blocks if present
                    clean_text = transcript_text
                    if clean_text.startswith("```json"):
                        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                    elif clean_text.startswith("```"):
                        clean_text = clean_text.split("```")[1].split("```")[0].strip()
                    
                    # Try to parse as JSON
                    conversation_data = json.loads(clean_text)
                    
                    # Validate structure
                    if isinstance(conversation_data, list) and len(conversation_data) > 0:
                        # Create plain text version from JSON for backward compatibility
                        plain_text_parts = []
                        for turn in conversation_data:
                            speaker = turn.get('speaker', 'unknown')
                            text = turn.get('text', '')
                            plain_text_parts.append(f"{speaker.upper()}: {text}")
                        transcript_text = "\n".join(plain_text_parts)
                    else:
                        # Invalid structure, treat as plain text
                        conversation_data = None
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, keep as plain text
                    conversation_data = None
            
            result = {
                "text": transcript_text,
                "language": language or 'unknown',
                "segments": [],
                "confidence": None,
                "success": True,
                "method": "gemini_api"
            }
            
            # Add conversation structure if available
            if conversation_data:
                result["conversation"] = conversation_data
            
            return result
            
        except Exception as e:
            return {
                "text": "",
                "error": str(e),
                "success": False
            }


def print_banner():
    """Print a nice banner"""
    print("=" * 70)
    print("🎤 Audio Transcription Tool")
    print("=" * 70)
    if TRANSCRIPTION_AVAILABLE:
        print(f"✅ Transcription method: {TRANSCRIPTION_METHOD}")
    else:
        print("❌ No transcription method available!")
        print("💡 Install: pip install google-genai")
        sys.exit(1)
    print("=" * 70)
    print()


def format_transcript_result(result: dict) -> str:
    """Format transcription result for display"""
    output = []
    output.append("-" * 70)
    
    if result.get('success'):
        text = result.get('text', '')
        language = result.get('language', 'unknown')
        confidence = result.get('confidence')
        method = result.get('method', TRANSCRIPTION_METHOD)
        
        output.append(f"✅ Transcription successful!")
        output.append(f"   Method: {method}")
        output.append(f"   Language: {language}")
        output.append(f"   Length: {len(text)} characters")
        if confidence:
            output.append(f"   Confidence: {confidence:.3f}")
        output.append("")
        output.append("📝 Transcript:")
        output.append("-" * 70)
        output.append(text)
    else:
        error = result.get('error', 'Unknown error')
        output.append(f"❌ Transcription failed: {error}")
    
    output.append("-" * 70)
    return "\n".join(output)


def save_transcript_to_file(audio_path: Path, result: dict) -> Path:
    """Save transcript to a text file next to the audio file"""
    transcript_path = audio_path.with_suffix('.txt')
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"Audio File: {audio_path.name}\n")
        f.write(f"Transcribed: {datetime.now().isoformat()}\n")
        f.write(f"Method: {result.get('method', TRANSCRIPTION_METHOD)}\n")
        f.write(f"Language: {result.get('language', 'unknown')}\n")
        if result.get('confidence'):
            f.write(f"Confidence: {result.get('confidence'):.3f}\n")
        f.write("\n")
        f.write("-" * 70)
        f.write("\n")
        f.write(result.get('text', ''))
        f.write("\n")
    
    return transcript_path


def transcribe_single_file(audio_path: Path, language: str = None, save: bool = True, verbose: bool = True) -> dict:
    """Transcribe a single audio file"""
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return {"success": False, "error": "File not found"}
    
    if audio_path.suffix.lower() not in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
        print(f"⚠️ Warning: {audio_path.suffix} might not be supported. Best results with .wav files.")
    
    if verbose:
        print(f"🎙️ Transcribing: {audio_path}")
        if language:
            print(f"🎯 Language hint: {language}")
        print()
    
    # Create transcriber instance
    transcriber = AudioTranscriber()
    
    # Transcribe the file
    try:
        result = transcriber.transcribe_audio_file(str(audio_path), language=language)
        
        if verbose:
            print(format_transcript_result(result))
            print()
        
        # Save to file if requested
        if save and result.get('success'):
            transcript_path = save_transcript_to_file(audio_path, result)
            if verbose:
                print(f"💾 Transcript saved to: {transcript_path}")
                print()
        
        return result
        
    except Exception as e:
        error_msg = f"Error during transcription: {e}"
        if verbose:
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
        return {"success": False, "error": error_msg}


def transcribe_session_directory(session_dir: Path, language: str = None, save: bool = True) -> dict:
    """Transcribe all audio files in a session directory"""
    print(f"📁 Processing session directory: {session_dir}")
    print()
    
    # Look for audio files
    audio_files = {
        'incoming': None,
        'outgoing': None,
        'mixed': None
    }
    
    for file_path in session_dir.glob("*.wav"):
        filename = file_path.name.lower()
        if 'incoming' in filename:
            audio_files['incoming'] = file_path
        elif 'outgoing' in filename:
            audio_files['outgoing'] = file_path
        elif 'mixed' in filename:
            audio_files['mixed'] = file_path
    
    if not any(audio_files.values()):
        print(f"❌ No WAV files found in {session_dir}")
        return {}
    
    results = {}
    
    # Transcribe each file
    for audio_type, audio_path in audio_files.items():
        if audio_path:
            print(f"▶️ Transcribing {audio_type} audio...")
            result = transcribe_single_file(audio_path, language=language, save=save)
            results[audio_type] = result
            print()
    
    # Update session_info.json if it exists
    info_path = session_dir / "session_info.json"
    if info_path.exists() and save:
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
            
            session_info["transcripts"] = {}
            for audio_type, result in results.items():
                if result:
                    session_info["transcripts"][audio_type] = {
                        "success": result.get('success', False),
                        "language": result.get('language', 'unknown'),
                        "text_length": len(result.get('text', '')),
                        "confidence": result.get('confidence'),
                        "method": result.get('method', TRANSCRIPTION_METHOD),
                        "transcribed_at": datetime.now().isoformat()
                    }
                    if not result.get('success'):
                        session_info["transcripts"][audio_type]["error"] = result.get('error')
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Updated session info: {info_path}")
            print()
        except Exception as e:
            print(f"⚠️ Could not update session_info.json: {e}")
    
    return results


def transcribe_all_sessions(sessions_dir: Path = Path("sessions"), language: str = None):
    """Transcribe all sessions that don't have transcripts yet"""
    if not sessions_dir.exists():
        print(f"❌ Sessions directory not found: {sessions_dir}")
        return
    
    print(f"🔍 Scanning for sessions in: {sessions_dir}")
    print()
    
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print(f"❌ No session directories found in {sessions_dir}")
        return
    
    print(f"📊 Found {len(session_dirs)} session(s)")
    print()
    
    transcribed = 0
    skipped = 0
    failed = 0
    
    for i, session_dir in enumerate(session_dirs, 1):
        print(f"[{i}/{len(session_dirs)}] Processing: {session_dir.name}")
        
        # Check if transcripts already exist
        info_path = session_dir / "session_info.json"
        has_transcripts = False
        
        if info_path.exists():
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                    if session_info.get('transcripts'):
                        has_transcripts = True
            except:
                pass
        
        if has_transcripts:
            print(f"⏭️ Skipping - transcripts already exist")
            skipped += 1
            print()
            continue
        
        # Transcribe the session
        results = transcribe_session_directory(session_dir, language=language)
        
        if results and any(r.get('success') for r in results.values() if r):
            transcribed += 1
        else:
            failed += 1
        
        print("-" * 70)
        print()
    
    # Summary
    print("=" * 70)
    print("📊 Summary:")
    print(f"   ✅ Transcribed: {transcribed} session(s)")
    print(f"   ⏭️ Skipped: {skipped} session(s) (already had transcripts)")
    print(f"   ❌ Failed: {failed} session(s)")
    print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Gemini API or Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  python transcribe_audio.py recording.wav
  
  # Transcribe with language hint
  python transcribe_audio.py recording.wav --language bg
  
  # Transcribe a session directory
  python transcribe_audio.py sessions/abc-123-def-456/
  
  # Transcribe all sessions without transcripts
  python transcribe_audio.py --all-sessions
  
  # Don't save transcript files (just display)
  python transcribe_audio.py recording.wav --no-save
        """
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to audio file or session directory'
    )
    
    parser.add_argument(
        '--all-sessions',
        action='store_true',
        help='Transcribe all sessions in the sessions/ directory'
    )
    
    parser.add_argument(
        '--language', '-l',
        help='Language hint (e.g., bg, en, ro, el, de, fr, es, it, ru)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save transcript to file (just display)'
    )
    
    parser.add_argument(
        '--sessions-dir',
        default='sessions',
        help='Path to sessions directory (default: sessions/)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Less verbose output'
    )
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate arguments
    if not args.all_sessions and not args.path:
        parser.print_help()
        print("\n❌ Error: Please provide a path or use --all-sessions")
        sys.exit(1)
    
    save_transcripts = not args.no_save
    
    # Execute the requested action
    if args.all_sessions:
        sessions_dir = Path(args.sessions_dir)
        transcribe_all_sessions(sessions_dir, language=args.language)
    
    elif args.path:
        path = Path(args.path)
        
        if not path.exists():
            print(f"❌ Error: Path not found: {path}")
            sys.exit(1)
        
        if path.is_file():
            # Transcribe single file
            transcribe_single_file(
                path, 
                language=args.language, 
                save=save_transcripts,
                verbose=not args.quiet
            )
        
        elif path.is_dir():
            # Transcribe session directory
            transcribe_session_directory(
                path, 
                language=args.language, 
                save=save_transcripts
            )
        
        else:
            print(f"❌ Error: Invalid path: {path}")
            sys.exit(1)
    
    if not args.quiet:
        print("✅ Done!")


if __name__ == "__main__":
    main()

