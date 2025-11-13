import os
import sys
import google.generativeai as genai
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API - Use GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

MODEL_NAME = "gemini-2.5-flash"  # Use same model as transcription

genai.configure(api_key=API_KEY)

# Valid status options
VALID_STATUSES = [
    "interested",
    "not interested"
]

def read_specific_files(file_paths):
    """Read specific files provided as a list."""
    files_content = {}
    
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        try:
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            if not file_path.is_file():
                print(f"Warning: Not a file: {file_path}")
                continue
            
            # Skip very large files (> 5MB)
            if file_path.stat().st_size > 5_000_000:
                print(f"Warning: File too large, skipping: {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                files_content[file_path.name] = content
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return files_content

def read_files_from_directory(directory="."):
    """Read all text files from the specified directory."""
    files_content = {}
    directory_path = Path(directory)
    
    # Common text file extensions
    text_extensions = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', 
                      '.yaml', '.yml', '.xml', '.csv', '.log', '.rst', '.ini', 
                      '.cfg', '.conf', '.sh', '.bat', '.c', '.cpp', '.h', '.java',
                      '.cs', '.php', '.rb', '.go', '.rs', '.ts', '.jsx', '.tsx']
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in text_extensions:
            try:
                # Skip very large files (> 1MB)
                if file_path.stat().st_size > 1_000_000:
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = file_path.relative_to(directory_path)
                    files_content[str(relative_path)] = content
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    return files_content

def create_prompt(files_content, language="English"):
    """Create a prompt for the Gemini API."""
    files_text = ""
    for filename, content in files_content.items():
        files_text += f"\n\n=== FILE: {filename} ===\n{content}\n"
    
    prompt = f"""You are analyzing a phone call session with three transcriptions:
- incoming audio: the customer/user speaking
- outgoing audio: the voice agent speaking  
- mixed audio: both customer and voice agent speaking

Directly analyze the conversation and provide:
1. A comprehensive summary of what happened in this call - the purpose, topics discussed, outcomes, and any important details (write in a single paragraph without line breaks)
2. A status indicating the customer's level of interest:
   - "interested": if the customer was engaged, showed interest in the product/service, confirmed orders, asked positive questions, or had a successful outcome
   - "not interested": if the customer declined, rejected the offer, showed disinterest, hung up early, or had a negative outcome

The transcriptions are:
{files_text}

Please respond in the following JSON format:
{{
    "summary": "Your detailed analysis of the conversation here in a single paragraph",
    "status": "either interested OR not interested"
}}

IMPORTANT: 
- Analyze the actual conversation content, not the files themselves
- The status field MUST be exactly one of these two values: "interested" or "not interested"
- Evaluate the customer's engagement and outcome to determine the status
- Write the summary as a single flowing paragraph without line breaks
- Use lowercase only for status and match these values exactly
- Write the summary in {language} language"""
    
    return prompt

def clean_text(text):
    """Remove special characters from text."""
    # Remove specified special characters: " * / \ | ^
    special_chars = ['"', '*', '/', '\\', '|', '^']
    cleaned = text
    for char in special_chars:
        cleaned = cleaned.replace(char, '')
    return cleaned

def normalize_status(status):
    """Normalize the status to one of the valid options."""
    status_lower = status.lower().strip()
    
    # Direct match
    if status_lower in VALID_STATUSES:
        return status_lower
    
    # Try to map common variations
    if "not interested" in status_lower or "uninterested" in status_lower or "not" in status_lower:
        return "not interested"
    elif "interested" in status_lower:
        return "interested"
    
    # Default fallback - if uncertain, mark as not interested
    return "not interested"

def summarize_files(file_paths=None, directory=".", language="English"):
    """Main function to summarize files using Gemini API.
    
    Args:
        file_paths: List of specific file paths to analyze. If provided, directory is ignored.
        directory: Directory to scan for files (default: current directory). Only used if file_paths is None.
        language: Language for the summary (default: "English").
    """
    
    # Read files
    if file_paths:
        print(f"Reading {len(file_paths)} specific files...")
        files_content = read_specific_files(file_paths)
    else:
        print("Reading files from directory...")
        files_content = read_files_from_directory(directory)
    
    if not files_content:
        return {
            "summary": "No readable files found.",
            "status": "not interested"
        }
    
    print(f"Found {len(files_content)} files to analyze")
    
    # Create the model
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Create prompt
    prompt = create_prompt(files_content, language=language)
    
    # Generate response
    print("Analyzing files with Gemini API...")
    response = model.generate_content(prompt)
    
    # Parse response
    response_text = response.text.strip()
    
    # Try to extract JSON from the response
    try:
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1]
            response_text = response_text.split("```")[0]
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            response_text = response_text.split("```")[0]
        
        result = json.loads(response_text.strip())
        
        # Clean up the summary - replace newlines with spaces
        if "summary" in result:
            result["summary"] = result["summary"].replace('\n', ' ').replace('\r', ' ')
            # Remove multiple spaces
            result["summary"] = ' '.join(result["summary"].split())
            # Remove special characters
            result["summary"] = clean_text(result["summary"])
        
        # Normalize the status
        if "status" in result:
            result["status"] = normalize_status(result["status"])
        else:
            result["status"] = "not interested"
            
    except json.JSONDecodeError:
        # If JSON parsing fails, create a structured response
        cleaned_text = response_text.replace('\n', ' ').replace('\r', ' ')
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = clean_text(cleaned_text)
        result = {
            "summary": cleaned_text,
            "status": "not interested"
        }
    
    # Only return summary and status
    return {
        "summary": result.get("summary", ""),
        "status": result.get("status", "not interested")
    }


def summarize_from_content(transcript_content: str, language: str = "English") -> dict:
    """
    Generate summary directly from transcript content (for MongoDB transcripts)
    
    Args:
        transcript_content: The combined transcript text
        language: Target language for the summary
    
    Returns:
        Dict with 'summary' and 'status' keys
    """
    try:
        # Create files_content dict with single entry
        files_content = {"transcript": transcript_content}
        
        print(f"Analyzing transcript ({len(transcript_content)} characters) with Gemini API...")
        
        # Create the model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Create prompt
        prompt = create_prompt(files_content, language=language)
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Parse response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
                response_text = response_text.split("```")[0]
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                response_text = response_text.split("```")[0]
            
            result = json.loads(response_text.strip())
            
            # Clean up the summary - replace newlines with spaces
            if "summary" in result:
                result["summary"] = result["summary"].replace('\n', ' ').replace('\r', ' ')
                # Remove multiple spaces
                result["summary"] = ' '.join(result["summary"].split())
                # Remove special characters
                result["summary"] = clean_text(result["summary"])
            
            # Normalize the status
            if "status" in result:
                result["status"] = normalize_status(result["status"])
            else:
                result["status"] = "not interested"
                
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response
            cleaned_text = response_text.replace('\n', ' ').replace('\r', ' ')
            cleaned_text = ' '.join(cleaned_text.split())
            cleaned_text = clean_text(cleaned_text)
            result = {
                "summary": cleaned_text,
                "status": "not interested"
            }
        
        # Only return summary and status
        return {
            "summary": result.get("summary", ""),
            "status": result.get("status", "not interested")
        }
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {
            "summary": f"Error generating summary: {str(e)}",
            "status": "not interested"
        }


if __name__ == "__main__":
    # Parse command line arguments
    # Format: python file_summarizer.py [--language LANGUAGE] file1 file2 file3...
    language = "English"
    file_paths_args = []
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
            i += 2
        else:
            file_paths_args.append(sys.argv[i])
            i += 1
    
    if file_paths_args:
        # Specific files provided
        result = summarize_files(file_paths=file_paths_args, language=language)
    else:
        # No arguments, scan current directory
        result = summarize_files(language=language)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULT")
    print("="*60)
    print(f"\nLanguage: {language}")
    print(f"\nStatus: {result['status'].upper()}")
    print(f"\nSummary length: {len(result['summary'])} characters")
    print("(Summary content saved to JSON file)")
    print("\n" + "="*60)
    
    # Also save to JSON file
    with open("analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\nResult saved to: analysis_result.json")

