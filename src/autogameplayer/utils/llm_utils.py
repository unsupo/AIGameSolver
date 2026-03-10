import json
import re
from typing import Optional, Dict, Any

def extract_json_from_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts and parses the first JSON object found in an LLM string."""
    if not text:
        return None
        
    # Try simple strip first
    text = text.strip()
    
    # Try to find JSON block using regex if there is markdown or noise
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # If the greedy match fails, try a non-greedy one in case there are multiple blocks
            json_match = re.search(r'\{.*?\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except (ValueError, json.JSONDecodeError):
                    pass

    # Fallback to finding first { and last }
    try:
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass
        
    return None
