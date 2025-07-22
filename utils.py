import json
import re


def robust_json_parse(content: str):
    """
    Robustly parse JSON from LLM output, handling common formatting issues.
    """
    # Try to extract JSON block
    if '```json' in content:
        content = content.split('```json')[1].split('```')[0]
    elif '```' in content:
        content = content.split('```')[1].split('```')[0]
    # Fallback: extract first {...} block
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    # Try normal parsing
    try:
        return json.loads(content)
    except Exception:
        # Attempt to fix common issues (e.g., trailing commas)
        fixed = re.sub(r',\s*([}\]])', r'\1', content)
        try:
            return json.loads(fixed)
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": content}
