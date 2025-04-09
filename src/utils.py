import re


def extract_json_block(text: str) -> str:
    """Extract the JSON block from the text"""
    # Match code blocks with any language identifier and extract just the content
    pattern = r'```json\n([\s\S]*?)\n```'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1).strip()
    return text.strip()