import re


def extract_lang_block(
    text: str, language: str = "json", ensure_block: bool = False
) -> str:
    """Extract the JSON block from the text"""
    # Match code blocks with any language identifier and extract just the content
    pattern = rf"```{language}\n([\s\S]*?)\n```"
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    if ensure_block:
        raise ValueError(f"No {language} block found in text")

    return text.strip()
