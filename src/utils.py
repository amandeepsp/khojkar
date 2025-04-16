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


def remove_thinking_output(text: str) -> str:
    """Remove the thinking output from the text
    e.g. <think> </think>
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
