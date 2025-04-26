import re


def extract_lang_block(
    text: str | None, language: str = "json", ensure_block: bool = False
) -> str:
    """Extract the JSON block from the text"""
    if text is None:
        return ""

    # Match code blocks with any language identifier and extract just the content
    pattern = rf"```{language}\n([\s\S]*)\n```"
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    if ensure_block:
        raise ValueError(f"No {language} block found in text")

    return text.strip()


def remove_thinking_output(text: str | None) -> str:
    """Remove the thinking output from the text
    e.g. <think> </think>
    """
    if text is None:
        return ""

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
