import pytest

from utils import extract_lang_block


def test_extract_lang_block_simple():
    """Test extracting a simple code block."""
    text = """Some text
```json
{"key": "value"}
```
More text"""
    result = extract_lang_block(text, language="json")
    assert result == '{"key": "value"}'


def test_extract_lang_block_empty():
    """Test with empty input."""
    assert extract_lang_block(None) == ""
    assert extract_lang_block("") == ""


def test_extract_lang_block_no_blocks():
    """Test with text that has no code blocks."""
    text = "This is just plain text with no code blocks."
    result = extract_lang_block(text, language="json")
    assert result == text.strip()

    # Test with ensure_block=True should raise an error
    with pytest.raises(ValueError):
        extract_lang_block(text, language="json", ensure_block=True)


def test_extract_lang_block_nested_different_languages():
    """Test extraction with nested blocks of different languages."""
    text = """Here's a python block with nested JSON:
```python
def get_data():
    # This is a nested JSON example
    data = ```json
    {
        "key": "value",
        "nested": true
    }
    ```
    return data
```
"""
    # Should get the Python block when requesting Python
    python_result = extract_lang_block(text, language="python")
    assert "def get_data():" in python_result
    assert '"key": "value"' in python_result

    # Should get the JSON block when requesting JSON
    json_result = extract_lang_block(text, language="json")
    assert '"key": "value"' in json_result
