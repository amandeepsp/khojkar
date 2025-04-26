from chunking.recursive import RecursiveChunker


def test_recursive_chunker_small_text():
    """Test chunking with text smaller than chunk_size."""
    chunker = RecursiveChunker(chunk_size=1000)
    text = "This is a small text."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_recursive_chunker_large_text():
    """Test chunking with text larger than chunk_size."""
    chunker = RecursiveChunker(chunk_size=20, separators=[" ", ""], chunk_overlap=0)
    text = "This is a longer text that should be split into multiple chunks."
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Each chunk should be smaller than or equal to chunk_size
    for chunk in chunks:
        assert len(chunk) <= 20


def test_recursive_chunker_very_large_text():
    """Test chunking with text larger than chunk_size."""
    chunker = RecursiveChunker(chunk_size=100, separators=[" ", ""], chunk_overlap=0)
    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Vitae suscipit tellus mauris a diam maecenas sed.
    Amet nulla facilisi morbi tempus iaculis urna id volutpat lacus. Mattis rhoncus urna neque viverra justo nec ultrices. Non arcu risus quis varius quam quisque id diam.
    Semper feugiat nibh sed pulvinar proin gravida. Vestibulum morbi blandit cursus risus at ultrices mi. Aliquam etiam erat velit scelerisque in dictum non consectetur a.
    Laoreet id donec ultrices tincidunt arcu non sodales neque sodales. Orci dapibus ultrices in iaculis nunc sed augue. Dolor magna eget est lorem ipsum dolor sit amet consectetur.
    Mi bibendum neque egestas congue quisque egestas diam in arcu. Sit amet nisl purus in mollis. Gravida dictum fusce ut placerat orci nulla pellentesque dignissim.
    Pulvinar neque laoreet suspendisse interdum consectetur. Amet facilisis magna etiam tempor orci eu. Ipsum dolor sit amet consectetur adipiscing elit pellentesque habitant. Enim ut tellus elementum sagittis vitae et leo duis.
    Non tellus orci ac auctor augue mauris augue neque. Curabitur vitae nunc sed velit dignissim sodales ut eu.
    """
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Each chunk should be smaller than or equal to chunk_size
    for chunk in chunks:
        assert len(chunk) <= 100


def test_recursive_chunker_with_different_separators():
    """Test chunking with different separators."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    # Test with paragraph separator
    chunker1 = RecursiveChunker(chunk_size=20, separators=["\n\n"], chunk_overlap=0)
    chunks1 = chunker1.chunk(text)
    assert len(chunks1) == 3

    # Test with sentence separator
    chunker2 = RecursiveChunker(chunk_size=20, separators=["."], chunk_overlap=0)
    chunks2 = chunker2.chunk(text)
    assert len(chunks2) == 3


def test_recursive_chunker_empty_text():
    """Test chunking with empty text."""
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk("")
    assert len(chunks) == 1
    assert chunks[0] == ""


def test_recursive_chunker_no_separators():
    """Test chunking when no separators match."""
    chunker = RecursiveChunker(
        chunk_size=10, separators=["|||"], chunk_overlap=0
    )  # Unlikely separator
    text = "This text has no matching separators."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_basic_overlap():
    """Test basic chunk overlap functionality."""
    # Create a chunker with chunk_size=20 and chunk_overlap=5
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5, separators=[" "])

    # Create a text that will be split into multiple chunks
    text = "This is a test text that should be split into multiple chunks with overlap."

    chunks = chunker.chunk(text)

    # Check that we have multiple chunks
    assert len(chunks) > 1

    # Check that there's overlap between adjacent chunks
    for i in range(len(chunks) - 1):
        # The end of the current chunk should be the same as the beginning of the next chunk
        current_chunk_end = chunks[i][-5:] if len(chunks[i]) >= 5 else chunks[i]
        next_chunk_start = (
            chunks[i + 1][:5] if len(chunks[i + 1]) >= 5 else chunks[i + 1]
        )

        assert current_chunk_end == next_chunk_start, (
            f"No overlap between chunks {i} and {i + 1}"
        )


def test_overlap_size():
    """Test that the overlap size is respected."""
    # Test with different overlap sizes
    for overlap_size in [5, 10, 15]:
        chunker = RecursiveChunker(
            chunk_size=30, chunk_overlap=overlap_size, separators=[" "]
        )

        # Create a long text
        text = "word " * 50  # Creates a text with 50 "word " sequences

        chunks = chunker.chunk(text)

        # Verify overlap between chunks
        for i in range(len(chunks) - 1):
            # Get the end of current chunk and start of next chunk based on overlap size
            current_end = (
                chunks[i][-overlap_size:]
                if len(chunks[i]) >= overlap_size
                else chunks[i]
            )
            next_start = (
                chunks[i + 1][:overlap_size]
                if len(chunks[i + 1]) >= overlap_size
                else chunks[i + 1]
            )

            assert current_end == next_start, (
                f"Overlap size {overlap_size} not respected"
            )


def test_overlap_with_different_separators():
    """Test overlap works with different separator types."""
    # Test with paragraph separator
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5, separators=["\n\n"])
    text = (
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
    )

    chunks = chunker.chunk(text)

    # Check overlap between chunks
    for i in range(len(chunks) - 1):
        current_end = chunks[i][-5:] if len(chunks[i]) >= 5 else chunks[i]
        next_start = chunks[i + 1][:5] if len(chunks[i + 1]) >= 5 else chunks[i + 1]

        assert current_end == next_start, "No overlap with paragraph separator"


def test_overlap_edge_cases():
    """Test overlap with edge cases."""
    # Case 1: Text slightly larger than chunk_size
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=3, separators=[" "])
    text = "word1 word2"  # 12 characters, just above chunk_size

    chunks = chunker.chunk(text)
    assert len(chunks) > 1

    # Check overlap
    if len(chunks) > 1:
        assert chunks[0][-3:] == chunks[1][:3]

    # Case 2: Text exactly at chunk_size + overlap
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=5, separators=[""])
    text = "abcdefghijklmno"  # 15 characters (chunk_size + overlap)

    chunks = chunker.chunk(text)
    # This should result in two chunks with overlap
    assert len(chunks) == 2
    assert chunks[0][-5:] == chunks[1][:5]


def test_reconstructing_text_with_overlap():
    """Test that original text can be reconstructed from overlapping chunks."""
    chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5, separators=[" "])
    original_text = (
        "This is a test text that should be split into multiple chunks with overlap."
    )

    chunks = chunker.chunk(original_text)

    # Reconstruct the text by removing overlapping parts
    reconstructed_text = chunks[0]
    for i in range(1, len(chunks)):
        # Remove the overlapping part (first chunk_overlap characters) from subsequent chunks
        reconstructed_text += chunks[i][5:]

    assert reconstructed_text == original_text
