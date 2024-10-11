# test_coreset_sc.py
# test


def test_sum_as_string():
    """Test the sum_as_string function from the coreset_sc module."""
    # Import the Rust-based module
    import coreset_sc

    # Call the function
    result = coreset_sc.sum_as_string(3, 4)

    # Assert that the result is the expected string
    assert result == "7", f"Expected '7' but got {result}"
