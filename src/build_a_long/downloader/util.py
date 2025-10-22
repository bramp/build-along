"""General utility functions for the downloader."""


def is_valid_set_id(set_id: str) -> bool:
    """
    Validates if a string is a valid LEGO set ID (only digits).
    """
    return set_id.isdigit()
