def typical_string_processing(x: str) -> str:
    """
    This function removes accent marks, converts to lowercase, removes dots and trims the string

    Args:
        x: str, the string to process

    Returns:
        str, the processed string
    """
    tildes = "áéíóú"
    no_tildes = "aeiou"
    x = x.lower()
    x = x.strip()
    for i in range(len(tildes)):
        x = x.replace(tildes[i], no_tildes[i])
    return x
