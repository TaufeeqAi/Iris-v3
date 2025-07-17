import logging

def setup_logging(name: str):
    """Configures a standardized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# Example of a generic helper function
def validate_id(id_str: str) -> bool:
    """Validates if a string can be converted to an integer ID."""
    try:
        int(id_str)
        return True
    except ValueError:
        return False