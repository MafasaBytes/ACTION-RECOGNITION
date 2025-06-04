import logging
import sys

def setup_logger(name='anomaly_detection', level=logging.DEBUG):
    """Sets up a basic logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger already configured
    if logger.hasHandlers():
        return logger

    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stdout_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)

    return logger

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    logger = setup_logger()
    logger.info("Logger initialized successfully.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.") 