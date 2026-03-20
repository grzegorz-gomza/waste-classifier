import os
import sys
import logging

logging_str = (
    "[%(asctime)s]: "  # Timestamp of when the log entry was created
    "%(levelname)s: "  # Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "%(module)s: "     # Name of the module (Python file) where the log message originated
    "%(funcName)s: "   # Name of the function from which the log message was generated
    "%(message)s"      # The actual log message that will be logged
)

log_dir = "log_msgs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, # Set the logging level
    format=logging_str, # Set the logging format
    handlers=[
        logging.FileHandler(log_filepath), # Log messages to a file
        logging.StreamHandler(sys.stdout) # Log messages to the console
    ]
)

logger = logging.getLogger("WasteClassifier")