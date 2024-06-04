import logging, sys
from warnings import simplefilter


def setup_logging(filename="executions.log"):
    format = "%(asctime)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=format)
    logger = logging.getLogger()
    consoleHandler = logging.FileHandler(filename)
    consoleHandler.setFormatter(logging.Formatter(format))
    logger.addHandler(consoleHandler)


simplefilter(action="ignore", category=FutureWarning)
