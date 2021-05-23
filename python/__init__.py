import logging

formatter = logging.Formatter(
    fmt='%(levelname)s: %(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
#logger.addHandler(handler)