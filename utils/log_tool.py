import logging
import logging.handlers as handlers
import utils.file_tool as file_tool
import math
import sys


def set_standard_format():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def set_clear_format():
    logging.basicConfig(format="")


def get_logger(name=None, filename=None, log_format=None, propagate=False):
    if name is None:
        name = 'global_logger'
    if name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if filename is None:
            filename = file_tool.PathManager.model_running_data_log_file
        file_handler = handlers.RotatingFileHandler(filename=filename, maxBytes=math.pow(2,30), backupCount=10)
        console_handler = logging.StreamHandler(stream=sys.stderr)

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        if log_format == None:
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            # console_handler.setFormatter(
            #     logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            file_handler.setFormatter(log_format)
            console_handler.setFormatter(log_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = propagate
    else:
        logger = logging.getLogger(name)

    return logger


def get_model_result_logger(name=None, filename=None):
    return get_logger(name, filename)


# model_result_logger = get_model_result_logger('model_result_logger')
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)