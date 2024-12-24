import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Кастомный форматтер для красивого вывода логов"""

    # Цвета для консоли
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Форматирование для разных уровней логов
    format_template = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_template + reset,
        logging.INFO: blue + format_template + reset,
        logging.WARNING: yellow + format_template + reset,
        logging.ERROR: red + format_template + reset,
        logging.CRITICAL: bold_red + format_template + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name='api_client'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    # Файловый обработчик
    file_handler = logging.FileHandler(f'api_{datetime.now().strftime("%Y%m%d")}.log')
    file_format = logging.Formatter(
        '%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger