import logging


class LoggingSetter:
    """
    A class for creatin a logger for the different modules in the proyect

    Parameters:
        name (str): The name of the module

    Attributes:
        name (str): The name of the module
    """

    def __init__(self, name):
        self.name = name

    def setting_log(self, log_name):
        """
        Sets the configuration for a log

        Returns:
            str: The logger
        """

        # Extract the first cabin | Extract the title from 'name'

        # Taking modudle's name
        logger = logging.getLogger(self.name)

        # Setting loggin level
        logger.setLevel(logging.DEBUG)

        # Setting logging format
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s')

        # Setting logging name
        file_handler = logging.FileHandler(log_name)

        # Configuring format in the file handler
        file_handler.setFormatter(formatter)

        # Adding the log file
        logger.addHandler(file_handler)

        return logger
