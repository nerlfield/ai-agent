import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from ai_agent.utils.config import CONFIG


def addLoggingLevel(levelName, levelNum, methodName=None):
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError(f'{levelName} already defined in logging module')
	if hasattr(logging, methodName):
		raise AttributeError(f'{methodName} already defined in logging module')
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError(f'{methodName} already defined in logger class')

	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)


def setup_logging(stream=None, log_level=None, force_setup=False):
	try:
		addLoggingLevel('RESULT', 35)
	except AttributeError:
		pass

	log_type = log_level or CONFIG.AI_AGENT_LOGGING_LEVEL

	if logging.getLogger().hasHandlers() and not force_setup:
		return logging.getLogger('ai_agent')

	# Clear existing handlers
	root_logger = logging.getLogger()
	for handler in root_logger.handlers[:]:
		root_logger.removeHandler(handler)

	# Set up new handler
	handler = logging.StreamHandler(stream or sys.stdout)
	
	# Set log level
	level_map = {
		'debug': logging.DEBUG,
		'info': logging.INFO,
		'warning': logging.WARNING,
		'error': logging.ERROR,
		'critical': logging.CRITICAL,
	}
	
	level = level_map.get(log_type.lower(), logging.INFO)
	
	# Configure formatter
	if log_type.lower() == 'debug':
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
	else:
		formatter = logging.Formatter('%(message)s')
	
	handler.setFormatter(formatter)
	handler.setLevel(level)
	
	root_logger.addHandler(handler)
	root_logger.setLevel(level)
	
	# Suppress noisy third-party loggers
	for noisy_logger in ['httpx', 'httpcore', 'openai', 'anthropic']:
		logging.getLogger(noisy_logger).setLevel(logging.WARNING)
	
	return logging.getLogger('ai_agent')