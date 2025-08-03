class ModelError(Exception):
	pass


class ModelProviderError(ModelError):
	def __init__(
		self,
		message: str,
		status_code: int = 502,
		model: str | None = None,
	):
		super().__init__(message, status_code)
		self.model = model


class ModelRateLimitError(ModelProviderError):
	def __init__(
		self,
		message: str,
		status_code: int = 429,
		model: str | None = None,
	):
		super().__init__(message, status_code, model)