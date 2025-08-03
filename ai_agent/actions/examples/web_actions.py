"""
Example web interaction actions demonstrating the action framework.

These actions showcase web browser interactions with proper validation,
error handling, and result formatting.
"""

from typing import ClassVar

from pydantic import Field, validator

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.results import EnhancedActionResult, ResultCategory
from ai_agent.actions.validation import NonEmptyStringField, UrlField, validate_url


class WebActionContext(ActionContext):
	"""Context for web browser operations"""
	
	def __init__(self, **kwargs):
		super().__init__(context_type="web_browser", **kwargs)
		self.capabilities.update({"navigation", "element_interaction", "javascript"})
		
		# Mock browser resources
		self.resources["browser"] = MockBrowser()
		self.resources["current_page"] = None
		self.resources["element_cache"] = {}
	
	def get_browser(self) -> 'MockBrowser':
		"""Get the browser instance"""
		return self.resources["browser"]


class MockBrowser:
	"""Mock browser for demonstration purposes"""
	
	def __init__(self):
		self.current_url = ""
		self.page_title = ""
		self.elements = {}
		self.is_loaded = False
	
	async def navigate(self, url: str) -> dict:
		"""Navigate to a URL"""
		self.current_url = url
		self.page_title = f"Page for {url}"
		self.is_loaded = True
		
		# Simulate some page elements
		self.elements = {
			"login_button": {"type": "button", "text": "Login", "visible": True},
			"search_input": {"type": "input", "placeholder": "Search...", "visible": True},
			"nav_menu": {"type": "nav", "text": "Navigation", "visible": True}
		}
		
		return {
			"url": self.current_url,
			"title": self.page_title,
			"status": "loaded"
		}
	
	async def click_element(self, selector: str) -> dict:
		"""Click an element by selector"""
		if selector not in self.elements:
			raise ValueError(f"Element not found: {selector}")
		
		element = self.elements[selector]
		if not element.get("visible", False):
			raise ValueError(f"Element not visible: {selector}")
		
		# Simulate click effects
		if "button" in selector:
			return {"clicked": True, "element": selector, "action": "button_click"}
		elif "link" in selector:
			return {"clicked": True, "element": selector, "action": "navigation"}
		else:
			return {"clicked": True, "element": selector, "action": "generic_click"}
	
	async def type_text(self, selector: str, text: str) -> dict:
		"""Type text into an element"""
		if selector not in self.elements:
			raise ValueError(f"Element not found: {selector}")
		
		element = self.elements[selector]
		if element.get("type") != "input":
			raise ValueError(f"Element is not an input: {selector}")
		
		# Simulate typing
		element["value"] = text
		
		return {
			"typed": True,
			"element": selector,
			"text": text,
			"length": len(text)
		}


class NavigateParameters(ActionParameter):
	"""Parameters for navigating to a URL"""
	
	url: str = UrlField(
		description="URL to navigate to",
		examples=["https://example.com", "https://google.com", "https://github.com"]
	)
	
	wait_for_load: bool = Field(
		default=True,
		description="Whether to wait for the page to fully load"
	)
	
	timeout_seconds: float = Field(
		default=30.0,
		description="Maximum time to wait for page load",
		ge=1.0,
		le=120.0
	)


class ClickElementParameters(ActionParameter):
	"""Parameters for clicking an element"""
	
	selector: str = NonEmptyStringField(
		description="CSS selector or element identifier to click",
		examples=["#login-button", ".nav-link", "button[type=submit]"]
	)
	
	wait_visible: bool = Field(
		default=True,
		description="Whether to wait for element to be visible before clicking"
	)
	
	timeout_seconds: float = Field(
		default=10.0,
		description="Maximum time to wait for element",
		ge=1.0,
		le=60.0
	)


class TypeTextParameters(ActionParameter):
	"""Parameters for typing text into an element"""
	
	selector: str = NonEmptyStringField(
		description="CSS selector of the input element",
		examples=["#username", "input[name=email]", ".search-input"]
	)
	
	text: str = Field(
		description="Text to type into the element",
		examples=["user@example.com", "search query", "password123"]
	)
	
	clear_first: bool = Field(
		default=True,
		description="Whether to clear existing text before typing"
	)
	
	typing_delay_ms: float = Field(
		default=50.0,
		description="Delay between keystrokes in milliseconds",
		ge=0.0,
		le=1000.0
	)


class NavigateAction(BaseAction[NavigateParameters, EnhancedActionResult, WebActionContext]):
	"""Action to navigate to a web page"""
	
	name: ClassVar[str] = "navigate"
	description: ClassVar[str] = "Navigate to a web page URL"
	category: ClassVar[str] = ActionCategory.NAVIGATION
	tags: ClassVar[set[str]] = {ActionTag.WEB, ActionTag.REQUIRES_NETWORK, ActionTag.SAFE}
	
	async def execute(
		self,
		parameters: NavigateParameters,
		context: WebActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute navigation"""
		try:
			browser = context.get_browser()
			
			# Perform navigation
			result = await browser.navigate(parameters.url)
			
			# Update context
			context.resources["current_page"] = result
			context.update_state({"current_url": parameters.url})
			
			return EnhancedActionResult.success_with_data(
				data=result,
				result_type="navigation",
				summary=f"Successfully navigated to {parameters.url}",
				attachments={
					"page_info": {
						"url": result["url"],
						"title": result["title"],
						"status": result["status"]
					}
				}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Navigation failed: {str(e)}",
				category=ResultCategory.NETWORK,
				context={"url": parameters.url, "timeout": parameters.timeout_seconds}
			)


class ClickElementAction(BaseAction[ClickElementParameters, EnhancedActionResult, WebActionContext]):
	"""Action to click an element on a web page"""
	
	name: ClassVar[str] = "click_element"
	description: ClassVar[str] = "Click an element on the current web page"
	category: ClassVar[str] = ActionCategory.INTERACTION
	tags: ClassVar[set[str]] = {ActionTag.WEB, ActionTag.CLICK, ActionTag.SIDE_EFFECTS}
	
	async def execute(
		self,
		parameters: ClickElementParameters,
		context: WebActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute element click"""
		try:
			browser = context.get_browser()
			
			# Check if page is loaded
			if not browser.is_loaded:
				return EnhancedActionResult.error_with_details(
					"No page loaded - navigate to a page first",
					category=ResultCategory.VALIDATION,
					context={"selector": parameters.selector}
				)
			
			# Perform click
			result = await browser.click_element(parameters.selector)
			
			# Update context with interaction
			context.update_state({
				"last_interaction": {
					"type": "click",
					"selector": parameters.selector,
					"timestamp": __import__('time').time()
				}
			})
			
			return EnhancedActionResult.success_with_data(
				data=result,
				result_type="element_interaction",
				summary=f"Successfully clicked element: {parameters.selector}",
				attachments={
					"interaction_info": {
						"selector": parameters.selector,
						"action_type": result.get("action", "click"),
						"element_type": result.get("element_type")
					}
				}
			)
		
		except ValueError as e:
			return EnhancedActionResult.error_with_details(
				str(e),
				category=ResultCategory.VALIDATION,
				context={"selector": parameters.selector}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Click failed: {str(e)}",
				category=ResultCategory.EXECUTION,
				context={"selector": parameters.selector}
			)


class TypeTextAction(BaseAction[TypeTextParameters, EnhancedActionResult, WebActionContext]):
	"""Action to type text into an input element"""
	
	name: ClassVar[str] = "type_text"
	description: ClassVar[str] = "Type text into an input element on the web page"
	category: ClassVar[str] = ActionCategory.INTERACTION
	tags: ClassVar[set[str]] = {ActionTag.WEB, ActionTag.TYPE, ActionTag.SIDE_EFFECTS}
	
	async def execute(
		self,
		parameters: TypeTextParameters,
		context: WebActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute text typing"""
		try:
			browser = context.get_browser()
			
			# Check if page is loaded
			if not browser.is_loaded:
				return EnhancedActionResult.error_with_details(
					"No page loaded - navigate to a page first",
					category=ResultCategory.VALIDATION,
					context={"selector": parameters.selector}
				)
			
			# Perform typing
			result = await browser.type_text(parameters.selector, parameters.text)
			
			# Update context with interaction
			context.update_state({
				"last_interaction": {
					"type": "type",
					"selector": parameters.selector,
					"text_length": len(parameters.text),
					"timestamp": __import__('time').time()
				}
			})
			
			# Mask sensitive data in attachments
			display_text = parameters.text
			if any(keyword in parameters.selector.lower() for keyword in ['password', 'secret', 'token']):
				display_text = "*" * len(parameters.text)
			
			return EnhancedActionResult.success_with_data(
				data=result,
				result_type="text_input",
				summary=f"Successfully typed {len(parameters.text)} characters into {parameters.selector}",
				attachments={
					"interaction_info": {
						"selector": parameters.selector,
						"text_length": len(parameters.text),
						"display_text": display_text[:50] + "..." if len(display_text) > 50 else display_text
					}
				}
			)
		
		except ValueError as e:
			return EnhancedActionResult.error_with_details(
				str(e),
				category=ResultCategory.VALIDATION,
				context={"selector": parameters.selector}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Typing failed: {str(e)}",
				category=ResultCategory.EXECUTION,
				context={"selector": parameters.selector, "text_length": len(parameters.text)}
			)