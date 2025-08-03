from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import Field

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.validation import UrlField
from ai_agent.registry.views import ActionResult


class HttpRequestParams(ActionParameter):
	"""Parameters for making HTTP requests"""
	url: str = UrlField(description='URL to make request to')
	method: Literal['GET', 'POST', 'PUT', 'DELETE', 'PATCH'] = Field(default='GET', description='HTTP method')
	headers: dict[str, str] = Field(default_factory=dict, description='HTTP headers')
	data: dict[str, Any] | None = Field(None, description='Request body data (will be JSON encoded)')
	timeout: float = Field(default=30.0, description='Request timeout in seconds', ge=1, le=300)


class HttpGetParams(ActionParameter):
	"""Simplified parameters for GET requests"""
	url: str = UrlField(description='URL to fetch')
	headers: dict[str, str] = Field(default_factory=dict, description='HTTP headers')


class HttpRequestAction(BaseAction[HttpRequestParams, ActionResult, Any]):
	"""Action to make HTTP requests"""
	
	name = 'http_request'
	description = 'Make HTTP requests to APIs or web services'
	category = ActionCategory.API_NAVIGATION
	tags = {ActionTag.NETWORK_REQUIRED, ActionTag.RETRY_SAFE}
	
	async def execute(self, parameters: HttpRequestParams, context: ActionContext[Any]) -> ActionResult:
		try:
			async with httpx.AsyncClient() as client:
				# Prepare request
				kwargs = {
					'method': parameters.method,
					'url': parameters.url,
					'headers': parameters.headers,
					'timeout': parameters.timeout,
				}
				
				if parameters.data is not None:
					if parameters.method in ['POST', 'PUT', 'PATCH']:
						kwargs['json'] = parameters.data
						if 'Content-Type' not in parameters.headers:
							kwargs['headers']['Content-Type'] = 'application/json'
				
				# Make request
				response = await client.request(**kwargs)
				
				# Parse response
				content_type = response.headers.get('content-type', '')
				if 'application/json' in content_type:
					try:
						response_data = response.json()
						extracted_content = json.dumps(response_data, indent=2)
					except:
						response_data = None
						extracted_content = response.text
				else:
					response_data = None
					extracted_content = response.text
				
				return ActionResult(
					success=response.is_success,
					extracted_content=extracted_content,
					output_data=response_data,
					metadata={
						'status_code': response.status_code,
						'headers': dict(response.headers),
						'content_type': content_type,
						'content_length': len(response.content),
					},
					error=f'HTTP {response.status_code}' if not response.is_success else None,
				)
				
		except httpx.TimeoutException:
			return ActionResult(
				success=False,
				error=f'Request timed out after {parameters.timeout} seconds',
			)
		except httpx.ConnectError:
			return ActionResult(
				success=False,
				error='Failed to connect to server',
			)
		except Exception as e:
			return ActionResult(
				success=False,
				error=f'HTTP request failed: {str(e)}',
			)


class HttpGetAction(BaseAction[HttpGetParams, ActionResult, Any]):
	"""Simplified action for GET requests"""
	
	name = 'http_get'
	description = 'Fetch content from a URL using GET request'
	category = ActionCategory.API_NAVIGATION
	tags = {ActionTag.SAFE, ActionTag.READ_ONLY, ActionTag.NETWORK_REQUIRED, ActionTag.RETRY_SAFE}
	priority = 5  # Higher priority than generic http_request for GET operations
	
	async def execute(self, parameters: HttpGetParams, context: ActionContext[Any]) -> ActionResult:
		# Delegate to HttpRequestAction
		request_params = HttpRequestParams(
			url=parameters.url,
			method='GET',
			headers=parameters.headers,
		)
		
		http_action = HttpRequestAction()
		return await http_action.execute(request_params, context)