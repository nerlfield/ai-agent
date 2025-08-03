"""
Example data processing actions demonstrating the action framework.

These actions showcase data manipulation, database operations, and text processing
with proper validation, error handling, and result formatting.
"""

import json
import re
from typing import Any, ClassVar

from pydantic import Field, validator

from ai_agent.actions.base import ActionContext, ActionParameter, BaseAction
from ai_agent.actions.categories import ActionCategory, ActionTag
from ai_agent.actions.results import EnhancedActionResult, ResultCategory
from ai_agent.actions.validation import NonEmptyStringField, ChoiceRule


class DataActionContext(ActionContext):
	"""Context for data processing operations"""
	
	def __init__(self, **kwargs):
		super().__init__(context_type="data_processing", **kwargs)
		self.capabilities.update({"text_processing", "json_parsing", "database_query"})
		
		# Mock data resources
		self.resources["database"] = MockDatabase()
		self.resources["text_processor"] = MockTextProcessor()
		self.resources["data_cache"] = {}
	
	def get_database(self) -> 'MockDatabase':
		"""Get the database instance"""
		return self.resources["database"]
	
	def get_text_processor(self) -> 'MockTextProcessor':
		"""Get the text processor instance"""
		return self.resources["text_processor"]


class MockDatabase:
	"""Mock database for demonstration purposes"""
	
	def __init__(self):
		self.data = {
			"users": [
				{"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
				{"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
				{"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35}
			],
			"orders": [
				{"id": 1, "user_id": 1, "amount": 100.50, "status": "completed"},
				{"id": 2, "user_id": 2, "amount": 75.25, "status": "pending"},
				{"id": 3, "user_id": 1, "amount": 200.00, "status": "completed"}
			]
		}
	
	async def query(self, table: str, conditions: dict = None) -> list[dict]:
		"""Execute a simple query"""
		if table not in self.data:
			raise ValueError(f"Table not found: {table}")
		
		results = self.data[table].copy()
		
		if conditions:
			filtered_results = []
			for row in results:
				match = True
				for key, value in conditions.items():
					if key not in row or row[key] != value:
						match = False
						break
				if match:
					filtered_results.append(row)
			results = filtered_results
		
		return results
	
	async def count(self, table: str, conditions: dict = None) -> int:
		"""Count rows matching conditions"""
		results = await self.query(table, conditions)
		return len(results)


class MockTextProcessor:
	"""Mock text processor for demonstration purposes"""
	
	async def extract_emails(self, text: str) -> list[str]:
		"""Extract email addresses from text"""
		email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
		return re.findall(email_pattern, text)
	
	async def extract_urls(self, text: str) -> list[str]:
		"""Extract URLs from text"""
		url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
		return re.findall(url_pattern, text)
	
	async def word_count(self, text: str) -> dict[str, int]:
		"""Count words in text"""
		words = re.findall(r'\b\w+\b', text.lower())
		word_count = {}
		for word in words:
			word_count[word] = word_count.get(word, 0) + 1
		return word_count
	
	async def sentiment_analysis(self, text: str) -> dict[str, Any]:
		"""Simple sentiment analysis"""
		positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
		negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
		
		text_lower = text.lower()
		positive_count = sum(1 for word in positive_words if word in text_lower)
		negative_count = sum(1 for word in negative_words if word in text_lower)
		
		if positive_count > negative_count:
			sentiment = "positive"
			confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
		elif negative_count > positive_count:
			sentiment = "negative"
			confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
		else:
			sentiment = "neutral"
			confidence = 0.5
		
		return {
			"sentiment": sentiment,
			"confidence": confidence,
			"positive_words": positive_count,
			"negative_words": negative_count
		}


class ProcessTextParameters(ActionParameter):
	"""Parameters for text processing"""
	
	text: str = NonEmptyStringField(
		description="Text content to process",
		examples=["Hello world! Contact us at info@example.com", "Visit https://example.com for more info"]
	)
	
	operations: list[str] = Field(
		description="List of operations to perform on the text",
		examples=[["extract_emails"], ["word_count", "sentiment_analysis"], ["extract_urls", "word_count"]]
	)
	
	options: dict[str, Any] = Field(
		default_factory=dict,
		description="Additional options for text processing",
		examples=[{"min_word_length": 3}, {"case_sensitive": False}]
	)
	
	def validate_constraints(self) -> None:
		"""Validate processing operations"""
		valid_operations = {"extract_emails", "extract_urls", "word_count", "sentiment_analysis"}
		for operation in self.operations:
			if operation not in valid_operations:
				raise ValueError(f"Invalid operation: {operation}. Valid operations: {valid_operations}")


class QueryDatabaseParameters(ActionParameter):
	"""Parameters for database queries"""
	
	table: str = NonEmptyStringField(
		description="Name of the database table to query",
		examples=["users", "orders", "products"]
	)
	
	conditions: dict[str, Any] = Field(
		default_factory=dict,
		description="Conditions to filter the query results",
		examples=[{"status": "active"}, {"age": 25, "city": "New York"}]
	)
	
	limit: int = Field(
		default=100,
		description="Maximum number of results to return",
		ge=1,
		le=1000
	)
	
	operation: str = Field(
		default="select",
		description="Type of database operation to perform"
	)
	
	def validate_constraints(self) -> None:
		"""Validate database operation"""
		valid_operations = {"select", "count", "exists"}
		if self.operation not in valid_operations:
			raise ValueError(f"Invalid operation: {self.operation}. Valid operations: {valid_operations}")


class TransformDataParameters(ActionParameter):
	"""Parameters for data transformation"""
	
	data: Any = Field(
		description="Data to transform (JSON-serializable)",
		examples=[{"key": "value"}, [1, 2, 3], "string data"]
	)
	
	transformation: str = Field(
		description="Type of transformation to apply",
		examples=["to_json", "to_csv", "normalize", "aggregate"]
	)
	
	options: dict[str, Any] = Field(
		default_factory=dict,
		description="Transformation options",
		examples=[{"indent": 2}, {"delimiter": ","}, {"group_by": "category"}]
	)
	
	def validate_constraints(self) -> None:
		"""Validate transformation type"""
		valid_transformations = {"to_json", "to_csv", "normalize", "aggregate", "filter", "sort"}
		if self.transformation not in valid_transformations:
			raise ValueError(f"Invalid transformation: {self.transformation}")


class ProcessTextAction(BaseAction[ProcessTextParameters, EnhancedActionResult, DataActionContext]):
	"""Action to process text with various operations"""
	
	name: ClassVar[str] = "process_text"
	description: ClassVar[str] = "Process text with operations like email extraction, word counting, sentiment analysis"
	category: ClassVar[str] = ActionCategory.TEXT_PROCESSING
	tags: ClassVar[set[str]] = {ActionTag.TEXT, ActionTag.ANALYSIS, ActionTag.STATELESS}
	
	async def execute(
		self,
		parameters: ProcessTextParameters,
		context: DataActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute text processing"""
		try:
			processor = context.get_text_processor()
			results = {}
			
			for operation in parameters.operations:
				if operation == "extract_emails":
					results["emails"] = await processor.extract_emails(parameters.text)
				elif operation == "extract_urls":
					results["urls"] = await processor.extract_urls(parameters.text)
				elif operation == "word_count":
					results["word_count"] = await processor.word_count(parameters.text)
				elif operation == "sentiment_analysis":
					results["sentiment"] = await processor.sentiment_analysis(parameters.text)
			
			# Calculate processing statistics
			text_stats = {
				"character_count": len(parameters.text),
				"word_count": len(parameters.text.split()),
				"line_count": parameters.text.count('\n') + 1
			}
			
			return EnhancedActionResult.success_with_data(
				data=results,
				result_type="text_processing",
				summary=f"Successfully processed {len(parameters.text)} characters with {len(parameters.operations)} operations",
				attachments={
					"text_stats": text_stats,
					"operations_performed": parameters.operations
				}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Text processing failed: {str(e)}",
				category=ResultCategory.EXECUTION,
				context={"operations": parameters.operations, "text_length": len(parameters.text)}
			)


class QueryDatabaseAction(BaseAction[QueryDatabaseParameters, EnhancedActionResult, DataActionContext]):
	"""Action to query a database"""
	
	name: ClassVar[str] = "query_database"
	description: ClassVar[str] = "Query a database table with optional filtering conditions"
	category: ClassVar[str] = ActionCategory.DATABASE
	tags: ClassVar[set[str]] = {ActionTag.DATABASE, ActionTag.READ_ONLY, ActionTag.REQUIRES_DATABASE}
	
	async def execute(
		self,
		parameters: QueryDatabaseParameters,
		context: DataActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute database query"""
		try:
			database = context.get_database()
			
			if parameters.operation == "select":
				results = await database.query(parameters.table, parameters.conditions)
				
				# Apply limit
				if len(results) > parameters.limit:
					results = results[:parameters.limit]
				
				return EnhancedActionResult.success_with_data(
					data=results,
					result_type="database_query",
					summary=f"Successfully queried {parameters.table} table, returned {len(results)} rows",
					attachments={
						"query_info": {
							"table": parameters.table,
							"conditions": parameters.conditions,
							"result_count": len(results),
							"limit_applied": len(results) == parameters.limit
						}
					}
				)
			
			elif parameters.operation == "count":
				count = await database.count(parameters.table, parameters.conditions)
				
				return EnhancedActionResult.success_with_data(
					data={"count": count},
					result_type="database_count",
					summary=f"Successfully counted rows in {parameters.table} table: {count}",
					attachments={
						"query_info": {
							"table": parameters.table,
							"conditions": parameters.conditions,
							"operation": "count"
						}
					}
				)
			
			else:
				return EnhancedActionResult.error_with_details(
					f"Unsupported operation: {parameters.operation}",
					category=ResultCategory.VALIDATION,
					context={"operation": parameters.operation}
				)
		
		except ValueError as e:
			return EnhancedActionResult.error_with_details(
				str(e),
				category=ResultCategory.VALIDATION,
				context={"table": parameters.table, "conditions": parameters.conditions}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Database query failed: {str(e)}",
				category=ResultCategory.DATABASE,
				context={"table": parameters.table, "operation": parameters.operation}
			)


class TransformDataAction(BaseAction[TransformDataParameters, EnhancedActionResult, DataActionContext]):
	"""Action to transform data between different formats"""
	
	name: ClassVar[str] = "transform_data"
	description: ClassVar[str] = "Transform data between formats like JSON, CSV, or apply normalization"
	category: ClassVar[str] = ActionCategory.DATA
	tags: ClassVar[set[str]] = {ActionTag.JSON, ActionTag.CSV, ActionTag.STATELESS}
	
	async def execute(
		self,
		parameters: TransformDataParameters,
		context: DataActionContext,
		**kwargs
	) -> EnhancedActionResult:
		"""Execute data transformation"""
		try:
			data = parameters.data
			transformation = parameters.transformation
			options = parameters.options
			
			if transformation == "to_json":
				indent = options.get("indent", None)
				result = json.dumps(data, indent=indent, default=str)
				
			elif transformation == "to_csv":
				if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
					return EnhancedActionResult.error_with_details(
						"CSV transformation requires a list of dictionaries",
						category=ResultCategory.VALIDATION,
						context={"data_type": type(data).__name__}
					)
				
				delimiter = options.get("delimiter", ",")
				if not data:
					result = ""
				else:
					headers = list(data[0].keys())
					csv_lines = [delimiter.join(headers)]
					for row in data:
						csv_lines.append(delimiter.join(str(row.get(header, "")) for header in headers))
					result = "\n".join(csv_lines)
			
			elif transformation == "normalize":
				# Simple normalization - flatten nested structures
				if isinstance(data, dict):
					result = self._flatten_dict(data)
				elif isinstance(data, list):
					result = [self._flatten_dict(item) if isinstance(item, dict) else item for item in data]
				else:
					result = data
			
			elif transformation == "aggregate":
				if not isinstance(data, list):
					return EnhancedActionResult.error_with_details(
						"Aggregation requires a list of items",
						category=ResultCategory.VALIDATION,
						context={"data_type": type(data).__name__}
					)
				
				group_by = options.get("group_by")
				if group_by:
					# Group by field
					groups = {}
					for item in data:
						if isinstance(item, dict) and group_by in item:
							key = item[group_by]
							if key not in groups:
								groups[key] = []
							groups[key].append(item)
					result = groups
				else:
					# Simple count aggregation
					result = {"total_items": len(data), "item_types": {}}
					for item in data:
						item_type = type(item).__name__
						result["item_types"][item_type] = result["item_types"].get(item_type, 0) + 1
			
			else:
				return EnhancedActionResult.error_with_details(
					f"Unsupported transformation: {transformation}",
					category=ResultCategory.VALIDATION,
					context={"transformation": transformation}
				)
			
			# Calculate transformation statistics
			input_size = len(str(data))
			output_size = len(str(result))
			
			return EnhancedActionResult.success_with_data(
				data=result,
				result_type="data_transformation",
				summary=f"Successfully transformed data using {transformation}",
				attachments={
					"transformation_info": {
						"transformation": transformation,
						"input_size": input_size,
						"output_size": output_size,
						"options": options
					}
				}
			)
		
		except json.JSONEncodeError as e:
			return EnhancedActionResult.error_with_details(
				f"JSON encoding failed: {str(e)}",
				category=ResultCategory.VALIDATION,
				context={"transformation": parameters.transformation}
			)
		
		except Exception as e:
			return EnhancedActionResult.error_with_details(
				f"Data transformation failed: {str(e)}",
				category=ResultCategory.EXECUTION,
				context={"transformation": parameters.transformation}
			)
	
	def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
		"""Flatten a nested dictionary"""
		items = []
		for k, v in d.items():
			new_key = f"{parent_key}{sep}{k}" if parent_key else k
			if isinstance(v, dict):
				items.extend(self._flatten_dict(v, new_key, sep=sep).items())
			else:
				items.append((new_key, v))
		return dict(items)