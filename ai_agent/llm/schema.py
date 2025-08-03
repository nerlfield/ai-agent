from typing import Any

from pydantic import BaseModel


class SchemaOptimizer:
	@staticmethod
	def create_optimized_json_schema(model: type[BaseModel]) -> dict[str, Any]:
		original_schema = model.model_json_schema()
		defs_lookup = original_schema.get('$defs', {})

		def optimize_schema(
			obj: Any,
			defs_lookup: dict[str, Any] | None = None,
			*,
			in_properties: bool = False,
		) -> Any:
			if isinstance(obj, dict):
				optimized: dict[str, Any] = {}
				flattened_ref: dict[str, Any] | None = None

				skip_fields = ['additionalProperties', '$defs']

				for key, value in obj.items():
					if key in skip_fields:
						continue

					if key == 'title' and not in_properties:
						continue

					elif key == 'description':
						optimized[key] = value

					elif key == 'type':
						optimized[key] = value

					elif key == '$ref' and defs_lookup:
						ref_path = value.split('/')[-1]
						if ref_path in defs_lookup:
							referenced_def = defs_lookup[ref_path]
							flattened_ref = optimize_schema(referenced_def, defs_lookup)

					elif key == 'anyOf' and isinstance(value, list):
						optimized[key] = [optimize_schema(item, defs_lookup) for item in value]

					elif key in ['properties', 'items']:
						optimized[key] = optimize_schema(
							value,
							defs_lookup,
							in_properties=(key == 'properties'),
						)

					elif key in ['type', 'required', 'minimum', 'maximum', 'minItems', 'maxItems', 'pattern', 'default']:
						optimized[key] = value if not isinstance(value, (dict, list)) else optimize_schema(value, defs_lookup)

					else:
						optimized[key] = optimize_schema(value, defs_lookup) if isinstance(value, (dict, list)) else value

				if flattened_ref is not None and isinstance(flattened_ref, dict):
					result = flattened_ref.copy()

					for key, value in optimized.items():
						if key == 'description' and 'description' not in result:
							result[key] = value
						elif key != 'description':
							result[key] = value

					return result
				else:
					if optimized.get('type') == 'object':
						optimized['additionalProperties'] = False

					return optimized

			elif isinstance(obj, list):
				return [optimize_schema(item, defs_lookup, in_properties=in_properties) for item in obj]
			return obj

		optimized_result = optimize_schema(original_schema, defs_lookup)

		if not isinstance(optimized_result, dict):
			raise ValueError('Optimized schema result is not a dictionary')

		optimized_schema: dict[str, Any] = optimized_result

		def ensure_additional_properties_false(obj: Any) -> None:
			if isinstance(obj, dict):
				if obj.get('type') == 'object':
					obj['additionalProperties'] = False

				for value in obj.values():
					if isinstance(value, (dict, list)):
						ensure_additional_properties_false(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						ensure_additional_properties_false(item)

		ensure_additional_properties_false(optimized_schema)
		SchemaOptimizer._make_strict_compatible(optimized_schema)

		return optimized_schema

	@staticmethod
	def _make_strict_compatible(schema: dict[str, Any] | list[Any]) -> None:
		if isinstance(schema, dict):
			for key, value in schema.items():
				if isinstance(value, (dict, list)) and key != 'required':
					SchemaOptimizer._make_strict_compatible(value)

			if 'properties' in schema and 'type' in schema and schema['type'] == 'object':
				all_props = list(schema['properties'].keys())
				schema['required'] = all_props

		elif isinstance(schema, list):
			for item in schema:
				SchemaOptimizer._make_strict_compatible(item)