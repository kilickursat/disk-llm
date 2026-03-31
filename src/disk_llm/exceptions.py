"""Project-specific exceptions."""


class DiskLLMError(Exception):
    """Base exception for Disk-LLM."""


class DependencyMissingError(DiskLLMError):
    """Raised when an optional dependency is required for a code path."""


class ManifestError(DiskLLMError):
    """Raised when a packed-model manifest is missing or malformed."""


class SafetensorsFormatError(DiskLLMError):
    """Raised when a safetensors file cannot be parsed."""


class ConversionError(DiskLLMError):
    """Raised when model conversion fails."""


class RuntimeShapeError(DiskLLMError):
    """Raised when runtime tensor shapes or names do not line up."""
