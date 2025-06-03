"""
Compatibility layer for vLLM integration.
"""

from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass 
class TextTokensPrompt:
    """
    Represents a tokenized prompt compatible with vLLM's expectations.
    
    This mimics the structure that vLLM expects from tokenization operations.
    """
    prompt: str
    prompt_token_ids: List[int]
    
    def __post_init__(self):
        # Ensure token IDs are integers
        if self.prompt_token_ids:
            self.prompt_token_ids = [int(token_id) for token_id in self.prompt_token_ids]


@dataclass
class EmbedsPrompt:
    """
    Represents an embedded prompt for vLLM compatibility.
    
    This is used when prompts are provided as embeddings rather than text.
    """
    prompt_embeds: Union[List[float], List[List[float]]]
    
    
class EncodingCompat:
    """
    Compatibility wrapper that mimics HuggingFace tokenizer encoding objects.
    
    This allows the Rust tokenization results to be used with existing vLLM code
    that expects standard tokenizer outputs.
    """
    
    def __init__(self, input_ids: List[int], text: str):
        self.input_ids = input_ids
        self.text = text
        
    @property
    def tokens(self) -> List[str]:
        """Return tokens (stub implementation)."""
        # This is a simplified implementation - in practice you might want
        # to decode the token IDs back to strings if needed
        return [f"<token_{i}>" for i in self.input_ids]


class VLLMCompatTokenizer:
    """
    Drop-in replacement tokenizer that provides HuggingFace-compatible interface
    while using the fast Rust backend for actual tokenization.
    """
    
    def __init__(self, base_tokenizer, rust_service):
        self.base_tokenizer = base_tokenizer
        self.rust_service = rust_service
        
        # Copy relevant attributes from base tokenizer
        for attr in ['vocab_size', 'model_max_length', 'pad_token_id', 'eos_token_id']:
            if hasattr(base_tokenizer, attr):
                setattr(self, attr, getattr(base_tokenizer, attr))
    
    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kwargs):
        """
        Main tokenization interface that's compatible with HuggingFace tokenizers.
        """
        # Check if we should use the fast path
        if self._should_use_fast_path(text, kwargs):
            return self._fast_encode(text, add_special_tokens, truncation, max_length)
        
        # Fallback to original tokenizer
        return self.base_tokenizer(text, add_special_tokens=add_special_tokens, 
                                  truncation=truncation, max_length=max_length, **kwargs)
    
    def _should_use_fast_path(self, text, kwargs) -> bool:
        """
        Determine if we should use the fast Rust path.
        """
        # Skip if there are unsupported kwargs
        unsupported_kwargs = set(kwargs.keys()) - {'add_special_tokens', 'truncation', 'max_length'}
        if unsupported_kwargs:
            return False
        
        # Skip for very complex inputs
        if isinstance(text, (list, tuple)) and len(text) > 1000:
            return False
            
        return True
    
    def _fast_encode(self, text, add_special_tokens, truncation, max_length):
        """
        Fast encoding using Rust backend.
        """
        # Create parameters
        from .service import TokenizationParams
        params = TokenizationParams(
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            do_lower_case=False  # Would need to be determined from model config
        )
        
        if isinstance(text, str):
            # Single text
            # Note: This would need to be made async in practice
            token_ids = self.rust_service.tokenize_single(text, params.to_rust_params())
            return EncodingCompat(token_ids, text)
        else:
            # Batch of texts
            # Note: This would need to be made async in practice
            token_batches = self.rust_service.tokenize_batch(text, params.to_rust_params())
            return [EncodingCompat(tokens, txt) for tokens, txt in zip(token_batches, text)]


def create_vllm_compatible_tokenizer(base_tokenizer, rust_service=None):
    """
    Create a vLLM-compatible tokenizer that uses the fast Rust backend when possible.
    """
    if rust_service is None:
        from .service import create_vllm_tokenizer_service
        rust_service = create_vllm_tokenizer_service(base_tokenizer)
    
    return VLLMCompatTokenizer(base_tokenizer, rust_service)