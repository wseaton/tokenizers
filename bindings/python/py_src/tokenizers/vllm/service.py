"""
High-level vLLM tokenization service that integrates with the Rust backend.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional

from ..tokenizers import vllm
from .compat import TextTokensPrompt


@dataclass
class TokenizationParams:
    """Parameters for tokenization compatible with vLLM's expectations."""
    add_special_tokens: bool = True
    truncation: bool = False
    max_length: Optional[int] = None
    do_lower_case: bool = False

    def to_rust_params(self) -> vllm.TokenizationParams:
        """Convert to Rust TokenizationParams object."""
        return vllm.TokenizationParams(
            add_special_tokens=self.add_special_tokens,
            truncation=self.truncation,
            max_length=self.max_length,
            do_lower_case=self.do_lower_case
        )


class VLLMTokenizerService:
    """
    High-level tokenization service that provides vLLM-compatible interface
    while using the fast Rust backend for performance.
    """
    
    def __init__(self, base_tokenizer, max_concurrent: int = 4):
        self.base_tokenizer = base_tokenizer
        self.rust_service = vllm.TokenizationService(base_tokenizer, max_concurrent)
        self._fallback_enabled = True
    
    def extract_params(
        self, 
        request: Any,
        truncate_prompt_tokens: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> TokenizationParams:
        """
        Extract tokenization parameters from a vLLM request object.
        
        This avoids needing to model complex Request objects in Rust.
        """
        # Determine if we should do lowercase normalization
        do_lower_case = False
        if hasattr(request, 'model_config') and request.model_config:
            encoder_config = getattr(request.model_config, 'encoder_config', None)
            if encoder_config and isinstance(encoder_config, dict):
                do_lower_case = encoder_config.get('do_lower_case', False)
        
        # Handle truncation parameters
        truncation = truncate_prompt_tokens is not None
        max_length = None
        if truncate_prompt_tokens is not None and truncate_prompt_tokens > 0:
            max_length = truncate_prompt_tokens
        elif truncate_prompt_tokens == -1:
            # Use model's max length
            max_length = getattr(request, 'max_model_len', None)
            if max_length is None and hasattr(request, 'model_config'):
                max_length = getattr(request.model_config, 'max_model_len', None)
        
        return TokenizationParams(
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            do_lower_case=do_lower_case
        )
    
    async def tokenize_prompt_inputs_fast(
        self,
        request: Any,
        prompt_inputs: List[str],
        truncate_prompt_tokens: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> List[TextTokensPrompt]:
        """
        Fast tokenization of multiple prompt inputs using Rust backend.
        """
        try:
            # Extract parameters from request
            params = self.extract_params(request, truncate_prompt_tokens, add_special_tokens)
            rust_params = params.to_rust_params()
            
            # Route to fast Rust path using TokenByteBuffer for optimal performance
            if self._should_use_fast_path(prompt_inputs, params):
                from ..tokenizers import vllm
                buffer = vllm.TokenByteBuffer(prompt_inputs)
                token_ids_array = await self.rust_service.tokenize_buffer(buffer, rust_params)
                
                # Convert numpy array back to list of token sequences
                # The array should be 1D with all tokens concatenated
                token_ids_list = token_ids_array.tolist()
                
                # For now, return as single sequence - this will need refinement
                # based on how tokenize_buffer actually handles multiple inputs
                token_ids_batch = [token_ids_list]
                
                return [
                    TextTokensPrompt(prompt=text, prompt_token_ids=ids)
                    for text, ids in zip(prompt_inputs, token_ids_batch)
                ]
            
        except Exception as e:
            if not self._fallback_enabled:
                raise
            # Log the error for debugging
            print(f"Fast tokenization failed, falling back: {e}")
        
        # Fallback to original implementation
        return self._tokenize_fallback(request, prompt_inputs, truncate_prompt_tokens, add_special_tokens)
    
    async def tokenize_single_fast(
        self,
        request: Any,
        prompt: str,
        truncate_prompt_tokens: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """
        Fast tokenization of a single prompt using Rust backend.
        """
        try:
            # Extract parameters from request
            params = self.extract_params(request, truncate_prompt_tokens, add_special_tokens)
            rust_params = params.to_rust_params()
            
            # Route to fast Rust path using TokenByteBuffer
            if self._should_use_fast_path([prompt], params):
                from ..tokenizers import vllm
                buffer = vllm.TokenByteBuffer([prompt])
                token_ids_array = await self.rust_service.tokenize_buffer(buffer, rust_params)
                token_ids = token_ids_array.tolist()
                return TextTokensPrompt(prompt=prompt, prompt_token_ids=token_ids)
            
        except Exception as e:
            if not self._fallback_enabled:
                raise
            # Log the error for debugging
            print(f"Fast tokenization failed, falling back: {e}")
        
        results = self._tokenize_fallback(request, [prompt], truncate_prompt_tokens, add_special_tokens)
        return results[0]
    
    def _should_use_fast_path(self, prompts: List[str], params: TokenizationParams) -> bool:
        """
        Determine if we should use the fast Rust path based on input characteristics.
        """
        if os.getenv('VLLM_DISABLE_FAST_TOKENIZATION'):
            return False
        # Skip if there are unsupported parameters
        unsupported_params = set(vars(params).keys()) - {'add_special_tokens', 'truncation', 'max_length', 'do_lower_case'}
        if unsupported_params:
            return False
    
    def _tokenize_fallback(
        self,
        request: Any,
        prompt_inputs: List[str],
        truncate_prompt_tokens: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> List[TextTokensPrompt]:
        """
        Fallback tokenization using the original tokenizer.
        
        This should mimic the original vLLM tokenization logic.
        """
        results = []
        
        for prompt in prompt_inputs:
            processed_prompt = prompt
            if hasattr(request, 'model_config') and request.model_config:
                encoder_config = getattr(request.model_config, 'encoder_config', None)
                if encoder_config and isinstance(encoder_config, dict):
                    if encoder_config.get('do_lower_case', False):
                        processed_prompt = processed_prompt.lower()
            

            if truncate_prompt_tokens is None:
                encoded = self.base_tokenizer(processed_prompt, add_special_tokens=add_special_tokens)
            elif truncate_prompt_tokens < 0:
                max_len = getattr(request, 'max_model_len', 4096)
                encoded = self.base_tokenizer(
                    processed_prompt,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_len
                )
            else:
                encoded = self.base_tokenizer(
                    processed_prompt,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=truncate_prompt_tokens
                )
            
            results.append(TextTokensPrompt(
                prompt=processed_prompt,
                prompt_token_ids=encoded.input_ids
            ))
        
        return results


def create_vllm_tokenizer_service(base_tokenizer, max_concurrent: int = 4) -> VLLMTokenizerService:
    """
    Factory function to create a vLLM tokenizer service with the fast Rust backend.
    """
    return VLLMTokenizerService(base_tokenizer, max_concurrent)