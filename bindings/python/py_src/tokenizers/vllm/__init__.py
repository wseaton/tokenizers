from typing import Callable
from ..tokenizers import vllm

# Export Rust components
encode: Callable = vllm.encode
TokenByteBuffer = vllm.TokenByteBuffer
TokenizationParams = vllm.TokenizationParams
TokenizationService = vllm.TokenizationService

# Export Python wrappers
from .service import VLLMTokenizerService, create_vllm_tokenizer_service
from .compat import VLLMCompatTokenizer, TextTokensPrompt, create_vllm_compatible_tokenizer
