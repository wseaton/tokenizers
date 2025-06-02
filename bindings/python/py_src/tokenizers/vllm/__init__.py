from typing import Callable
from ..tokenizers import vllm

encode: Callable = vllm.encode
TokenByteBuffer = vllm.TokenByteBuffer
