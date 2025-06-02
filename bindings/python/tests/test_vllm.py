import asyncio
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.vllm import TokenByteBuffer, encode


@pytest.fixture
def simple_tokenizer():
    # create wordlevel tokenizer that maps tokens directly
    vocab = {"hello": 0, "world": 1, "test": 2, "sample": 3, "text": 4, "<unk>": 5}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    return tokenizer


@pytest.mark.asyncio
async def test_vllm_encode_basic(simple_tokenizer):
    """test basic encode functionality with asyncio"""
    # test that regular encoding works as expected first
    regular = simple_tokenizer.encode("hello world test")
    assert regular.ids == [0, 1, 2]
    
    # create token buffer with sample tokens 
    tokens = ["hello", "world", "test"]
    buffer = TokenByteBuffer(tokens)
    
    # encode using the vllm module
    result = await encode(simple_tokenizer, buffer)
    
    # check that we get a numpy array back
    assert hasattr(result, "shape")
    assert hasattr(result, "dtype")
    assert len(result.shape) == 1
    assert result.shape[0] == 3
    assert list(result) == [0, 1, 2]


@pytest.mark.asyncio
async def test_vllm_encode_empty_buffer(simple_tokenizer):
    """test encode with empty token buffer"""
    tokens = []
    buffer = TokenByteBuffer(tokens)
    
    result = await encode(simple_tokenizer, buffer)
    
    assert result.shape[0] == 0


@pytest.mark.asyncio
async def test_vllm_encode_single_token(simple_tokenizer):
    """test encode with single token"""
    tokens = ["hello"]
    buffer = TokenByteBuffer(tokens)
    
    result = await encode(simple_tokenizer, buffer)
    
    assert result.shape[0] == 1
    assert result[0] == 0  # "hello" should map to id 0


@pytest.mark.asyncio
async def test_vllm_encode_unknown_token(simple_tokenizer):
    """test encode with unknown token"""
    tokens = ["unknown"]
    buffer = TokenByteBuffer(tokens)
    
    result = await encode(simple_tokenizer, buffer)
    
    assert result.shape[0] == 1
    assert result[0] == 5  # should map to unk token id


@pytest.mark.asyncio
async def test_token_byte_buffer_properties():
    """test token buffer basic properties"""
    tokens = ["hello", "world", "test"]
    buffer = TokenByteBuffer(tokens)
    
    assert buffer.token_count() == 3
    assert buffer.buffer_size() > 0
    
    # test individual token retrieval
    assert buffer.get_token(0) == "hello"
    assert buffer.get_token(1) == "world"
    assert buffer.get_token(2) == "test"
    
    # test out of bounds
    with pytest.raises(Exception):
        buffer.get_token(10)


def test_token_byte_buffer_memory_stats():
    """test memory statistics"""
    tokens = ["hello", "world"]
    buffer = TokenByteBuffer(tokens)
    
    stats = buffer.memory_stats()
    
    assert "buffer_bytes" in stats
    assert "lengths_bytes" in stats
    assert "offsets_bytes" in stats
    assert "total_bytes" in stats
    assert "token_count" in stats
    assert stats["token_count"] == 2