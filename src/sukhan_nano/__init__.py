"""Sukhan-Nano: a tiny from-scratch Urdu language model (educational showcase)."""

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

__version__ = "0.1.0"
__all__ = ["GPT", "GPTConfig", "BPETokenizer"]
