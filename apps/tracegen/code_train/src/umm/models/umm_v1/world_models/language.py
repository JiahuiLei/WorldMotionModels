"""Pretrained language encoders for text-conditioned motion generation.

Provides frozen T5 and CLIP text encoders that produce fixed-length token
representations used as conditioning signals in the MotionMachine pipeline.

Classes:
    T5LanguageEncoder: Wraps ``T5EncoderModel`` to produce per-token hidden
        states of shape ``[B, fixed_length, hidden_size]``.
    CLIPLanguageEncoder: Wraps ``CLIPModel.get_text_features`` to produce
        global text embeddings of shape ``[B, hidden_size]``.

Example:
    >>> enc = T5LanguageEncoder(fixed_length=16)
    >>> tokens = enc.encode(["pick up the cube"])  # [1, 16, 768]
"""

from transformers import T5TokenizerFast, T5EncoderModel
import torch
from torch import nn


class T5LanguageEncoder(nn.Module):
    """Frozen T5 encoder that maps text strings to fixed-length token embeddings.

    The model is set to ``eval()`` mode on init and all forward passes use
    ``torch.no_grad()``, so no gradients flow through the encoder.

    Args:
        model_name (str): HuggingFace model identifier. Defaults to ``"t5-base"``.
        fixed_length (int): Token sequence length (pad/truncate). Defaults to 32.

    Example:
        >>> enc = T5LanguageEncoder(model_name="t5-base", fixed_length=16)
        >>> out = enc.encode(["hello world"])  # [1, 16, 768]
    """

    def __init__(self, model_name="t5-base", fixed_length=32):
        """Initialize tokenizer and frozen T5 encoder model.

        Args:
            model_name (str): HuggingFace T5 model name.
            fixed_length (int): Fixed token length for padding/truncation.

        Example:
            >>> enc = T5LanguageEncoder(fixed_length=16)
        """
        super(T5LanguageEncoder, self).__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.fixed_length = fixed_length
        self.model.eval()

    def encode(self, batch_strings):
        """Encode a batch of strings into fixed-length T5 hidden states.

        Args:
            batch_strings (list[str]): List of text strings to encode.

        Returns:
            torch.Tensor: Last hidden state of shape
                ``[batch_size, fixed_length, hidden_size]``.

        Example:
            >>> enc = T5LanguageEncoder(fixed_length=16)
            >>> out = enc.encode(["pick up the red cube", "open the drawer"])
        """
        # Tokenize with padding and truncation to ensure fixed length
        inputs = self.tokenizer(
            batch_strings,
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer
            max_length=self.fixed_length,  # Fixed length
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Move inputs to the same device as the model
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = list(self.model.parameters())[0].device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode to get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

        return (
            outputs.last_hidden_state
        )  # Shape: (batch_size, fixed_length, hidden_size)


class CLIPLanguageEncoder(nn.Module):
    """Frozen CLIP text encoder that maps strings to global text feature vectors.

    Uses ``CLIPModel.get_text_features`` to produce a single embedding vector
    per input string. The model is set to ``eval()`` on init and all forward
    passes use ``torch.no_grad()``.

    Args:
        model_name (str): HuggingFace CLIP model identifier.
            Defaults to ``"openai/clip-vit-base-patch32"``.
        fixed_length (int): Token sequence length (pad/truncate). Defaults to 32.

    Example:
        >>> enc = CLIPLanguageEncoder(fixed_length=16)
        >>> out = enc.encode(["pick up the cube"])  # [1, 512]
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", fixed_length=32):
        """Initialize tokenizer and frozen CLIP model.

        Args:
            model_name (str): HuggingFace CLIP model name.
            fixed_length (int): Fixed token length for padding/truncation.

        Example:
            >>> enc = CLIPLanguageEncoder(fixed_length=16)
        """
        super(CLIPLanguageEncoder, self).__init__()
        from transformers import CLIPTokenizer, CLIPModel

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.fixed_length = fixed_length
        self.model.eval()

    def encode(self, batch_strings):
        """Encode a batch of strings into CLIP global text features.

        Args:
            batch_strings (list[str]): List of text strings to encode.

        Returns:
            torch.Tensor: Text feature vectors of shape ``[batch_size, hidden_size]``.

        Example:
            >>> enc = CLIPLanguageEncoder(fixed_length=16)
            >>> out = enc.encode(["pick up the cube"])
        """
        # Tokenize with padding and truncation to ensure fixed length
        inputs = self.tokenizer(
            batch_strings,
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer
            max_length=self.fixed_length,  # Fixed length
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Move inputs to the same device as the model
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = list(self.model.parameters())[0].device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Encode to get hidden states
        with torch.no_grad():
            outputs = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )


if __name__ == "__main__":
    # Example usage
    encoder = T5LanguageEncoder(fixed_length=16).cuda()
    strings = ["Hello, world!", "This is a test string."]
    encoded = encoder.encode(strings)

    print(encoder.encode([""])[0])
    print(encoder.encode(["", ""])[0])

    print(encoded.std())
    print(encoded.shape)  # Should print (batch_size, fixed_length, hidden_size)

    clip_encoder = CLIPLanguageEncoder()
    clip_encoded = clip_encoder.encode(strings)
    print(clip_encoded.shape)  # Should print (batch_size, hidden_size)
