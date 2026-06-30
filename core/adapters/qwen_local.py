"""
QWEN LOCAL ADAPTER
==================
Implements BaseAdapter by loading Qwen2.5-0.5B-Instruct weights
DIRECTLY into memory using the transformers library.

This is fundamentally different from QwenAdapter (qwen.py):
- QwenAdapter     : HTTP request → HuggingFace's servers → response
- QwenLocalAdapter: load weights into THIS machine → run inference locally

Why 0.5B and not 7B?
HuggingFace Spaces free tier gives you a CPU machine with ~16GB RAM.
Qwen2.5-0.5B weights are ~1GB. Fits easily.
Qwen2.5-7B weights are ~14GB. Does not fit on free tier.

The tradeoff: 0.5B is weaker than 7B on quality, but it's
free to deploy, fully under your control, and measurable.

Key concepts introduced here:
- Model loading (from_pretrained)
- Tokenization (text → token IDs → text)
- Generation parameters (temperature, top_p, max_new_tokens)
- Device management (CPU vs GPU)
- Chat template (how to format messages for this specific model)
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseAdapter, AdapterResponse


class QwenLocalAdapter(BaseAdapter):
    """
    Adapter for Qwen2.5-0.5B-Instruct running locally via transformers.

    The model is loaded ONCE at initialization and kept in memory.
    Every generate() call reuses the same loaded model — no re-loading
    on each request (that would be catastrophically slow).

    Args:
        model_id   : HuggingFace model identifier to load
        device     : "cpu", "cuda", or "auto" (auto-detects GPU if available)
        max_tokens : max new tokens to generate per response
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
        max_tokens: int = 512,
    ):
        self._model_id = model_id
        self._max_tokens = max_tokens

        # Determine device
        # "auto" → use GPU if available, otherwise CPU
        # On HF Spaces free tier this will always be CPU
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        print(f"[QwenLocal] Loading {model_id} on {self._device}...")
        print(f"[QwenLocal] This may take 30-60 seconds on first run (downloading weights)...")

        # ── Load tokenizer ──
        # The tokenizer converts text ↔ token IDs
        # trust_remote_code=True is required for Qwen models — they have
        # custom tokenization logic that lives in the model repo
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # ── Load model weights ──
        # torch_dtype=torch.float16 uses half-precision floats
        # This halves memory usage with minimal quality loss
        # On CPU, float32 is sometimes more stable — but float16 fits better
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
            trust_remote_code=True,
        )

        # Put model in evaluation mode — disables dropout layers
        # that are only needed during training, not inference
        self._model.eval()

        print(f"[QwenLocal] Model loaded successfully.")

    # ── ABSTRACT METHOD IMPLEMENTATIONS ──────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "local"

    def generate(self, messages: list[dict], system_prompt: str) -> AdapterResponse:
        """
        Run local inference on the loaded model.

        The key difference from API adapters:
        Instead of making an HTTP request, we:
        1. Apply the model's chat template to format messages
        2. Tokenize the formatted text into token IDs (numbers)
        3. Pass token IDs through the model's forward pass
        4. Decode the output token IDs back to text

        This all happens in-process, in memory, on this machine.
        """
        start = self._start_timer()

        try:
            # ── Step 1: Apply chat template ──
            # Each model has a specific way it expects conversations formatted.
            # Qwen's template looks roughly like:
            #   <|im_start|>system\nYou are helpful<|im_end|>
            #   <|im_start|>user\nHello<|im_end|>
            #   <|im_start|>assistant\n
            #
            # apply_chat_template() handles this formatting automatically.
            # tokenize=False means return the formatted STRING, not token IDs yet.
            # add_generation_prompt=True appends the "assistant turn start" marker
            # so the model knows to generate a response.
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # ── Step 2: Tokenize ──
            # Convert the formatted string into a tensor of token IDs
            # return_tensors="pt" means return PyTorch tensors
            inputs = self._tokenizer(
                formatted,
                return_tensors="pt",
            ).to(self._device)

            input_token_count = inputs["input_ids"].shape[1]

            # ── Step 3: Generate ──
            # This is the actual model inference — the forward pass
            # torch.no_grad() disables gradient tracking (only needed for training)
            # This saves memory and speeds up inference
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_tokens,

                    # Temperature: controls randomness
                    # 0.0 = deterministic (always picks most likely token)
                    # 1.0 = full randomness
                    # 0.7 = balanced — good for assistant responses
                    temperature=0.7,

                    # top_p (nucleus sampling): at each step, only consider
                    # tokens whose cumulative probability exceeds top_p.
                    # Prevents very unlikely tokens from being selected.
                    top_p=0.9,

                    # do_sample=True enables probabilistic sampling
                    # False would use greedy decoding (always pick top token)
                    do_sample=True,

                    # Prevent the model from repeating the input tokens
                    # in the output — we only want the NEW tokens
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # ── Step 4: Decode ──
            # output_ids contains BOTH input tokens and generated tokens
            # We slice [input_token_count:] to get ONLY the new generated tokens
            new_tokens = output_ids[0][input_token_count:]
            output_token_count = len(new_tokens)

            # Convert token IDs back to text
            # skip_special_tokens=True removes <|im_end|> and similar markers
            response_text = self._tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            return AdapterResponse(
                text=response_text,
                model=self._model_id,
                provider=self.provider,
                latency_ms=self._elapsed_ms(start),
                input_tokens=input_token_count,
                output_tokens=output_token_count,
                success=True,
            )

        except Exception as e:
            return self._error_response(e, self._elapsed_ms(start))
