"""
Unified Chat Client for OpenAI, TogetherAI, and Google Generative AI APIs

This module provides a unified interface for OpenAI, TogetherAI, and Google Generative AI (Gemini) chat completion APIs,
normalizing their different input/output formats to provide a consistent experience.

Key features:
- Unified request parameters (automatically converts logprobs parameters)
- Normalized response format with consistent logprobs structure
- Support for OpenAI, TogetherAI, and Google Generative AI (Gemini) models
- Backward compatibility with existing code using .choices[0].logprobs.content[0].top_logprobs[0].token
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Dict, Any, Union, Optional
import logging
import os
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
import re

try:
    from together import Together

    # TogetherAI might not have the same exception structure, use generic handling
    TogetherRateLimitError = Exception
    TogetherAPITimeoutError = Exception
    TogetherAPIConnectionError = Exception
except ImportError:
    Together = None
    TogetherRateLimitError = Exception
    TogetherAPITimeoutError = Exception
    TogetherAPIConnectionError = Exception
    print(
        "Warning: Together AI library not installed. TogetherAI functionality will be disabled."
    )

try:
    from google import genai
    from google.genai.types import GenerateContentConfig

    # Google Generative AI exception handling
    GeminiRateLimitError = Exception
    GeminiAPITimeoutError = Exception
    GeminiAPIConnectionError = Exception
except ImportError:
    genai = None
    GenerateContentConfig = None
    GeminiRateLimitError = Exception
    GeminiAPITimeoutError = Exception
    GeminiAPIConnectionError = Exception
    print(
        "Warning: Google Generative AI library not installed. Gemini functionality will be disabled."
    )


@dataclass
class TokenLogprob:
    """Data class for token and its log probability"""

    token: str
    logprob: float


class UnifiedChatClient:
    """
    Unified client for OpenAI, TogetherAI, and Google Generative AI (Gemini) chat completion APIs.

    Automatically detects the provider based on the model name and handles
    the different parameter formats and response structures.
    """

    def __init__(
        self,
        openai_api_key: str = None,
        together_api_key: str = None,
        gemini_api_key: str = None,
        timeout: int = 30,
        max_retries: int = 2,
    ):
        """
        Initialize the unified client.

        Args:
            openai_api_key: OpenAI API key (can be None if only using other providers)
            together_api_key: TogetherAI API key (can be None if only using other providers)
            gemini_api_key: Google Generative AI API key (can be None if only using other providers)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.openai_client = None
        self.together_client = None
        self.gemini_client = None

        if openai_api_key:
            self.openai_client = OpenAI(
                api_key=openai_api_key, timeout=timeout, max_retries=max_retries
            )

        if together_api_key:
            if Together is None:
                raise ImportError(
                    "Together AI library not installed. Please install it with: pip install together"
                )

            self.together_client = Together(
                api_key=together_api_key, timeout=timeout, max_retries=max_retries
            )

        if gemini_api_key:
            if genai is None:
                raise ImportError(
                    "Google Generative AI library not installed. Please install it with: pip install google-genai"
                )

            # Suppress AFC (Automatic Function Calling) logs from Google AI library
            # Set to ERROR level to completely silence INFO/WARNING AFC messages
            google_loggers = [
                "google.genai",
                "google.ai.generativelanguage",
                "google.generativeai",
                "google.ai",
                "google.api_core",
                "google.auth",
                "google.oauth2",
                "google.cloud",
                "google",
            ]

            for logger_name in google_loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.ERROR)
                # Also disable propagation to prevent messages from bubbling up
                logger.propagate = False

            self.gemini_client = genai.Client(
                api_key=gemini_api_key,
            )

    def _is_openai_model(self, model: str) -> bool:
        """Check if the model is an OpenAI model"""
        return model.startswith("gpt-") or model in [
            "text-davinci-003",
            "text-davinci-002",
        ]

    def _is_together_model(self, model: str) -> bool:
        """Check if the model is a TogetherAI model"""
        together_prefixes = [
            "meta-llama/",
            "mistralai/",
            "google/",
            "microsoft/",
            "NousResearch/",
            "teknium/",
            "Qwen/",
            "togethercomputer/",
        ]
        return any(model.startswith(prefix) for prefix in together_prefixes)

    def _is_gemini_model(self, model: str) -> bool:
        """Check if the model is a Gemini model"""
        gemini_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-pro-vision",
        ]
        return model in gemini_models or model.startswith("gemini-")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]] = None,
        prompt: str = None,
        top_logprobs: int = None,
        **kwargs,
    ) -> SimpleNamespace:
        """
        Unified chat completion method that works with OpenAI, TogetherAI, and Gemini.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "meta-llama/Llama-3-8b-chat-hf", or "gemini-2.5-pro")
            messages: List of message dicts (OpenAI format)
            prompt: Simple prompt string (alternative to messages)
            top_logprobs: Number of top log probabilities to return
            **kwargs: Additional parameters passed to the underlying API

        Returns:
            Normalized response with consistent .choices[0].logprobs.content structure

        Raises:
            ValueError: If required parameters are missing or clients not initialized
            RateLimitError, APITimeoutError, APIConnectionError: API-specific errors
        """
        # Convert prompt to messages format if needed
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Determine which client to use
        if self._is_openai_model(model):
            if not self.openai_client:
                raise ValueError(
                    "OpenAI client not initialized. Provide openai_api_key."
                )

            return self._call_openai(model, messages, top_logprobs, **kwargs)
        elif self._is_together_model(model):
            if not self.together_client:
                raise ValueError(
                    "TogetherAI client not initialized. Provide together_api_key."
                )
            return self._call_together(model, messages, top_logprobs, **kwargs)
        elif self._is_gemini_model(model):
            if not self.gemini_client:
                raise ValueError(
                    "Gemini client not initialized. Provide gemini_api_key."
                )
            return self._call_gemini(model, messages, top_logprobs, **kwargs)
        else:
            # Default to OpenAI for unknown models
            logging.warning(f"Unknown model '{model}', defaulting to OpenAI client")
            if not self.openai_client:
                raise ValueError(
                    "OpenAI client not initialized. Provide openai_api_key."
                )
            return self._call_openai(model, messages, top_logprobs, **kwargs)

    def _call_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = None,
        **kwargs,
    ) -> SimpleNamespace:
        """Call OpenAI API and normalize response"""
        # Check if this is a GPT-5 model that might not support logprobs
        is_gpt5_model = "gpt-5" in model.lower()
        fake_logprobs = is_gpt5_model
        # Set up OpenAI-specific parameters
        params = {"model": model, "messages": messages, **kwargs}

        # Only set logprobs for non-GPT-5 models or if not faking
        if top_logprobs is not None and not is_gpt5_model:
            params["logprobs"] = True
            params["top_logprobs"] = top_logprobs

        # Make the API call
        response = self.openai_client.chat.completions.create(**params)

        # Normalize response, with fake logprobs for GPT-5 if needed
        if fake_logprobs:
            return self._normalize_response(
                response, source="gpt-5", fake_logprobs=True
            )
        else:
            return self._normalize_response(response)

    def _call_together(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = None,
        **kwargs,
    ) -> SimpleNamespace:
        """Call TogetherAI API and normalize response"""
        # Set up TogetherAI-specific parameters
        params = {"model": model, "messages": messages, **kwargs}

        if top_logprobs is not None:
            params["logprobs"] = top_logprobs  # TogetherAI uses integer directly

        # Make the API call
        response = self.together_client.chat.completions.create(**params)

        # Normalize response
        return self._normalize_response(response)

    def _call_gemini(
        self,
        model: str,
        messages: List[Dict[str, str]],
        top_logprobs: int = None,
        **kwargs,
    ) -> SimpleNamespace:
        """Call Gemini API and normalize response"""
        # Convert messages to Gemini format (simple content string for now)
        # For multi-turn conversations, Gemini expects a different format
        if len(messages) == 1 and messages[0]["role"] == "user":
            # Simple single message
            contents = messages[0]["content"]
        else:
            # Multi-turn conversation - convert to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "user":
                    contents.append(
                        {"role": "user", "parts": [{"text": msg["content"]}]}
                    )
                elif msg["role"] == "assistant":
                    contents.append(
                        {"role": "model", "parts": [{"text": msg["content"]}]}
                    )
                elif msg["role"] == "system":
                    # System messages need special handling in Gemini
                    contents.append(
                        {
                            "role": "user",
                            "parts": [{"text": f"System: {msg['content']}"}],
                        }
                    )

        # Set up Gemini-specific parameters (only accept temperature)
        config_params = {}

        # Only accept temperature parameter for Gemini
        if "temperature" in kwargs:
            config_params["temperature"] = kwargs["temperature"]

        # Create config if we have parameters
        config = GenerateContentConfig(**config_params) if config_params else None

        # Make the API call
        try:
            if config:
                response = self.gemini_client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            else:
                response = self.gemini_client.models.generate_content(
                    model=model, contents=contents
                )

        except Exception as e:
            raise ValueError(f"Gemini API error: {e}")

        # Normalize response and fake logprobs if requested
        return self._normalize_response(
            response, source="gemini", fake_logprobs=top_logprobs is not None
        )

    def _normalize_response(
        self, raw_response, source: str = "auto", fake_logprobs: bool = False
    ) -> SimpleNamespace:
        """
        Normalize response from OpenAI, TogetherAI, or Gemini to consistent format.

        Args:
            raw_response: Raw response from OpenAI, TogetherAI, or Gemini
            source: Source of the response ("auto", "gemini", "openai", "together")
            fake_logprobs: Whether to fake logprobs for Gemini responses

        Returns:
            Normalized response with consistent structure
        """

        # Handle Gemini responses
        if source == "gemini" or self._is_gemini_response(raw_response):
            return self._normalize_gemini_response(
                raw_response, fake_logprobs=fake_logprobs
            )

        # First, detect if this is a TogetherAI object that needs conversion
        if hasattr(raw_response, "choices") and len(raw_response.choices) > 0:
            choice = raw_response.choices[0]
            if hasattr(choice, "logprobs") and choice.logprobs:
                if hasattr(choice.logprobs, "top_logprobs"):
                    # This is a TogetherAI object, convert to dict first
                    raw_dict = self._together_obj_to_dict(raw_response)
                    openai_format = self._together_to_openai_dict(raw_dict)
                    return self._build_normalized_response(openai_format)

        # Try to convert to dict (works for OpenAI objects)
        try:
            if hasattr(raw_response, "to_dict"):
                raw_dict = raw_response.to_dict()
            elif hasattr(raw_response, "model_dump"):
                raw_dict = raw_response.model_dump()
            else:
                # Assume it's already a dict
                raw_dict = raw_response
        except Exception as e:
            # Fallback: treat as dict
            raw_dict = raw_response

        # Check if this is TogetherAI format that needs conversion
        if self._is_together_format(raw_dict):
            raw_dict = self._together_to_openai_dict(raw_dict)

        if source == "gpt-5" and fake_logprobs:
            # For GPT-5 models, we need to fake logprobs
            text_content = raw_dict["choices"][0]["message"]["content"].strip()
            letter_match = re.search(r"([A-E])", text_content)
            if letter_match:
                text_content = letter_match.group(1)
                first_token = text_content.split()[0]
                raw_dict["choices"][0]["logprobs"] = {
                    "content": [
                        {"top_logprobs": [{"token": first_token, "logprob": 0.0}]}
                    ]
                }
        return self._build_normalized_response(raw_dict)

    def _is_gemini_response(self, response) -> bool:
        """Check if response is from Gemini API"""
        # Gemini responses have candidates structure but different from OpenAI/Together
        return (
            hasattr(response, "candidates")
            and hasattr(response, "text")
            and not hasattr(response, "choices")
        )

    def _normalize_gemini_response(
        self, response, fake_logprobs: bool = False
    ) -> SimpleNamespace:
        """Convert Gemini response to OpenAI-compatible format"""
        try:
            # Extract text content
            text_content = response.text if hasattr(response, "text") else ""

            # Clean up common formatting issues in Gemini responses
            if text_content.strip() and fake_logprobs:
                # Remove trailing parentheses from single letter responses (e.g., "A)" -> "A")

                letter_match = re.search(r"([A-E])", text_content)
                if letter_match:
                    text_content = letter_match.group(1)

            # Handle empty responses - check if it was blocked by safety filters
            if not text_content.strip():
                logging.warning("Gemini returned empty response")
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason:
                        logging.warning(f"Finish reason: {finish_reason}")
                    if hasattr(candidate, "safety_ratings"):
                        logging.warning(f"Safety ratings: {candidate.safety_ratings}")

                # For empty responses, use a placeholder to avoid breaking the pipeline
                text_content = "No response generated"

            # Create OpenAI-compatible structure
            normalized_dict = {
                "choices": [
                    {
                        "message": {"content": text_content, "role": "assistant"},
                        "logprobs": None,
                    }
                ]
            }

            # Fake logprobs if requested (since Gemini doesn't support them)
            if fake_logprobs and text_content.strip():
                # Create fake logprobs for the response token
                # Use the first token/word of the response
                first_token = (
                    text_content.strip().split()[0]
                    if text_content.strip()
                    else "#ERROR#"  # Default fallback
                )

                # Create fake high-probability logprobs
                fake_logprob = 0  # Very high probability (exp will be close to 1.0)
                fake_top_logprobs = [{"token": first_token, "logprob": fake_logprob}]

                normalized_dict["choices"][0]["logprobs"] = {
                    "content": [{"top_logprobs": fake_top_logprobs}]
                }

            # Note: Real Gemini logprobs parsing code removed since Gemini 2.5 doesn't support logprobs

            return self._build_normalized_response(normalized_dict)

        except Exception as e:
            logging.warning(f"Failed to normalize Gemini response: {e}")
            # Fallback: create minimal structure
            fallback_dict = {
                "choices": [
                    {
                        "message": {
                            "content": (
                                str(response) if hasattr(response, "text") else ""
                            ),
                            "role": "assistant",
                        },
                        "logprobs": None,
                    }
                ]
            }
            return self._build_normalized_response(fallback_dict)

    def _is_together_format(self, response_dict: Dict) -> bool:
        """Check if response is in TogetherAI format"""
        try:
            choices = response_dict.get("choices", [])
            if not choices:
                return False

            logprobs = choices[0].get("logprobs", {})
            # TogetherAI has 'tokens' and 'token_logprobs' in logprobs
            # OpenAI has 'content' in logprobs
            return (
                "tokens" in logprobs
                and "token_logprobs" in logprobs
                and "content" not in logprobs
            )
        except (IndexError, TypeError):
            return False

    def _together_obj_to_dict(self, response) -> Dict:
        """Convert TogetherAI response object to dictionary"""
        try:
            # Convert the TogetherAI object to dict
            if hasattr(response, "model_dump"):
                result = response.model_dump()
                return result
            elif hasattr(response, "to_dict"):
                result = response.to_dict()
                return result
            else:
                # Manual conversion for TogetherAI objects
                result = {"choices": []}

                for choice in response.choices:
                    choice_dict = {
                        "message": {
                            "content": (
                                choice.message.content
                                if hasattr(choice.message, "content")
                                else ""
                            ),
                            "role": (
                                choice.message.role
                                if hasattr(choice.message, "role")
                                else "assistant"
                            ),
                        }
                    }

                    if hasattr(choice, "logprobs") and choice.logprobs:
                        # TogetherAI format: choice.logprobs.top_logprobs is a list of dicts
                        # Like: [{'work': -0.012512207}, {'<|eot_id|>': -0.005706787}]
                        choice_dict["logprobs"] = {
                            "top_logprobs": choice.logprobs.top_logprobs
                        }

                    result["choices"].append(choice_dict)

                return result
        except Exception as e:
            logging.warning(f"Failed to convert TogetherAI object to dict: {e}")
            import traceback

            return {"choices": []}

    def _together_to_openai_dict(self, together_dict: Dict) -> Dict:
        """Convert TogetherAI response format to OpenAI format"""
        try:
            openai_dict = {"choices": []}

            for choice in together_dict.get("choices", []):
                openai_choice = {"message": choice.get("message", {}), "logprobs": None}

                if "logprobs" in choice and choice["logprobs"]:
                    together_logprobs = choice["logprobs"]

                    # TogetherAI format has 'tokens' and 'token_logprobs' arrays
                    if (
                        "tokens" in together_logprobs
                        and "token_logprobs" in together_logprobs
                    ):
                        tokens = together_logprobs["tokens"]
                        token_logprobs = together_logprobs["token_logprobs"]
                        # Convert to OpenAI format with content array
                        openai_choice["logprobs"] = {"content": []}

                        # For each token position, create a content item
                        # But typically we just need the first token for the main response
                        if len(tokens) > 0 and len(token_logprobs) > 0:
                            # Create top_logprobs list for the first token
                            top_logprobs = []

                            # The first token and its logprob
                            first_token = tokens[0]
                            first_logprob = token_logprobs[0]

                            # Add the main token
                            top_logprobs.append(
                                {"token": first_token, "logprob": first_logprob}
                            )

                            # Add other tokens if available (as alternatives)
                            for i, (token, logprob) in enumerate(
                                zip(tokens[1:], token_logprobs[1:]), 1
                            ):
                                top_logprobs.append(
                                    {"token": token, "logprob": logprob}
                                )

                            # Create the content item
                            openai_choice["logprobs"]["content"].append(
                                {"top_logprobs": top_logprobs}
                            )

                openai_dict["choices"].append(openai_choice)

            return openai_dict
        except Exception as e:
            logging.warning(
                f"Failed to convert TogetherAI format to OpenAI format: {e}"
            )
            import traceback

            return together_dict

    def _build_normalized_response(self, response_dict: Dict) -> SimpleNamespace:
        """Build the final normalized response with proper namespace structure"""
        # Convert dict to namespace recursively
        normalized = self._dict_to_namespace(response_dict)

        # Post-process to build the .content structure for logprobs
        if hasattr(normalized, "choices"):
            for choice in normalized.choices:
                if hasattr(choice, "logprobs") and choice.logprobs:
                    if hasattr(choice.logprobs, "content"):
                        # Process each content item
                        for content_item in choice.logprobs.content:
                            if hasattr(content_item, "top_logprobs"):
                                # Convert top_logprobs to TokenLogprob objects
                                token_logprobs = []
                                for logprob_item in content_item.top_logprobs:
                                    if hasattr(logprob_item, "token") and hasattr(
                                        logprob_item, "logprob"
                                    ):
                                        token_logprobs.append(
                                            TokenLogprob(
                                                token=logprob_item.token,
                                                logprob=logprob_item.logprob,
                                            )
                                        )
                                content_item.top_logprobs = token_logprobs

        return normalized

    def _dict_to_namespace(self, obj):
        """Recursively convert dict to SimpleNamespace"""
        if isinstance(obj, dict):
            namespace = SimpleNamespace()
            for key, value in obj.items():
                setattr(namespace, key, self._dict_to_namespace(value))
            return namespace
        elif isinstance(obj, list):
            return [self._dict_to_namespace(item) for item in obj]
        else:
            return obj


def create_unified_client(
    openai_api_key: str = None,
    together_api_key: str = None,
    gemini_api_key: str = None,
    timeout: int = 30,
    max_retries: int = 2,
) -> UnifiedChatClient:
    """
    Factory function to create a unified chat client.

    Args:
        openai_api_key: OpenAI API key
        together_api_key: TogetherAI API key
        gemini_api_key: Google Generative AI API key
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries

    Returns:
        UnifiedChatClient instance
    """
    return UnifiedChatClient(
        openai_api_key=openai_api_key,
        together_api_key=together_api_key,
        gemini_api_key=gemini_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
