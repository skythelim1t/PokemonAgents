"""LLM provider implementations for Anthropic, OpenAI, and AWS Bedrock."""

import base64
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: Literal["anthropic", "openai", "bedrock"]
    model: str
    temperature: float = 0.7
    max_tokens: int = 64
    api_key: str | None = None  # Only used for anthropic/openai
    region: str = "us-east-1"  # AWS region for Bedrock

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        if self.provider == "anthropic" and self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Missing ANTHROPIC_API_KEY environment variable")
        elif self.provider == "openai" and self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Missing OPENAI_API_KEY environment variable")
        # Bedrock uses default AWS credential chain, no explicit key needed


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    usage: dict[str, int]  # tokens used
    raw_response: Any  # Full response for debugging


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User message/query
            image: Optional PNG image bytes for vision models

        Returns:
            LLMResponse with content, usage, and raw response
        """
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None = None,
    ) -> LLMResponse:
        """Generate completion using Anthropic Claude."""
        if image:
            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image).decode("utf-8"),
                    },
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            content = user_prompt  # type: ignore

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        return LLMResponse(
            content=response.content[0].text,
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
            raw_response=response,
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None = None,
    ) -> LLMResponse:
        """Generate completion using OpenAI GPT."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        if image:
            content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image).decode('utf-8')}"
                    },
                },
                {"type": "text", "text": user_prompt},
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=messages,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
            raw_response=response,
        )


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider for Claude, Llama, Mistral, etc."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import boto3

            self.client = boto3.client(
                "bedrock-runtime",
                region_name=config.region,
            )
        except ImportError:
            raise ImportError("boto3 package not installed. Run: pip install boto3")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None = None,
    ) -> LLMResponse:
        """Generate completion using AWS Bedrock."""
        import json

        model_id = self.config.model

        # Determine model type and format request accordingly
        if "anthropic" in model_id.lower():
            return self._complete_anthropic(system_prompt, user_prompt, image)
        elif "meta" in model_id.lower() or "llama" in model_id.lower():
            return self._complete_llama(system_prompt, user_prompt)
        elif "mistral" in model_id.lower():
            return self._complete_mistral(system_prompt, user_prompt)
        else:
            # Default to Anthropic format
            return self._complete_anthropic(system_prompt, user_prompt, image)

    def _complete_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        image: bytes | None = None,
    ) -> LLMResponse:
        """Complete using Anthropic Claude on Bedrock."""
        import json

        if image:
            content: list[dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image).decode("utf-8"),
                    },
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            content = [{"type": "text", "text": user_prompt}]

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }

        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        return LLMResponse(
            content=result["content"][0]["text"],
            usage={
                "input": result.get("usage", {}).get("input_tokens", 0),
                "output": result.get("usage", {}).get("output_tokens", 0),
            },
            raw_response=result,
        )

    def _complete_llama(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Complete using Meta Llama on Bedrock."""
        import json

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        body = {
            "prompt": prompt,
            "max_gen_len": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        return LLMResponse(
            content=result.get("generation", ""),
            usage={
                "input": result.get("prompt_token_count", 0),
                "output": result.get("generation_token_count", 0),
            },
            raw_response=result,
        )

    def _complete_mistral(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Complete using Mistral on Bedrock."""
        import json

        prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        body = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        response = self.client.invoke_model(
            modelId=self.config.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        return LLMResponse(
            content=result.get("outputs", [{}])[0].get("text", ""),
            usage={
                "input": 0,  # Mistral doesn't return token counts
                "output": 0,
            },
            raw_response=result,
        )


def create_provider(config: LLMConfig) -> BaseLLMProvider:
    """
    Factory function to create the appropriate LLM provider.

    Args:
        config: LLM configuration

    Returns:
        Configured LLM provider instance
    """
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    elif config.provider == "openai":
        return OpenAIProvider(config)
    elif config.provider == "bedrock":
        return BedrockProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
