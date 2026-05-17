"""AWS Bedrock client for Claude (and other Bedrock-hosted) models.

Authentication is handled by boto3's standard credential chain, so any of
these will work without code changes:
  - AWS_BEARER_TOKEN_BEDROCK  (Bedrock API key, simplest)
  - AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
  - ~/.aws/credentials profile
  - IAM instance / task role

Region resolution order: explicit ``region_name`` kwarg > AWS_REGION >
AWS_DEFAULT_REGION > boto3 default.
"""

import os
from typing import Any, Optional

from langchain_aws import ChatBedrockConverse

from .base_client import BaseLLMClient, normalize_content


_PASSTHROUGH_KWARGS = (
    "temperature", "max_tokens", "timeout", "max_retries",
    "callbacks", "region_name",
)


class NormalizedChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with normalized content output.

    Bedrock Converse returns content as typed blocks for thinking/tool-use
    models; agents downstream expect a plain string.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class BedrockClient(BaseLLMClient):
    """Client for Anthropic Claude (and other models) via AWS Bedrock."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        llm_kwargs: dict[str, Any] = {"model": self.model}

        region = (
            self.kwargs.get("region_name")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        if region:
            llm_kwargs["region_name"] = region

        for key in _PASSTHROUGH_KWARGS:
            if key == "region_name":
                continue
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        # Bedrock model IDs include version dates and cross-region inference
        # prefixes (us./eu./apac.) that change frequently; skip whitelisting.
        return True
