from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from lfx.base.models.anthropic_constants import ANTHROPIC_MODELS
from lfx.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS
from lfx.base.models.google_generative_ai_model import ChatGoogleGenerativeAIFixed
from lfx.base.models.model import LCModelComponent
from lfx.base.models.openai_constants import OPENAI_CHAT_MODEL_NAMES, OPENAI_REASONING_MODEL_NAMES
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput
from lfx.io import DropdownInput, MessageInput, MessageTextInput, MultilineInput, SecretStrInput, SliderInput
from lfx.schema.dotdict import dotdict


class LanguageModelComponent(LCModelComponent):
    display_name = "Language Model"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0  # Set priority to 0 to make it appear first

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Anthropic", "Google", "vLLM"],
            value="OpenAI",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[
                {"icon": "OpenAI"},
                {"icon": "Anthropic"},
                {"icon": "GoogleGenerativeAI"},
                {"icon": "vLLM"},
            ],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_CHAT_MODEL_NAMES[0],
            info="Select the model to use",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info="Custom endpoint URL (required for vLLM)",
            advanced=True,
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)

            if model_name in OPENAI_REASONING_MODEL_NAMES:
                # reasoning models do not support temperature (yet)
                temperature = None

            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key,
                base_url=self.base_url,
            )
        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )
        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAIFixed(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )
        if provider == "vLLM":
            if not self.base_url:
                msg = "Base URL is required when using vLLM provider"
                raise ValueError(msg)
            # vLLM is OpenAI-compatible, so we use ChatOpenAI with custom base_url
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key or "EMPTY",  # vLLM may not require API key
                base_url=self.base_url,
            )
        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "OpenAI":
                # Restore dropdown for non-vLLM providers
                build_config["model_name"]["type"] = "str"
                build_config["model_name"]["input_types"] = ["DropdownInput"]
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["model_name"]["real_time_refresh"] = True
                build_config["model_name"]["info"] = "Select the model to use"
                build_config["api_key"]["display_name"] = "OpenAI API Key"
            elif field_value == "Anthropic":
                # Restore dropdown for non-vLLM providers
                build_config["model_name"]["type"] = "str"
                build_config["model_name"]["input_types"] = ["DropdownInput"]
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["model_name"]["real_time_refresh"] = True
                build_config["model_name"]["info"] = "Select the model to use"
                build_config["api_key"]["display_name"] = "Anthropic API Key"
            elif field_value == "Google":
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
            elif field_value == "vLLM":
                # Change model_name to a text input for vLLM to allow custom model names
                build_config["model_name"]["type"] = "str"
                build_config["model_name"]["input_types"] = ["MessageTextInput"]
                build_config["model_name"]["value"] = ""
                build_config["model_name"]["info"] = "Enter the model name (e.g., gemini-2.5-pro)"
                # Remove dropdown-specific properties
                if "options" in build_config["model_name"]:
                    del build_config["model_name"]["options"]
                if "real_time_refresh" in build_config["model_name"]:
                    del build_config["model_name"]["real_time_refresh"]
                build_config["api_key"]["display_name"] = "API Key (Optional)"
                build_config["base_url"]["advanced"] = False
                build_config["base_url"]["info"] = "vLLM server endpoint URL (e.g., http://localhost:8000/v1)"
        elif field_name == "model_name" and field_value.startswith("o1") and self.provider == "OpenAI":
            # Hide system_message for o1 models - currently unsupported
            if "system_message" in build_config:
                build_config["system_message"]["show"] = False
        elif field_name == "model_name" and not field_value.startswith("o1") and "system_message" in build_config:
            build_config["system_message"]["show"] = True
        return build_config
