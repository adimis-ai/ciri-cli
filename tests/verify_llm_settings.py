import os
import unittest
from unittest.mock import patch
from ciri.serializers import LLMConfig
from ciri.__main__ import _get_models_from_env


class TestLLMSettings(unittest.TestCase):
    def test_llm_config_gateway_provider_default(self):
        # Default should follow environment or 'openrouter'
        with patch.dict(os.environ, {"LLM_GATEWAY_PROVIDER": "openrouter"}):
            config = LLMConfig(model="openai/gpt-5")
            self.assertEqual(config.gateway_provider, "openrouter")

    def test_llm_config_gateway_provider_langchain(self):
        with patch.dict(os.environ, {"LLM_GATEWAY_PROVIDER": "langchain"}):
            config = LLMConfig(model="openai/gpt-5")
            self.assertEqual(config.gateway_provider, "langchain")

    @patch("src.serializers.init_chat_model")
    def test_init_langchain_model_respects_gateway(self, mock_init):
        # Test default (openrouter)
        with patch.dict(
            os.environ,
            {"LLM_GATEWAY_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test"},
        ):
            config = LLMConfig(model="openai/gpt-5")
            config.init_langchain_model()
            # For openrouter, it calls init_chat_model with model_provider="openai"
            mock_init.assert_called_with(
                model="openai/gpt-5",
                model_provider="openai",
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
            )

        mock_init.reset_mock()

        # Test langchain gateway
        with patch.dict(
            os.environ, {"LLM_GATEWAY_PROVIDER": "langchain", "OPENAI_API_KEY": "test"}
        ):
            config = LLMConfig(model="openai:gpt-5")
            config.init_langchain_model()
            # For langchain, it calls init_chat_model directly with the model string
            mock_init.assert_called_with(model="openai:gpt-5", api_key="test")

    def test_get_models_from_env_prioritize_llm_model_list(self):
        with patch.dict(
            os.environ,
            {"LLM_MODEL_LIST": "model1,model2", "LITE_LLM_MODEL_LIST": "model3,model4"},
        ):
            models = _get_models_from_env()
            self.assertEqual(models, ["model1", "model2"])

    def test_get_models_from_env_fallback_to_lite_llm_model_list(self):
        with patch.dict(
            os.environ, {"LLM_MODEL_LIST": "", "LITE_LLM_MODEL_LIST": "model3,model4"}
        ):
            models = _get_models_from_env()
            self.assertEqual(models, ["model3", "model4"])


if __name__ == "__main__":
    unittest.main()
