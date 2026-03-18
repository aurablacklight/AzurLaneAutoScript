import unittest
from unittest.mock import MagicMock, patch

from ai_client import AIClient


def make_client(**kwargs):
    """Helper: create an AIClient with the OpenAI constructor patched out."""
    defaults = dict(
        base_url="http://localhost:11434/v1",
        api_key="test-key",
        model="test-model",
    )
    defaults.update(kwargs)
    with patch("ai_client.OpenAI"):
        client = AIClient(**defaults)
    return client


def make_mock_response(content: str):
    """Build a minimal mock that mimics openai.types.chat.ChatCompletion."""
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class TestAIClientBasicConsult(unittest.TestCase):
    def test_basic_consult_returns_response(self):
        """consult() returns the text content from the AI response."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response(
            "Action: click button"
        )

        result = client.consult(
            system_prompt="You are a helpful assistant.",
            user_message="What should I do?",
        )

        self.assertEqual(result, "Action: click button")

    def test_basic_consult_sends_correct_messages(self):
        """consult() sends system + user messages in the correct format."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response("ok")

        client.consult(
            system_prompt="sys prompt",
            user_message="user msg",
        )

        call_kwargs = client.client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "sys prompt"})
        self.assertEqual(messages[1], {"role": "user", "content": "user msg"})

    def test_basic_consult_uses_correct_model_and_params(self):
        """consult() passes model, temperature, and max_tokens to the API."""
        client = make_client(model="gpt-4o")
        client.client.chat.completions.create.return_value = make_mock_response("ok")

        client.consult(system_prompt="s", user_message="u")

        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-4o")
        self.assertEqual(call_kwargs["temperature"], 0.2)
        self.assertEqual(call_kwargs["max_tokens"], 256)


class TestAIClientWithScreenshot(unittest.TestCase):
    def test_consult_with_screenshot_uses_multimodal_format(self):
        """When screenshot_base64 is provided the user message is a list with text + image."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response(
            "I see the screen"
        )

        result = client.consult(
            system_prompt="Analyze the screen.",
            user_message="What is happening?",
            screenshot_base64="abc123==",
        )

        self.assertEqual(result, "I see the screen")

        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message = messages[1]
        self.assertEqual(user_message["role"], "user")
        content = user_message["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        text_part = content[0]
        self.assertEqual(text_part["type"], "text")
        self.assertEqual(text_part["text"], "What is happening?")

        image_part = content[1]
        self.assertEqual(image_part["type"], "image_url")
        self.assertEqual(
            image_part["image_url"]["url"],
            "data:image/png;base64,abc123==",
        )
        self.assertEqual(image_part["image_url"]["detail"], "low")

    def test_consult_without_screenshot_uses_plain_string(self):
        """When no screenshot is given the user message is a plain string."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response("ok")

        client.consult(system_prompt="s", user_message="plain text")

        call_kwargs = client.client.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        self.assertIsInstance(user_content, str)


class TestAIClientErrorHandling(unittest.TestCase):
    def test_api_error_returns_none(self):
        """When the API raises an exception consult() returns None."""
        client = make_client()
        client.client.chat.completions.create.side_effect = Exception("network error")

        result = client.consult(
            system_prompt="sys",
            user_message="user",
        )

        self.assertIsNone(result)

    def test_empty_response_returns_none(self):
        """When the API returns empty content consult() returns None."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response("")

        result = client.consult(
            system_prompt="sys",
            user_message="user",
        )

        self.assertIsNone(result)

    def test_none_content_returns_none(self):
        """When the API returns None as content consult() returns None."""
        client = make_client()
        client.client.chat.completions.create.return_value = make_mock_response(None)

        result = client.consult(
            system_prompt="sys",
            user_message="user",
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
