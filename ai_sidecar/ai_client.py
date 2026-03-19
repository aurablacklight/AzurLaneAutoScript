import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 1,
    ):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def consult(
        self,
        system_prompt: str,
        user_message: str,
        screenshot_base64: Optional[str] = None,
    ) -> Optional[str]:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
            ]

            if screenshot_base64:
                user_content = [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}",
                            "detail": "low",
                        },
                    },
                ]
                messages.append({"role": "user", "content": user_content})
            else:
                messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("AI returned empty response")
                return None
            return content

        except Exception as e:
            logger.warning(f"AI consultation failed: {e}")
            return None
