from typing import List, Dict, Any

from together import Together


class HFClient:
    def _rewrite_history(self, messages: List[Dict[str, Any]]) -> list[dict[str, list[dict[str, str | Any]] | Any]]:
        return [{
            "role": message['role'],
            "content": [{
                "type": "text",
                "text": message['content']
            }] if idx == len(messages) else message['content']
        }
            for idx, message in enumerate(messages)
        ]

    def stream(self, messages: List[Dict[str, Any]], stream: bool = True):
        client = Together(
            api_key="b06e5d4320ab888bcd7331308c0fbb2f3b42b994f46408dce4092a797d6bdb3c"
        )
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            stream=stream
        )
        if stream:
            for line in response:
                yield line.choices[0].delta.content or None
        else:
            yield response.choices[0].message.content or None


hf_client = HFClient()
