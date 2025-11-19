import json
import re

from langchain_core.runnables import Runnable

_json_markdown_re = re.compile(r"```(json)?(.*)```", re.DOTALL)

class JsonParser(Runnable):

    def invoke(self, *args, **kwargs) -> dict:
        try:
            text = args[0].content
            match = _json_markdown_re.search(text)
            if match:
                text = match.group(2)

            text = text.replace("\\n", "\n")
            text = text.replace("```", "")

            return text
        except Exception as e:
            raise
