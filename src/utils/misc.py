import asyncio
import os
from PIL import Image
from ocr_wrapper import GoogleOCR

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic


from utils import latin

try:
    from utils import sft
except ImportError:
    sft = None


# Per default GoogleOCR is used. Any OCR scanner can be used that is supported by the ocr_wrapper :)
ocr_scanner = GoogleOCR(ocr_samples=1, cache_file=".ocr_cache")


async def doc_to_prompt(img, method: str) -> str:
    # Get OCR scan -- we also include a large timout to ensure that the OCR scan is always finished.
    scan = await asyncio.to_thread(ocr_scanner.ocr, img)

    if method == "simple":
        return " ".join([x["text"] for x in scan])
    elif method == "latin":
        return latin.to_prompt(scan, img.size)
    elif method == "sft":
        if sft is None:
            raise Exception(
                "Please install the sft package from DeepOpinion in order to use the sft prompting method."
            )

        return sft.to_prompt(scan, img)

    raise Exception(f"Unknown prompting method: {method}")


#
# LLM Factory methods
#
def create_llm(*, model: str):
    provider = get_provider(model)
    
    if model == "gpt-4-turbo":
        model = "gpt-4-1106-preview"
    elif model == "claude-3":
        model = "claude-3-opus-20240229"

    settings = {
        "temperature": 0.0,
    }
    if provider == "openai":
        return ChatOpenAI(model=model, **settings)

    elif provider == "vertexai":
        # We decided to use the same safety settings for all providers -- the default settings.
        # safety_settings = {
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        # }

        return ChatVertexAI(
            model_name=model,
            # safety_settings=safety_settings,
            convert_system_message_to_human=True, # This parameter is still not working -- if its used an exception is raised
            **settings,
        )
    elif provider == "mistral":
        endpoint = os.environ.get("MISTRAL_ENDPOINT")
        api_key = os.environ.get("MISTRAL_API_KEY")
        return ChatMistralAI(
            model=model,
            endpoint=endpoint,
            mistral_api_key=api_key,
            **settings,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model_name=model, 
            **settings,
        )

    raise Exception(f"Unknown provider: {provider}")


def get_provider(model: str):
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("gemini"):
        return "vertexai"
    elif model.startswith("mistral"):
        return "mistral"
    elif model.startswith("claude"):
        return "anthropic"
    
    raise Exception(f"Unknown model: {model}")


def sys_message(model: str):
    """  Gemini Pro does not support the system message.
    The provided "convert_system_message_to_human" arg is not working in langchain-google-vertexai 0.0.5.
    Therefore we convert it manually.
    """
    provider = get_provider(model)
    return "user" if provider == "vertexai" else "system"

def requires_human_message(model: str):
    provider = get_provider(model)
    return provider == "anthropic"


def create_die_prompt(model: str):
    sys_prompt = (
        "You are a document information extraction system.\n"
        "You are given a document and a json with keys that must be extracted from the document.\n"
    )
    doc_prompt = (
        "Here is the document:\n{document}\n"
        "{format_instructions}\n"
    )
    if not requires_human_message(model):
        sys_prompt += doc_prompt
        return [(sys_message(model), sys_prompt)]
    
    return [
        (sys_message(model), sys_prompt),
        ("human", doc_prompt),
    ]