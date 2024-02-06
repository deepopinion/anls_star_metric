import asyncio
import multiprocessing
from PIL import Image
from ocr_wrapper import GoogleOCR

from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold

from utils import latin

try:
    from utils import sft
except ImportError:
    sft = None


# Per default GoogleOCR is used. Any OCR scanner can be used that is supported by the ocr_wrapper :)
ocr_scanner = GoogleOCR(ocr_samples=1)


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
def create_llm(*, provider: str, model: str):
    settings = {
        "temperature": 0.0,
    }
    if provider == "openai":
        return ChatOpenAI(model=model, **settings)

    elif provider == "vertexai":
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
            convert_system_message_to_human=True,
            **settings,
        )

    raise Exception(f"Unknown provider: {provider}")
