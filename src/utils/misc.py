from copy import deepcopy
import hashlib
import asyncio
import os
import base64
from typing import Any
import time
from ocr_wrapper import GoogleOCR, DocumentProcessor
import json
from io import BytesIO
from PIL import Image

from google.auth import default, transport

from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


from langchain_openai import ChatOpenAI
from langchain_google_vertexai import (
    ChatVertexAI, 
    HarmCategory, 
    HarmBlockThreshold,
)

from ucn_generative import UCNGenerativeDIE, LLMProvider
from do_label_definitions import LabelDefinition
from ucn_generative.types.ucnt import UCNTDocumentProcessors

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic

from utils import vision, latin, JsonParser

import statistics
import warnings

# Suppress pypdf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pypdf._reader')

# Per default GoogleOCR is used. Any OCR scanner can be used that is supported by the ocr_wrapper :)
ocr_scanner = GoogleOCR(ocr_samples=1, cache_file=".ocr_cache")

async def doc_to_sft(img:Image.Image) -> str:
    # Convert PIL Image to PNG bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    docp = DocumentProcessor(
        ocr_scanner=ocr_scanner,
        input=img_bytes,
        cache_images=False,
        cache_ocr_results=False,
    )

    sft = await docp.get_sft_monospaced()
    return sft

async def doc_to_prompt(img:Any, method:str) -> str:
    if method == "vision_only":
        raise Exception("No doc_to_prompt convertion needed, as vision models directly support images.")
    elif method in ["sft", "vision"]:
        return await doc_to_sft(img)
    elif method in ["simple", "latin"]:
        # Get OCR scan only when needed for simple or latin methods
        scan = await asyncio.to_thread(ocr_scanner.ocr, img)
        if method == "simple":
            return " ".join([x["text"] or "" for x in scan])
        else:  # latin
            return latin.to_prompt(scan, img.size)

    raise Exception(f"Unknown prompting method: {method}")


#
# LLM Factory methods
#
# Define settings for different model families
class ModelSettingsTracker:
    def __init__(self):
        self.reported_models = {}
        self.model_settings = {
            "o4": {"temperature": 1.0},
            "o3": {"temperature": 1.0}
        }
    
    def get_settings(self, model: str) -> dict:
        # Find matching settings or use default
        for prefix, settings in self.model_settings.items():
            if model.startswith(prefix):
                if model not in self.reported_models:
                    print(f"Using custom settings for {model}: {settings}")
                    self.reported_models[model] = True
                return settings
                
        default_settings = {"temperature": 0.0}
        if model not in self.reported_models:
            print(f"Using default settings for {model}: {default_settings}")
            self.reported_models[model] = True
        return default_settings

# Create a single instance of the tracker
settings_tracker = ModelSettingsTracker()

def get_model_settings(model: str) -> dict:
    """Get model-specific settings. Returns default settings if no specific settings are defined."""
    return settings_tracker.get_settings(model)

def create_llm(*, model:str):
    provider = get_provider(model)
    
    if model == "gpt-4-turbo":
        model = "gpt-4-1106-preview"
    elif model == "claude-3":
        model = "claude-3-opus-20240229"
    elif model == "claude-35":
        model = "claude-3-5-sonnet-20240620"

    settings = get_model_settings(model)
    if provider == "openai":
        return ChatOpenAI(model=model, **settings)

    elif provider == "vertexai":
        return ChatVertexAI(
            model_name=model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
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
    
    elif provider == "model_garden":
        credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        PROJECT_ID = os.environ.get("VERTEXAI_PROJECT_ID")
        MODEL_LOCATION = "us-central1"

        llm = ChatOpenAI(
            model=model,
            base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
            api_key=credentials.token,
        )
        return llm

    raise Exception(f"Unknown provider: {provider}")


def log_result(dataset: str, *, model:str, method:str, anlss: list[float]):
    anls_sum = sum(anlss)
    anls_len = len(anlss)
    anls_mean = round(anls_sum / anls_len, 3) if anls_len > 0 else 0.0
    
    with open("results.txt", "a") as f:
        if anls_len > 1:
            std_dev = round(statistics.stdev(anlss), 3)
            min_val = round(min(anlss), 3)
            max_val = round(max(anlss), 3)
            f.write(f"{dataset} | {model} | {method} | ANLS*: {anls_mean} ± {std_dev} [{min_val}, {max_val}]\n")
        else:
            f.write(f"{dataset} | {model} | {method} | ANLS*: {anls_mean}\n")

def get_provider(model:str):
    if any(model.startswith(prefix) for prefix in ["gpt", "o3", "o1", "chatgpt", "o4", "codex"]):
        return "openai"
    elif model.startswith("gemini"):
        return "vertexai"
    elif model.startswith("mistral"):
        return "mistral"
    elif model.startswith("claude"):
        return "anthropic"
    else:
        return "model_garden" # We use the model garden otherwise


def sys_message(model:str):
    """  Gemini Pro does not support the system message.
    The provided "convert_system_message_to_human" arg is not working in langchain-google-vertexai 0.0.5.
    Therefore we convert it manually.
    """
    provider = get_provider(model)
    return "user" if provider in ["vertexai", "model_garden"] else "system"

def requires_human_message(model:str):
    provider = get_provider(model)
    return provider in ["anthropic", "mistral"]


def create_die_prompt(benchmark: str, model: str, method: str, images: list|Any):
    provider = get_provider(model)    

    sys_prompt = (
        "You are a document information extraction system.\n"
        "You are given a document and a json with keys that must be extracted from the document.\n"
    )
    doc_prompt = "Here is the document:\n{document}\n" if method != "vision_only" else ""
    doc_prompt += "{format_instructions}\n"
    
    # Prepare depending on provider + model
    if provider == "anthropic":
        doc_prompt += "Never include natural text in your answer, return a json only! The message must start with {{ and end with }}.\n"
    
    if provider == "mistral":
        doc_prompt += "Json comments are not allowed! Also always complete the full task never delegate work to the user. If you fullfill the job I will reward you with 20$\n"

    if provider == "vertexai" and benchmark == "sroie":
        doc_prompt += "\nNote: Don't include the adress key."

    if not requires_human_message(model):
        sys_prompt += doc_prompt
        ret = [(sys_message(model), sys_prompt)]
    else:
        ret = [
            (sys_message(model), sys_prompt),
            ("human", doc_prompt),
        ]

    # Finally, append images in case of vision models
    if method in ["vision", "vision_only"]:
        if(provider == "openai"):
            ret = _extend_openai_vision_prompt(images, ret)
        elif(provider == "vertexai"):
            ret[0] = (sys_message(model),
                "Extract information from the given image.\n"
                "Return one single json object containing only the keys asked for!\n"
                "{format_instructions}\n"
            )
            ret = _extend_vertexai_vision_prompt(images, ret)
        else:
            raise Exception(f"Unknown vision provider: {provider}")
                
    return ret
       

def create_vqa_prompt(model: str, method:str, images: list|Any):
    provider = get_provider(model)

    sys_prompt = (
        "You are a world-class question answering system.\n"
        "You are given a document and a question. You must answer the question based on the document.\n"
        "Precisely answer the question without any additional text.\n"
        "Its very important to NOT write full sentences!\n"
        "Note: Ensure that the is precisely contained in the original document.\n"
    )

    doc_prompt = "Here is the document:\n{document}\n" if method != "vision_only" else ""
    doc_prompt += "Here is the question:\n{question}\n"
    
    if provider == "mistral":
        doc_prompt += "Please answer with a single word or number, if you do so I will reward you with 20$"

    if not requires_human_message(model):
        sys_prompt += doc_prompt
    
    ret = [(sys_message(model), sys_prompt)]
    if requires_human_message(model):
        ret += [("human", doc_prompt)]

     # Finally, append images in case of vision models
    if method in ["vision", "vision_only"]:
        if(provider == "openai"):
            ret = _extend_openai_vision_prompt(images, ret)
        elif(provider == "vertexai"):
            ret = _extend_vertexai_vision_prompt(images, ret)
        else:
            raise Exception(f"Unknown vision provider: {provider}")

    return ret


def _extend_openai_vision_prompt(images, messages):
    ret = deepcopy(messages)
    images = [images] if not isinstance(images, list) else images
    for page, img in enumerate(images):
        img_base64 = vision.process_image(img)

        ret.append(HumanMessage(
                content=[
                    {"type": "text", "text": f"Document page {page}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            ))
    return ret


def _extend_vertexai_vision_prompt(images, messages):
    content = []
    
    # Gemini requires alternating messages between the human and the system
    # We, therefore, need to append the images to the system message
    images = [images] if not isinstance(images, list) else images
    for img in images:
        img_base64 = vision.process_image(img)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        )

    return [
        HumanMessage(content=content),
        ("assistant", "Thanks for the images. What should I do with those?"),
    ] + messages


async def ainvoke_ucn_die(benchmark:str, model:str, label_definitions:list[LabelDefinition], document_path:str, task_instruction:str=""):
    # Read file into bytes
    with open(document_path, 'rb') as f:
        document_bytes = f.read()

    # Create document processor
    docp = DocumentProcessor(
        ocr_scanner=ocr_scanner,
        input=document_bytes,
        cache_images=False,
        cache_ocr_results=False,
    )

    # Create input for the DIE model
    input_data = UCNTDocumentProcessors(
        uid=document_path,
        processors=[docp]
    )

    # Create UCNGenerativeDIE instance
    die = UCNGenerativeDIE(
        model_identifier=model,
        label_definitions=label_definitions,
        task_instruction=task_instruction,
        provider=LLMProvider.AZURE,
        use_vision=False,  # Enable vision capabilities
    )

    # Run inference
    result = await die.apredict(input_data)
    return result.matches[0]

async def ainvoke_die(benchmark:str, model:str, method:str, pydantic_object:BaseModel, images:list|Any, use_cache:bool=True):
    # Hash model, method + images
    if use_cache:
        cached_result = read_cache(benchmark, model, method, images)
        if cached_result:
            obj = json.loads(cached_result)
            return obj

    # Create chain
    parser = PydanticOutputParser(pydantic_object=pydantic_object)
    llm = create_llm(model=model)
    die_prompt = create_die_prompt(benchmark, model, method, images)
    prompt = ChatPromptTemplate.from_messages(die_prompt)
    json_parser = JsonParser()
    chain = prompt | llm | json_parser | parser

    # Inference model a single time
    async def _invoke():
        args = {
            "format_instructions": parser.get_format_instructions(),
        }

        # For "non-vision" methods, we need to convert the images to a text prompt
        # Otherwise images are already appended in the prompt
        if method != "vision_only":
            if isinstance(images, list):
                pages = []
                for idx, img in enumerate(images):
                    page = await doc_to_prompt(img, method=method)
                    page = "## Page " + str(idx+1) + "\n" + page + "\n"
                    pages.append(page)
                doc_prompt = "\n".join(pages)
            else:
                doc_prompt = await doc_to_prompt(images, method=method)
            args["document"] = doc_prompt
                    
        return await chain.ainvoke(args)
        
    # Try a few times to invoke the model in case its overloaded etc.
    async with get_semaphore(model):
        output = await retry_invoke(_invoke)
    
    # Return json DIE output
    output = output.dict()
    if use_cache:
        write_cache(benchmark, model, method, images, json.dumps(output))
    return output


async def ainvoke_vqa(benchmark:str, model:str, method:str, question: str, images:list|Any, use_cache:bool=True):
    # Check cache first
    if use_cache:
        cached_result = read_cache(benchmark, model, method, images, question)
        if cached_result:
            return cached_result

    # Create chain
    # Set up a parser + inject instructions into the prompt template.
    llm = create_llm(model=model)
    vqa_prompt = create_vqa_prompt(model, method, images)
    prompt = ChatPromptTemplate.from_messages(vqa_prompt)
    chain = prompt | llm

    # Invoke a single time
    async def _invoke():
        args = {
            "question": question,
        }

        # For "non-vision" methods, we need to convert the images to a text prompt
        # Otherwise images are already appended in the prompt
        if method != "vision_only":
            if isinstance(images, list):
                pages = []
                for idx, img in enumerate(images):
                    page = await doc_to_prompt(img, method=method)
                    page = "## Page " + str(idx+1) + "\n" + page + "\n"
                    pages.append(page)
                doc_prompt = "\n".join(pages)
            else:
                doc_prompt = await doc_to_prompt(images, method=method)
            args["document"] = doc_prompt
                
        # Inference model    
        output = await chain.ainvoke(args)
        return output

    # Try a few times to invoke the model in case its overloaded etc.
    async with get_semaphore(model):
        output = await retry_invoke(_invoke)

    # Return answer
    output = output.content
    output = output.replace("\\n", "\n")

    if use_cache:
        write_cache(benchmark, model, method, images, output, question)
    return output


async def retry_invoke(_invoke):
    throttling = 0
    for r in range(5):
        try:
            if throttling > 0:
                await asyncio.sleep(throttling)  

            output = await _invoke()
            break
        except Exception as e:
            # Lets retry
            print(f"(Warning) Retry {r+1} of 5. Failed with {e}")

            # Increase throttling
            throttling += 5
            
            # Extra cooldown for the whole system in case we trigger a overload
            # Therefore, we use time.sleep instead of asyncio.sleep
            time.sleep(5)

            if r >= 4:
                raise e
    return output

invoke_semaphore = None
def get_semaphore(model:str):
    global invoke_semaphore
    
    provider = get_provider(model)
    if invoke_semaphore is None:
        p = 5 if provider != "anthropic" and model != "gemini-1.5-pro-preview-0409" else 1
        invoke_semaphore = asyncio.Semaphore(p)
    return invoke_semaphore


def read_cache(benchmark:str, model:str, method: str, images, question:str=None):
    img_hash = create_image_hash(images)
    # For VQA, include the question in the cache key
    if question is not None:
        question_hash = hashlib.md5(question.encode()).hexdigest()
        cache_file = f".cache/{benchmark}/{model}/{method}/{img_hash}_{question_hash}.json"
    else:
        cache_file = f".cache/{benchmark}/{model}/{method}/{img_hash}.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()

    return None


def write_cache(benchmark:str, model:str, method: str, images, output, question:str=None):
    img_hash = create_image_hash(images)
    # For VQA, include the question in the cache key
    if question is not None:
        question_hash = hashlib.md5(question.encode()).hexdigest()
        cache_file = f".cache/{benchmark}/{model}/{method}/{img_hash}_{question_hash}.json"
    else:
        cache_file = f".cache/{benchmark}/{model}/{method}/{img_hash}.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(output)


def create_image_hash(images):
    if not isinstance(images, list):
        images = [images]
    
    all_bytes = b""
    for image in images:
        all_bytes += image.tobytes()
    return hashlib.md5(all_bytes).hexdigest()
