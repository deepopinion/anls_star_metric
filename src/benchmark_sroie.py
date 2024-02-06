import lzma
import sys
import os
import random
import asyncio
import tqdm.asyncio
import json
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from vertexai.generative_models._generative_models import ResponseBlockedError
from pydantic import BaseModel, Field

import utils
from anls_star import anls_score

#
# Configurable settings
#
MODEL = sys.argv[1] # gpt-3.5-turbo-16k, gpt-4-1106-preview (= gpt4-turbo), gemini-pro
DOC_PROMPT_METHOD = sys.argv[2] # simple, latin or sft

#
# Fixed Benchmark Settings
#
GITHUB_REPO_PATH = "../datasets/sroie/test/"
PARALLELISM = 5
TEST_SIZE = 50
MODEL = "gpt-4-1106-preview" if MODEL == "gpt-4-turbo" else MODEL
PROVIDER = "vertexai" if MODEL == "gemini-pro" else "openai"
semaphore = asyncio.Semaphore(PARALLELISM)

random.seed(42)

# Check availability of dataset
if not os.path.exists(GITHUB_REPO_PATH):
    print("Please download the dataset from https://www.kaggle.com/datasets/urbikn/sroie-datasetv2 first.")
    exit()


#
# Dataset
#  
class ModelOutput(BaseModel):
    company: str | None = Field(
        default=None,
        description="Name of the company that issued the reciept. Only one is correct! Format in upper-case letters."
    )

    date: str | None = Field(
        default=None,
        description="Date the reciept was issued. Only one is correct! Format it as found on the reciept."
    )

    address: str | None = Field(
        default=None,
        description="Address of the company that issued the reciept. Format in upper-case letters and separate information found on different lines with ','."
    )

    total: str | None = Field(
        default=None,
        description="Total amount billed. Only one is correct! Format to 2 decimal places. Do not include currency symbol."
    )


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=ModelOutput)

#
# Prepare the pipeline
#
llm = utils.create_llm(provider=PROVIDER, model=MODEL)
prompt = ChatPromptTemplate.from_messages(
    [
        (
             # Unfortunately, gemini-pro does not support the system message...
            "system" if PROVIDER == "openai" else "user",
            (
                "You are a document information extraction system.\n"
                "You are given a document and a json with keys that must be extracted from the document.\n"
                "Here is the document:\n{document}\n"
                "{format_instructions}\n"
            ),
        ),
    ]
)

chain = prompt | llm | parser

#
# MAIN
#

def load_dataset():
    # Load GTs for all images
    gt = {}
    for file_name in os.listdir(os.path.join(GITHUB_REPO_PATH, "entities/")):
        with open(os.path.join(GITHUB_REPO_PATH, "entities/", file_name), "r") as fp:
            gt[file_name.replace(".txt", ".jpg")] = json.load(fp)

    return gt


async def evaluate_sample(file_name, label):
    async with semaphore:
        try:
            file_path = os.path.join(GITHUB_REPO_PATH, "img/", file_name)
            img = Image.open(file_path)                
            prompt = await utils.doc_to_prompt(img, method=DOC_PROMPT_METHOD)
            output = await chain.ainvoke(
                {
                    "document": prompt, 
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            output = output.model_dump()

            anls = anls_score(label, output)
            return anls
        except Exception as e:
            return 0.0
            

async def main():
    ds = load_dataset()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    keys = list(ds.keys())

    # Run evaluation in parallel
    awaitables = []
    for sample_idx in idxs[:TEST_SIZE]:
        file_name = keys[sample_idx]
        label = ds[file_name]
        awaitables.append(evaluate_sample(file_name, label))

    anlss = []
    for awaitable in tqdm.asyncio.tqdm.as_completed(awaitables):
        anlss.append(await awaitable)
        anlss = [x for x in anlss if x is not None]
        tqdm.tqdm.write(f"{MODEL} | {DOC_PROMPT_METHOD} | ANLS*: {round(sum(anlss)/len(anlss), 3)}")



if __name__ == "__main__":
    asyncio.run(main())