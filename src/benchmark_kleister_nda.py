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
from pdf2image import convert_from_path
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
GITHUB_REPO_PATH = "../datasets/kleister-nda/"
PARALLELISM = 5
TEST_SIZE = 50
MODEL = "gpt-4-1106-preview" if MODEL == "gpt-4-turbo" else MODEL
PROVIDER = "vertexai" if MODEL == "gemini-pro" else "openai"

random.seed(42)

# Check availability of dataset
if not os.path.exists(GITHUB_REPO_PATH):
    print("Please download the dataset from https://github.com/applicaai/kleister-nda first.")
    exit()


#
# Dataset
#  
class ModelOutput(BaseModel):
    effective_date: str | None = Field(
        default=None,
        description="Date in YYYY-MM-DD format, at which point the contract is legally binding. Only one is correct!"
    )

    jurisdiction: str | None = Field(
        default=None,
        description="Under which state or country jurisdiction is the contract signed. Use the short name, e.g. 'Illinois' instead of 'State of Illinois', Only one is correct!"
    )

    party: str | None = Field(
        default=None,
        description="Signing counterparty of the contract. Not the party issuing the contract. Format: Replace whitespaces with '_'."
    )

    term: str | None = Field(
        default=None,
        description="Length of the legal contract as expressed in the document, e.g. '2_years'. Only one is correct!"
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


def load_dataset():
    in_xz_file = os.path.join(GITHUB_REPO_PATH, "train/in.tsv.xz")
    in_data = lzma.open(in_xz_file).readlines()
    in_data = [line.decode("utf-8").split("\t") for line in in_data]

    file_names = [line[0] for line in in_data]
    tsv_file = os.path.join(GITHUB_REPO_PATH, "train/expected.tsv")
    with open(tsv_file, "r") as fp:
        expected = fp.read().split("\n")
        expected = [line for line in expected if line]
        for i, sample in enumerate(expected):
            expected[i] = {x.split("=")[0]: x.split("=")[1] for x in sample.split(" ")}       

    # Ensure correctness of the dataset
    assert len(expected) == len(file_names)

    samples = list(zip(file_names, expected))
    return samples


#
# Evaluate a single sample
#
semaphore = asyncio.Semaphore(PARALLELISM)

async def evaluate_sample(sample):
    async with semaphore:
        try:
            file_name = sample[0]
            label = sample[1]
            
            file_path = os.path.join(GITHUB_REPO_PATH, "documents/", file_name)
            images = await asyncio.to_thread(convert_from_path, file_path)
            pages = []
            for idx, img in enumerate(images):
                page = await utils.doc_to_prompt(img, method=DOC_PROMPT_METHOD)
                page = "## Page " + str(idx+1) + "\n" + page + "\n"
                pages.append(page)

            doc = "\n".join(pages)
            output = await chain.ainvoke({"document": doc, "format_instructions": parser.get_format_instructions()})
            output = output.model_dump()

            anls = anls_score(label, output)
            return anls
        except Exception:
            # E.g. if we reach the max token limit we set a score of 0
            # If the content filter blocks the response, we also set a score of 0
            return 0.0


#
# MAIN
#
async def main():
    ds = load_dataset()

    # Shuffle dataset and take top n
    random.shuffle(ds)
    ds = ds[:TEST_SIZE]

    # Run evaluation in parallel
    awaitables = []
    for sample in ds[:TEST_SIZE]:
        awaitables.append(evaluate_sample(sample))

    anlss = []
    for awaitable in tqdm.asyncio.tqdm.as_completed(awaitables):
        anlss.append(await awaitable)
        anlss = [x for x in anlss if x is not None]
        tqdm.tqdm.write(f"{MODEL} | {DOC_PROMPT_METHOD} | ANLS*: {round(sum(anlss)/len(anlss), 3)}")


if __name__ == "__main__":
    asyncio.run(main())
