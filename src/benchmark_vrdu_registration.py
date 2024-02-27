import gzip
import sys
import os
import random
import asyncio
import tqdm
import tqdm.asyncio
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pdf2image import convert_from_path
from langchain.pydantic_v1 import BaseModel, Field

import utils
from anls_star import anls_score

#
# Configurable settings
#
MODEL = sys.argv[1]  # gpt-3.5-turbo-16k, gpt-4-1106-preview (= gpt4-turbo), gemini-pro
DOC_PROMPT_METHOD = sys.argv[2]  # simple, latin or sft

#
# Fixed Benchmark Settings
#
GITHUB_REPO_PATH = "../datasets/vrdu/registration-form/main"
PARALLELISM = 5
TEST_SIZE = 50
semaphore = asyncio.Semaphore(PARALLELISM)

random.seed(42)

# Check availability of dataset
if not os.path.exists(GITHUB_REPO_PATH):
    print(
        "Please download the dataset from https://github.com/google-research-datasets/vrdu first."
    )
    exit()


#
# Dataset
#
class ModelOutput(BaseModel):
    file_date: str | None = Field(
        default=None, description="Date the registraion was field. "
    )
    foreign_principle_name: str | None = Field(
        default=None, description="Name of the foreign principal registering."
    )
    registrant_name: str | None = Field(
        default=None, description="Name of the registrant."
    )
    registration_num: str | None = Field(
        default=None, description="Number/ID of the registration"
    )
    signer_name: str | None = Field(
        default=None, description="Name of the person signing the registration."
    )
    signer_title: str | None = Field(
        default=None, description="Title of the person signing the registration."
    )


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=ModelOutput)

#
# Prepare the pipeline
#
llm = utils.create_llm(model=MODEL)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            # Unfortunately, gemini-pro does not support the system message...
            utils.sys_message(MODEL),
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
    with gzip.open(os.path.join(GITHUB_REPO_PATH, "dataset.jsonl.gz"), "rt") as f:
        data = [json.loads(line) for line in f.readlines()]

    samples = []
    for item in data:
        kv = {}
        for key, regions in item["annotations"]:
            values = []
            for region in regions:
                value, bbox, spans = region
                values.append(value.strip())
            values = tuple(set(values))
            kv[key] = values[0] if len(values) == 1 else values

        samples.append((item["filename"], kv))
    return samples


async def evaluate_sample(ds, idx):
    async with semaphore:
        sample = ds[idx]
        try:
            file_name = sample[0]
            label = sample[1]

            file_path = os.path.join(GITHUB_REPO_PATH, "pdfs/", file_name)
            images = await asyncio.to_thread(convert_from_path, file_path)
            pages = []
            for page_nr, img in enumerate(images):
                page = await utils.doc_to_prompt(img, method=DOC_PROMPT_METHOD)
                page = "## Page " + str(page_nr + 1) + "\n" + page + "\n"
                pages.append(page)

            doc = "\n".join(pages)
            output = await chain.ainvoke(
                {
                    "document": doc,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            output = output.dict()

            anls = anls_score(label, output)
            return anls
        except Exception:
            # E.g. if we reach the max token limit we set a score of 0
            # If the content filter blocks the response, we also set a score of 0
            print("Error in sample: " + str(sample[0]) + ", setting score to 0")
            return 0.0


async def main():
    ds = load_dataset()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    # Run evaluation in parallel
    awaitables = []
    for sample_idx in idxs[:TEST_SIZE]:
        awaitables.append(evaluate_sample(ds, sample_idx))

    anlss = []
    for awaitable in tqdm.asyncio.tqdm.as_completed(awaitables):
        anls = await awaitable
        if anls is not None:
            anlss.append(anls)

    tqdm.tqdm.write(
        f"{MODEL} | {DOC_PROMPT_METHOD} | ANLS*: {round(sum(anlss)/len(anlss), 3)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
