import gzip
import sys
import os
import random
import asyncio
import tqdm
import tqdm.asyncio
import json
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
TEST_SIZE = 50
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

semaphore = asyncio.Semaphore(7)
async def evaluate_sample(ds, idx):
    async with semaphore:
        sample = ds[idx]
        try:
            file_name = sample[0]
            label = sample[1]

            file_path = os.path.join(GITHUB_REPO_PATH, "pdfs/", file_name)
            images = await asyncio.to_thread(convert_from_path, file_path)
            output = await utils.ainvoke_die(
                benchmark="vrdu_registration",
                model=MODEL,
                method=DOC_PROMPT_METHOD,
                pydantic_object=ModelOutput,
                images=images,
            )

            anls = anls_score(label, output)
            return anls
        except Exception as e:
            print("(ERROR) " + str(e))
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

    utils.log_result(
        "VRDU Registration",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
