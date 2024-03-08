import lzma
import sys
import os
import random
import asyncio
import tqdm.asyncio
from pdf2image import convert_from_path
from langchain.pydantic_v1 import BaseModel, Field

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
GITHUB_REPO_PATH = "../datasets/kleister-charity/"
TEST_SIZE = 50

random.seed(42)

# Check availability of dataset
if not os.path.exists(GITHUB_REPO_PATH):
    print("Please download the dataset from https://github.com/applicaai/kleister-charity first.")
    exit()


#
# Dataset
#
class ModelOutput(BaseModel):
    address__post_town: str | None = Field(
        default=None,
        description="post town of the address of the charitable organization. Only one is correct! (in upper-case letters separated with _)."
    )
    address__postcode: str | None = Field(
        default=None,
        description="postcode of the address of the charitable organization. Only one is correct! (in upper-case letters separated with _)"
    )
    address__street_line: str | None = Field(
        default=None,
        description="street line of the address of the charitable organization. Only one is correct!"
    )
    charity_name: str | None = Field(
        default=None,
        description="the name of the charitable organization. Only one is correct! (in upper-case letters separated with _)"
    )
    charity_number: int | None = Field(
        default=None,
        description="the registered number of the charitable organization. Only one is correct!"
    )
    income_annually_in_british_pounds: float | None = Field(
        default=None,
        description="the annual income in British Pounds of the charitable organization. Only one is correct! Convert to float."
    )
    report_date: str | None = Field(
        default=None,
        description="the reporting date of the annual document of the charitable organization. Only one is correct! Represent the date as YYYY-MM-DD"
    )
    spending_annually_in_british_pounds: float | None = Field(
        default=None,
        description="the annual spending in British Pounds of the charitable organization. Only one is correct! Convert to float."
    )

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
async def evaluate_sample(sample):
    try:
        file_name = sample[0]
        label = sample[1]
        
        file_path = os.path.join(GITHUB_REPO_PATH, "documents/", file_name)
        images = await asyncio.to_thread(convert_from_path, file_path)
        output = await utils.ainvoke_die(
            benchmark="kleister_charity",
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

    utils.log_result(
        "Kleister Charity",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )


if __name__ == "__main__":
    asyncio.run(main())
