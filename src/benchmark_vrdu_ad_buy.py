import gzip
import sys
import os
import random
import asyncio
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
GITHUB_REPO_PATH = "../datasets/vrdu/ad-buy-form/main"
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
class LineItem(BaseModel):
    channel: str | None = Field(
        default=None, description="TV channel the ad will run on."
    )
    program_desc: str | None = Field(
        default=None, description="Description of the program the ad will run on."
    )
    sub_amount: str | None = Field(default=None, description="Price of this line item.")
    program_start_date: str | None = Field(
        default=None,
        description="Start date of the ad on this line item. Keep the format as found in the document.",
    )
    program_end_date: str | None = Field(
        default=None,
        description="End date of the ad on this line item. Keep the format as found in the document.",
    )


class ModelOutput(BaseModel):
    advertiser: str | None = Field(
        default=None, description="The name of the campaign that is buying the ad."
    )
    agency: str | None = Field(
        default=None,
        description="The agency buying the ad. May be the same as the advertiser, but often is a media buying agency.",
    )
    contract_num: str | None = Field(
        default=None, description="The contract number/id for the order."
    )
    property: str | None = Field(
        default=None, description="The TV station where the ad will run."
    )
    gross_amount: str | None = Field(
        default=None, description="Total amount billed on the order."
    )
    product: str | None = Field(
        default=None, description="The product being advertised."
    )
    tv_address: str | None = Field(
        default=None, description="Address of the TV station running the ad."
    )
    flight_from: str | None = Field(
        default=None,
        description="The start date of the ad campaign. Keep the format as found in the document.",
    )
    flight_to: str | None = Field(
        default=None,
        description="The end date of the ad campaign. Keep the format as found in the document.",
    )
    line_items: list[LineItem] | None = Field(
        default=None,
        description="The individual line items that make up the order.",
    )


def load_dataset():
    with gzip.open(os.path.join(GITHUB_REPO_PATH, "dataset.jsonl.gz"), "rt") as f:
        data = [json.loads(line) for line in f.readlines()]

    samples = []
    for item in data:
        kv = {}
        for key, regions in item["annotations"]:
            if isinstance(key, list):
                line_items = []
                for region in regions:
                    line_item = {}
                    assert len(key) == len(region)
                    for li_key, li_row in zip(key, region):
                        value, bbox, spans = li_row
                        line_item[li_key] = value.strip()
                    line_items.append(line_item)
                if "line_items" in kv:
                    kv["line_items"].extend(line_items)
                else:
                    kv["line_items"] = line_items
            else:
                values = []
                for region in regions:
                    value, bbox, spans = region
                    values.append(value.strip())
                values = tuple(set(values))
                kv[key] = values[0] if len(values) == 1 else values

        samples.append((item["filename"], kv))
    return samples


async def evaluate_sample(ds, idx):
    sample = ds[idx]
    try:
        file_name = sample[0]
        label = sample[1]

        file_path = os.path.join(GITHUB_REPO_PATH, "pdfs/", file_name)
        images = await asyncio.to_thread(convert_from_path, file_path)
        output = await utils.ainvoke_die(
            benchmark="vrdu_ad_buy",
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
        "VRDU AdBuy",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
