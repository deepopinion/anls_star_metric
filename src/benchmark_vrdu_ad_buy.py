import gzip
import sys
import os
import random
import asyncio
import json
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import utils
from anls_star import anls_score
from do_label_definitions import LabelDefinition

#
# Configurable settings
#
MODEL = sys.argv[1]  # gpt-3.5-turbo-16k, gpt-4-1106-preview (= gpt4-turbo), gemini-pro
DOC_PROMPT_METHOD = sys.argv[2]  # simple, latin, sft, ucn

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


ucn_label_definitions = [
    LabelDefinition(
        name="advertiser",
        description="The name of the campaign that is buying the ad.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="agency",
        description="The agency buying the ad. May be the same as the advertiser, but often is a media buying agency.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="contract_num",
        description="The contract number/id for the order.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="property",
        description="The TV station where the ad will run.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="gross_amount",
        description="Total amount billed on the order.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="product",
        description="The product being advertised.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="tv_address",
        description="Address of the TV station running the ad.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="flight_from",
        description="The start date of the ad campaign. Keep the format as found in the document.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="flight_to",
        description="The end date of the ad campaign. Keep the format as found in the document.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="line_items",
        description="The individual line items that make up the order.",
        is_list=True,
        sub=[
            LabelDefinition(
                name="channel",
                description="TV channel the ad will run on.",
                datatype="string",
                is_list=False
            ),
            LabelDefinition(
                name="program_desc",
                description="Description of the program the ad will run on.",
                datatype="string",
                is_list=False
            ),
            LabelDefinition(
                name="sub_amount",
                description="Price of this line item.",
                datatype="string",
                is_list=False
            ),
            LabelDefinition(
                name="program_start_date",
                description="Start date of the ad on this line item. Keep the format as found in the document.",
                datatype="string",
                is_list=False
            ),
            LabelDefinition(
                name="program_end_date",
                description="End date of the ad on this line item. Keep the format as found in the document.",
                datatype="string",
                is_list=False
            )
        ]
    )
]

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

semaphore = asyncio.Semaphore(7)
async def evaluate_sample(ds, idx, method):
    async with semaphore:
        sample = ds[idx]
        try:
            file_name = sample[0]
            label = sample[1]

            file_path = os.path.join(GITHUB_REPO_PATH, "pdfs/", file_name)
            if method == "ucn":
                output = await utils.ainvoke_ucn_die(
                    benchmark="vrdu_ad_buy",
                    model=MODEL,
                    label_definitions=ucn_label_definitions,
                    document_path=file_path
                )
            else:
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
    console = Console()
    ds = load_dataset()

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    # Run evaluation in parallel
    awaitables = []
    for sample_idx in idxs[:TEST_SIZE]:
        awaitables.append(evaluate_sample(ds, sample_idx, DOC_PROMPT_METHOD))

    anlss = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating {MODEL} with {DOC_PROMPT_METHOD}...", total=TEST_SIZE)
        
        for awaitable in asyncio.as_completed(awaitables):
            anls = await awaitable
            if anls is not None:
                anlss.append(anls)
                current_anls = sum(anlss)/len(anlss)
                progress.update(task, advance=1, description=f"[cyan]Evaluating {MODEL} with {DOC_PROMPT_METHOD}... [green]ANLS*: {current_anls:.3f}")

    console.print("\n[bold green]Final Results:[/bold green]")
    console.print(f"Model: [cyan]{MODEL}[/cyan]")
    console.print(f"Method: [cyan]{DOC_PROMPT_METHOD}[/cyan]")
    console.print(f"ANLS*: [bold green]{sum(anlss)/len(anlss):.3f}[/bold green]")

    utils.log_result(
        "VRDU AdBuy",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
