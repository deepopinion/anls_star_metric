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

ucn_label_definitions = [
    LabelDefinition(
        name="file_date",
        description="Date the registraion was field.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="foreign_principle_name",
        description="Name of the foreign principal registering.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="registrant_name",
        description="Name of the registrant.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="registration_num",
        description="Number/ID of the registration",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="signer_name",
        description="Name of the person signing the registration.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="signer_title",
        description="Title of the person signing the registration.",
        datatype="string",
        is_list=False
    )
]

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
async def evaluate_sample(ds, idx, method):
    async with semaphore:
        sample = ds[idx]
        try:
            file_name = sample[0]
            label = sample[1]

            file_path = os.path.join(GITHUB_REPO_PATH, "pdfs/", file_name)
            if method == "ucn":
                output = await utils.ainvoke_ucn_die(
                    benchmark="vrdu_registration",
                    model=MODEL,
                    label_definitions=ucn_label_definitions,
                    document_path=file_path
                )
            else:
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
        "VRDU Registration",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
