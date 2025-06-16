import lzma
import sys
import os
import random
import asyncio
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
MODEL = sys.argv[1] # gpt-3.5-turbo-16k, gpt-4-1106-preview (= gpt4-turbo), gemini-pro
DOC_PROMPT_METHOD = sys.argv[2] # simple, latin, sft, ucn

#
# Fixed Benchmark Settings
#
GITHUB_REPO_PATH = "../datasets/kleister-nda/"
TEST_SIZE = 50

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

ucn_label_definitions = [
    LabelDefinition(
        name="effective_date",
        description="Date in YYYY-MM-DD format, at which point the contract is legally binding. Only one is correct!",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="jurisdiction",
        description="Under which state or country jurisdiction is the contract signed. Use the short name, e.g. 'Illinois' instead of 'State of Illinois', Only one is correct!",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="party",
        description="Signing counterparty of the contract. Not the party issuing the contract. Format: Replace whitespaces with '_'.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="term",
        description="Length of the legal contract as expressed in the document, e.g. '2_years'. Only one is correct!",
        datatype="string",
        is_list=False
    )
]

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
semaphore = asyncio.Semaphore(7) 
async def evaluate_sample(sample, method):
    async with semaphore:
        try:
            file_name = sample[0]
            label = sample[1]
            
            file_path = os.path.join(GITHUB_REPO_PATH, "documents/", file_name)
            if method == "ucn":
                output = await utils.ainvoke_ucn_die(
                    benchmark="kleister_nda",
                    model=MODEL, 
                    label_definitions=ucn_label_definitions,
                    document_path=file_path
                )
            else:
            images = await asyncio.to_thread(convert_from_path, file_path)
            output = await utils.ainvoke_die(
                benchmark="kleister_nda",
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
    console = Console()
    ds = load_dataset()

    # Shuffle dataset and take top n
    random.shuffle(ds)
    ds = ds[:TEST_SIZE]

    # Run evaluation in parallel
    awaitables = []
    for sample in ds[:TEST_SIZE]:
        awaitables.append(evaluate_sample(sample, DOC_PROMPT_METHOD))

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
        "Kleister NDA",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
