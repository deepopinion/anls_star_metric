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

ucn_label_definitions = [
    LabelDefinition(
        name="address__post_town",
        description="post town of the address of the charitable organization. Only one is correct! (in upper-case letters separated with _).",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="address__postcode",
        description="postcode of the address of the charitable organization. Only one is correct! (in upper-case letters separated with _)",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="address__street_line",
        description="street line of the address of the charitable organization. Only one is correct!",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="charity_name",
        description="the name of the charitable organization. Only one is correct! (in upper-case letters separated with _)",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="charity_number",
        description="the registered number of the charitable organization. Only one is correct!",
        datatype="number",
        is_list=False
    ),
    LabelDefinition(
        name="income_annually_in_british_pounds",
        description="the annual income in British Pounds of the charitable organization. Only one is correct! Convert to float.",
        datatype="number",
        is_list=False
    ),
    LabelDefinition(
        name="report_date",
        description="the reporting date of the annual document of the charitable organization. Only one is correct! Represent the date as YYYY-MM-DD",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="spending_annually_in_british_pounds",
        description="the annual spending in British Pounds of the charitable organization. Only one is correct! Convert to float.",
        datatype="number",
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
    # This semaphore limits the memory consumption as we not load all images at once.
    async with semaphore:
        try:
            file_name = sample[0]
            label = sample[1]
            
            file_path = os.path.join(GITHUB_REPO_PATH, "documents/", file_name)
            if method == "ucn":
                output = await utils.ainvoke_ucn_die(
                    benchmark="kleister_charity",
                    model=MODEL, 
                    label_definitions=ucn_label_definitions,
                    document_path=file_path
                )
            else:
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
        "Kleister Charity",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )


if __name__ == "__main__":
    asyncio.run(main())
