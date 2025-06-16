import sys
import os
import random
import asyncio
import json
import numpy as np
from PIL import Image
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
NUM_RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 1  # Number of runs, default to 1


#
# Fixed Benchmark Settings
#
GITHUB_REPO_PATH = "../datasets/sroie/test/"
TEST_SIZE = 50
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

ucn_label_definitions = [
    LabelDefinition(
        name="company",
        description="Name of the company that issued the reciept. Only one is correct! Format in upper-case letters.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="date", 
        description="Date the reciept was issued. Only one is correct! Format it as found on the reciept.",
        datatype="string",
        is_list=False
    ),
    LabelDefinition(
        name="address",
        description="Address of the company that issued the reciept. Format in upper-case letters and separate information found on different lines with ','.",
        datatype="string", 
        is_list=False
    ),
    LabelDefinition(
        name="total",
        description="Total amount billed. Only one is correct! Format to 2 decimal places. Do not include currency symbol.",
        datatype="string",
        is_list=False
    )
]

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

semaphore = asyncio.Semaphore(7)
async def evaluate_sample(file_name, label, method):
    async with semaphore:
        try:
            
            file_path = os.path.join(GITHUB_REPO_PATH, "img/", file_name)
            if method == "ucn":
                output = await utils.ainvoke_ucn_die(
                    benchmark="sroie",
                    model=MODEL, 
                    label_definitions=ucn_label_definitions,
                    document_path=file_path
                )
            else:
                img = Image.open(file_path)                
                output = await utils.ainvoke_die(
                    benchmark="sroie",
                    model=MODEL, 
                    method=DOC_PROMPT_METHOD, 
                    pydantic_object=ModelOutput, 
                    images=img,
                    use_cache=NUM_RUNS == 1  # Disable caching for multiple runs
                )

            anls = anls_score(label, output)
            return anls
        except Exception as e:
            print("(ERROR) " + str(e))
            return 0.0
            

async def main():
    console = Console()
    ds = load_dataset()

    all_run_anlss = []
    
    for run in range(NUM_RUNS):
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        keys = list(ds.keys())

        # Run evaluation in parallel
        awaitables = []
        for sample_idx in idxs[:TEST_SIZE]:
            file_name = keys[sample_idx]
            label = ds[file_name]
            awaitables.append(evaluate_sample(file_name, label, DOC_PROMPT_METHOD))

        anlss = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Run {run + 1}/{NUM_RUNS} - Evaluating {MODEL} with {DOC_PROMPT_METHOD}...", total=TEST_SIZE)
            
            for awaitable in asyncio.as_completed(awaitables):
                anls = await awaitable
                if anls is not None:
                    anlss.append(anls)
                    current_anls = sum(anlss)/len(anlss)
                    progress.update(task, advance=1, description=f"[cyan]Run {run + 1}/{NUM_RUNS} - Evaluating {MODEL} with {DOC_PROMPT_METHOD}... [green]ANLS*: {current_anls:.3f}")
        
        all_run_anlss.append(sum(anlss)/len(anlss))

    # Calculate mean and standard deviation for display
    mean_anls = np.mean(all_run_anlss)
    std_dev = np.std(all_run_anlss, ddof=1)

    console.print("\n[bold green]Final Results:[/bold green]")
    console.print(f"Model: [cyan]{MODEL}[/cyan]")
    console.print(f"Method: [cyan]{DOC_PROMPT_METHOD}[/cyan]")
    console.print(f"Number of runs: [cyan]{NUM_RUNS}[/cyan]")
    console.print(f"ANLS*: [bold green]{mean_anls:.3f} ± {std_dev:.3f}[/bold green]")

    utils.log_result(
        "SROIE",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=all_run_anlss
    )

if __name__ == "__main__":
    asyncio.run(main())