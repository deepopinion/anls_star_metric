import sys
import os
import random
import asyncio
import json
from PIL import Image
from vertexai.generative_models._generative_models import ResponseBlockedError
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

if DOC_PROMPT_METHOD == "ucn":
    console = Console()
    console.print("[bold yellow]Warning:[/bold yellow] The UCN method is using DIE (Document Information Extraction) to simulate question answering. This approach may not be ideal for true question answering tasks.")

#
# Fixed Benchmark Settings
#
TEST_SIZE = 100
DATASET_PATH = "../datasets/DocVQA"
random.seed(42)

# Check availability of dataset
val_json_file = os.path.join(DATASET_PATH, "labels", "val_v1.0_withQT.json")
if not os.path.exists(val_json_file):
    print(
        "Please download the dataset from https://rrc.cvc.uab.es/?ch=17&com=introduction first."
    )
    exit()


#
# Evaluate a single sample
#
semaphore = asyncio.Semaphore(7)
async def evaluate_sample(sample, method):
    async with semaphore:
        try:
            question = sample["question"]
            answers = tuple([a for a in sample["answers"]])
            file_name = sample["image"]
            file_path = os.path.join(DATASET_PATH, file_name)

            if method == "ucn":
                # Create dynamic label definition for this question
                label_definitions = [
                    LabelDefinition(
                        name="answer",
                        description=question,
                        datatype="string",
                        is_list=False
                    )
                ]
                
                output = await utils.ainvoke_ucn_die(
                    benchmark="doc_vqa",
                    model=MODEL,
                    label_definitions=label_definitions,
                    document_path=file_path
                )
                # Extract the answer from the UCN output
                answer = output.get("answer", "")
            else:
                img = Image.open(file_path)
                answer = await utils.ainvoke_vqa(
                    benchmark="doc_vqa",
                    model=MODEL,
                    method=DOC_PROMPT_METHOD,
                    question=question,
                    images=img            
                )

            anls = anls_score(answers, answer)
            return anls
        except Exception as e:
            print("(ERROR) " + str(e))
            return 0.0


#
# MAIN
#
async def main():
    console = Console()
    with open(val_json_file, "r") as f:
        val_data = json.load(f)

    samples = val_data["data"]
    random.shuffle(samples)

    # Run evaluation in parallel
    awaitables = []
    for sample in samples[:TEST_SIZE]:
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
        "DocVQA",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )
if __name__ == "__main__":
    asyncio.run(main())
