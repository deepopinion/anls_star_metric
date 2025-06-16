import sys
import os
import random
import asyncio
import json
import tempfile
from PIL import Image
from pdf2image import convert_from_path
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
TEST_SIZE = 50
DATASET_PATH = "../datasets/MPDocVQA"
random.seed(42)

# Check availability of dataset
val_json_file = os.path.join(DATASET_PATH, "labels", "val.json")
if not os.path.exists(val_json_file):
    print(
        "Please download the dataset from https://rrc.cvc.uab.es/?ch=17&com=introduction first."
    )
    exit()

#
# Evaluate a single sample
#
semaphore = asyncio.Semaphore(14)
async def evaluate_sample(sample, method):
    async with semaphore:
        try:
            question = sample["question"]
            answers = tuple([a for a in sample["answers"]])
            page_ids = sample["page_ids"]
            
            # Load all images first
            images = []
            for page_id in page_ids:
                file_path = os.path.join(DATASET_PATH, "documents", page_id + ".jpg")
                img = Image.open(file_path)
                images.append(img)
            
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
                
                # Create a temporary PDF with all pages
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                    temp_pdf_path = temp_pdf.name
                
                # Save as PDF
                images[0].save(temp_pdf_path, save_all=True, append_images=images[1:])
                
                try:
                    # Process the combined PDF with UCN
                    output = await utils.ainvoke_ucn_die(
                        benchmark="mp_doc_vqa",
                        model=MODEL,
                        label_definitions=label_definitions,
                        document_path=temp_pdf_path
                    )
                    answer = output.get("answer", "")
                finally:
                    # Clean up the temporary PDF
                    os.unlink(temp_pdf_path)
            else:
                answer = await utils.ainvoke_vqa(
                    benchmark="mp_doc_vqa",
                    model=MODEL,
                    method=DOC_PROMPT_METHOD,
                    question=question,
                    images=images            
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
        "MPDocVQA",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
