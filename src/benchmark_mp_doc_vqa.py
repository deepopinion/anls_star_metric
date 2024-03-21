import sys
import os
import random
import asyncio
import tqdm.asyncio
import json
from PIL import Image

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
semaphore = asyncio.Semaphore(10)
async def evaluate_sample(sample):
    async with semaphore:
        try:
            question = sample["question"]
            answers = tuple([a for a in sample["answers"]])
            page_ids = sample["page_ids"]
            images = []
            for page_id in page_ids:
                file_path = os.path.join(DATASET_PATH, "documents", page_id + ".jpg")
                img = Image.open(file_path)
                images.append(img)

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
    with open(val_json_file, "r") as f:
        val_data = json.load(f)

    samples = val_data["data"]
    random.shuffle(samples)

    # Run evaluation in parallel
    awaitables = []
    for sample in samples[:TEST_SIZE]:
        awaitables.append(evaluate_sample(sample))

    anlss = []
    for awaitable in tqdm.asyncio.tqdm.as_completed(awaitables):
        anlss.append(await awaitable)
        anlss = [x for x in anlss if x is not None]
        tqdm.tqdm.write(f"{MODEL} | {DOC_PROMPT_METHOD} | ANLS*: {round(sum(anlss)/len(anlss), 3)}")

    utils.log_result(
        "MPDocVQA",
        model=MODEL, 
        method=DOC_PROMPT_METHOD, 
        anlss=anlss,
    )

if __name__ == "__main__":
    asyncio.run(main())
