import sys
import os
import random
import asyncio
import tqdm.asyncio
import json
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from vertexai.generative_models._generative_models import ResponseBlockedError

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
PARALLELISM = 4
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
# Prepare the pipeline
#
llm = utils.create_llm(model=MODEL)
prompt = ChatPromptTemplate.from_messages(
    [
        (
             # Unfortunately, gemini-pro does not support the system message...
            utils.sys_message(MODEL),
            (
                "You are a world-class question answering system.\n"
                "You are given a document and a question. You must answer the question based on the document.\n"
                "Precisely answer the question without any additional text.\n"
                "Its very important to NOT write full sentences!\n"
                "Note: Ensure that the is precisely contained in the original document.\n"
                "Here is the document:\n{document}\n"
                "Here is the question:\n{question}\n"
            ),
        ),
    ]
)
chain = prompt | llm


#
# Evaluate a single sample
#
semaphore = asyncio.Semaphore(PARALLELISM)

async def evaluate_sample(sample):
    async with semaphore:
        try:
            question = sample["question"]
            answers = tuple([a for a in sample["answers"]])
            page_ids = sample["page_ids"]

            pages = []
            for idx, page_id in enumerate(page_ids):
                file_path = os.path.join(DATASET_PATH, "documents", page_id + ".jpg")
                img = Image.open(file_path)
                page = await utils.doc_to_prompt(img, method=DOC_PROMPT_METHOD)
                pages.append(f"\n## Page {idx+1}\n" + page)

            page = "\n".join(pages)
            img = Image.open(file_path)
            page = await utils.doc_to_prompt(img, method=DOC_PROMPT_METHOD)
            answer = await chain.ainvoke({"document": page, "question": question})
            answer = answer.content

            anls = anls_score(answers, answer)
            return anls
        except Exception as e:
            print(f"(Warning) Failed to process sample: {e}")
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


if __name__ == "__main__":
    asyncio.run(main())
