# ANLS ★
**🌟 A Universal Document Processing Metric for Generative Large Language Models 🌟**

<div align="center">

<a href="">[![arXiv](https://img.shields.io/badge/arXiv-2402.03848-30C251.svg)](https://arxiv.org/abs/2402.03848)</a>
<a href="">![Unit Tests](https://github.com/deepopinion/anls_star_metric/actions/workflows/test.yml/badge.svg)</a>

</div>

    @misc{anls_star,
        title={ANLS* -- A Universal Document Processing Metric for Generative Large Language Models}, 
        author={David Peer and Philemon Schöpf and Volckmar Nebendahl and Alexander Rietzler and Sebastian Stabinger},
        year={2024},
        eprint={2402.03848},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

## How to use the ANLS* score?
1. Copy the file to your project: [anls_star.py](src/anls_star.py)
2. Execute `pip install munkres` or `pip install -r requirements.txt` (from this repo). 
3. Call the `anls_score` function with the ground truth and the predictions:

```python
from anls_star import anls_score
anls = anls_score("Hello World", "Hello Wrld")
print(anls)
```

4. Thats it!

## Supported Types
Simply copy this file to your project and import the `anls_score` function from it. Then call the function with the ground truth and the predictions. 

The following types (and all combinations of it) are supported:
- `String`: To compare strings against each other using the normalized Levenshtein similarity.
- `None`: Sometimes questions are not answerable. With this type it can be checked, whether the model does not answer. Any answer other than None will be penalized.
- `Tuple`: Compare the given answer with each element in the tuple and select the element that produces the maximum ANLS* score. This is also provided by the classical ANLS metric.
- `List`: Sometimes it is required to information in the form of lists from a document. For example, extracting all purchased items found in an invoice. While the order is not important, the list should contain all items. Note that the same item can occur multiple times in lists. Hungarian matching is used to compare the ground truth and the predicted list against each other. Both missing elements as well as hallucinated elements are penalized as previously introduced.
- `Dict`: For document information extraction it is usually required to extract key-value pairs. For example, when extracting the date and total value from an invoice. Missing keys as well as hallucinated keys are penalized.

## Benchmarks

The following table shows the ANLS* score for the different models and prompt methods on different datasets. Note that we evaluate the models and prompt methods on 100 samples for single page datasets and 20 samples for multi page datasets in order to reduce the execution time and costs. Note that the provided validation set is used for the report.


<!-- Use the following page to convert to latex for the paper https://tableconvert.com/markdown-to-latex -->
| Dataset           | Method          | gpt-3.5-turbo-16k | gpt-4-turbo | gpt-4-vision | gemini-pro | mistral-large  | claude-3  |
| ----------------- | --------------- | ----------------- | ----------- | ------------ | ---------- | -------------- | --------- |
|**VQA**|
| DocVQA            | Simple          | 0.586             | 0.607       | 0.759        | 0.586      | 0.445          | 0.768     |
|                   | Latin Prompting | 0.659             | 0.699       | -            | 0.676      | 0.447          | 0.762     |
|                   | SFT (Ours)      | 0.809             | 0.790       | -            | 0.741      | 0.648          | **0.831** |
| MPDocVQA          | Simple          | 0.517             | 0.635       | 0.708        | 0.603      | 0.364          | 0.636     |
|                   | Latin Prompting | 0.499             | 0.739       | -            | 0.502      | 0.335          | 0.438     |
|                   | SFT (Ours)      | 0.734             | **0.781**   | -            | 0.616      | 0.476          | 0.575     |
|**Document Information Extraction**|
| Kleister Charity  | Simple          | 0.490             | 0.743       |              | 0.583      | 0.652          | **0.800** |
|                   | Latin Prompting | 0.442             | 0.735       | -            | 0.478      | 0.576          | 0.787     |
|                   | SFT (Ours)      | 0.476             | 0.763       | -            | 0.633      | 0.657          | 0.786     |
| Kleister NDA      | Simple          | 0.343             | 0.695       |              | 0.623      | 0.637          | 0.673     |
|                   | Latin Prompting | 0.434             | **0.705**   | -            | 0.599      | 0.624          | 0.67      |
|                   | SFT (Ours)      | 0.355             | 0.703       | -            | 0.552      | 0.641          | 0.677     |
| SROIE             | Simple          | 0.874             | 0.835       |              | 0.263      | 0.855          | 0.933     |
|                   | Latin Prompting | 0.849             | 0.851       | -            | 0.371      | 0.863          | 0.926     |
|                   | SFT (Ours)      | 0.893             | 0.873       | -            | 0.288      | 0.905          | **0.949** |
| VRDU AD Buy       | Simple          | 0.402             | 0.553       |              | 0.510      | 0.386          | 0.577     |
|                   | Latin Prompting | 0.389             | 0.586       | -            | 0.556      | 0.435          | 0.608     |
|                   | SFT (Ours)      | 0.661             | **0.770**   | -            | 0.685      | 0.594          | 0.633     |
| VRDU Registration | Simple          | 0.659             | 0.676       |              | 0.699      | 0.579          | 0.685     |
|                   | Latin Prompting | 0.693             | 0.673       | -            | 0.740      | 0.587          | 0.715     |
|                   | SFT (Ours)      | **0.723**         | 0.711       | -            | 0.720      | 0.639          | 0.705     |


### How To Execute
1. Install all dependencies via `pip install -r requirements_dev.txt`
2. Setup the keys
 - OpenAI: Ensure that your OpenAI API key is set as environment variable `OPENAI_API_KEY`. 
 - Gemini: Ensure that your VertexAI setup is correct in case you wanna benchmark gemini-pro too.
 - Mistral: Setup the `MISTRAL_API_KEY` env variable as well as `MISTRAL_ENDPOINT` (Azure)
 - Anthropic: Setup the `ANTHROPIC_API_KEY` env variable
3. Download all datasets - the download link is provided when executing the benchmark script for the first time. Please note that the `datasets` folder should be on the same level as the repository folder.
4. Execute the corresponding benchmark script. For example:

```bash
    python3 src/benchmark_doc_vqa.py "gpt-3.5-turbo-16k" "simple"
```

The following models are benchmarked:
- `gpt-3.5-turbo-16k`       (Version `gpt-3.5-turbo-16k-0613` )
- `gpt-4-turbo`             (Version `gpt-4-1106-preview`)
- `gemini-pro`              (Version 1.0)
- `mistral-large`           (Version 03/2024)
- `claude-3`                (Version `claude-3-opus-20240229`)
- `gpt-4-vision-preview`    (Version `gpt-4-1106-vision-preview	`)

The following prompt methods are supported:
- `simple`
- `latin`
- `sft` (DeepOpinion internal only)
- `vision` If images should directly be used. Requires a model with vision capabilities e.g. gpt-4-vision

5. The final ANLS* is shown on the console. 



## How to Execute all Unit Tests
To run all unit tests simply execute `pytest`
