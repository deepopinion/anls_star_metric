# ANLS* - A Universal Document Processing Metric for Generative Large Language Models
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

## ANLS* score - Supported Types
Simply copy this file to your project and import the `anls_score` function from it. Then call the function with the ground truth and the predictions. 

The following types (and all combinations of it) are supported:
- `String`: To compare strings against each other using the normalized Levenshtein similarity.
- `None`: Sometimes questions are not answerable. With this type it can be checked, whether the model does not answer. Any answer other than None will be penalized.
- `Tuple`: Compare the given answer with each element in the tuple and select the element that produces the maximum ANLS* score. This is also provided by the classical ANLS metric.
- `List`: Sometimes it is required to information in the form of lists from a document. For example, extracting all purchased items found in an invoice. While the order is not important, the list should contain all items. Note that the same item can occur multiple times in lists. Hungarian matching \citep{hungarian_matching} is used to compare the ground truth and the predicted list against each other. Both missing elements as well as hallucinated elements are penalized as introduced by \citep{anlsl}.
- `Dict`: For document information extraction it is usually required to extract key-value pairs. For example, when extracting the date and total value from an invoice. Missing keys as well as hallucinated keys are penalized.

## Benchmarks

The following table shows the ANLS* score for the different models and prompt methods on different datasets. Note that we evaluate the models and prompt methods on 100 samples for single page datasets and 20 samples for multi page datasets in order to reduce the execution time and costs. Note that the provided validation set is used for the report.


<!-- Use the following page to convert to latex for the paper https://tableconvert.com/markdown-to-latex -->
| Dataset           | Method     | gpt-3.5-turbo-16k | gpt-4-turbo | gemini-pro |
| ----------------- | ---------- | ----------------- | ----------- | ---------- |
|**VQA**|
| DocVQA            | Simple          | 0.586             | 0.607       | 0.586      |
|                   | Latin Prompting | 0.659             | 0.699       | 0.676      |
|                   | SFT (Ours)      | **0.809**             | 0.790       | 0.741      |
| MPDocVQA          | Simple          | 0.348             | 0.389       | 0.389      |
|                   | Latin Prompting | 0.413             | 0.463       | 0.467      |
|                   | SFT (Ours)      | 0.547             | **0.548**       | **0.548**      |
|**Document Information Extraction**|
| Kleister Charity  | Simple          | 0.490             | 0.743       | 0.583      |
|                   | Latin Prompting | 0.442             | 0.735       | 0.478      |
|                   | SFT (Ours)      | 0.476             | **0.763**       | 0.633      |
| Kleister NDA      | Simple          | 0.343             | 0.695       | 0.623      |
|                   | Latin Prompting | 0.434             | **0.705**       | 0.599      |
|                   | SFT (Ours)      | 0.355             | 0.703       | 0.552      |
| SROIE             | Simple          | 0.874             | 0.835       | 0.263      |
|                   | Latin Prompting | 0.849             | 0.851       | 0.371      |
|                   | SFT (Ours)      | 0.893             | **0.873**       | 0.288      |
| VRDU AD Buy       | Simple          | 0.402             | 0.553       | 0.510      |
|                   | Latin Prompting | 0.389             | 0.586       | 0.556      |
|                   | SFT (Ours)      | 0.661             | **0.770**       | 0.685      |
| VRDU Registration | Simple          | 0.659             | 0.676       | 0.699      |
|                   | Latin Prompting | 0.693             | 0.673       | 0.740      |
|                   | SFT (Ours)      | **0.723**             | 0.711       | 0.720      |


### How To Execute
1. Install all dependencies via `pip install -r requirements_dev.txt`
2. Ensure that your OpenAI API key is set as environment variable `OPENAI_API_KEY`. Also, ensure that your VertexAI setup is correct in case you wanna benchmark gemini-pro too.
3. Download all datasets - the download link is provided when executing the benchmark script for the first time. Please note that the `datasets` folder should be on the same level as the repository folder.
4. Execute the corresponding benchmark script:

```bash
    python3 src/benchmark_doc_vqa.py "gpt-3.5-turbo-16k" "simple"
```

The following models are supported:
- `gpt-3.5-turbo-16k`
- `gpt-4-turbo`
- `gemini-pro`

The following prompt methods are supported:
- `simple`
- `latin`
- `sft` (DeepOpinion internal only)

5. The final ANLS* is shown on the console. 



## How to Execute all Unit Tests
To run all unit tests simply execute `pytest`
