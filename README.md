# ANLS ★
**🌟 A Universal Metric for Generative Large Language Models 🌟**

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
1. `pip install anls_star`
2. Add to your code

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
| Dataset           | Method          | gpt-3.5-turbo-16k | gpt-4-turbo | gpt-4-vision | gemini-v1.0-pro | gemini-v1.5-pro | mistral-large  | claude-3  |
| ----------------- | --------------- | ----------------- | ----------- | ------------ | --------------- | --------------- | -------------- | --------- |
| DocVQA            | Simple          | 0.576             | 0.607       | 0.759        | 0.586           | 0.647           | 0.445          | 0.768     |
|                   | Latin Prompting | 0.62              | 0.699       | -            | 0.676           | 0.668           | 0.447          | 0.762     |
|                   | SFT (Ours)      | 0.768             | 0.790       | -            | 0.741           | 0.737           | 0.648          | **0.831** |
| MPDocVQA          | Simple          | 0.507             | 0.635       | 0.708        | 0.603           | 0.627           | 0.364          | 0.636     |
|                   | Latin Prompting | 0.489             | 0.739       | -            | 0.502           | 0.726           | 0.335          | 0.438     |
|                   | SFT (Ours)      | 0.724             | 0.781       | -            | 0.616           | **0.794**       | 0.476          | 0.575     |
| Kleister Charity  | Simple          | 0.527             | 0.743       | 0.751        | 0.583           | 0.764           | 0.652          | **0.800** |
|                   | Latin Prompting | 0.477             | 0.735       | -            | 0.478           | 0.763           | 0.576          | 0.787     |
|                   | SFT (Ours)      | 0.534             | 0.763       | -            | 0.633           | 0.779           | 0.657          | 0.786     |
| Kleister NDA      | Simple          | 0.361             | 0.695       | 0.664        | 0.623           | 0.722           | 0.637          | 0.673     |
|                   | Latin Prompting | 0.412             | 0.705       | -            | 0.599           | 0.707           | 0.624          | 0.67      |
|                   | SFT (Ours)      | 0.315             | 0.703       | -            | 0.552           | **0.715**       | 0.641          | 0.677     |
| SROIE             | Simple          | 0.509             | 0.835       | 0.834        | 0.263           | 0.594           | 0.855          | 0.933     |
|                   | Latin Prompting | 0.723             | 0.851       | -            | 0.371           | 0.623           | 0.863          | 0.926     |
|                   | SFT (Ours)      | 0.792             | 0.873       | -            | 0.288           | 0.623           | 0.905          | **0.949** |
| VRDU AD Buy       | Simple          | 0.414             | 0.553       | 0.640        | 0.510           | 0.645           | 0.386          | 0.577     |
|                   | Latin Prompting | 0.424             | 0.586       | -            | 0.556           | 0.686           | 0.435          | 0.608     |
|                   | SFT (Ours)      | 0.676             | **0.770**   | -            | 0.685           | 0.765           | 0.594          | 0.633     |
| VRDU Registration | Simple          | 0.586             | 0.676       | 0.665        | 0.699           | 0.730           | 0.579          | 0.685     |
|                   | Latin Prompting | 0.606             | 0.673       | -            | 0.740           | 0.749           | 0.587          | 0.715     |
|                   | SFT (Ours)      | 0.637             | 0.711       | -            | 0.720           | **0.780**       | 0.639          | 0.705     |


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

The following model versions are benchmarked:
- `gpt-3.5-turbo-16k`       (Version `gpt-3.5-turbo-16k-0613` )
- `gpt-4-turbo`             (Version `gpt-4-1106-preview`)
- `gemini-v1.0-pro`         (Version `gemini-1.0-pro`)
- `mistral-large`           (Version 03/2024)
- `claude-3`                (Version `claude-3-opus-20240229`)
- `gpt-4-vision`            (Version `gpt-4-1106-vision-preview	`)
- `gemini-v1.5-pro`         (Version `gemini-1.5-pro-preview-0409`)

The following prompt methods are supported:
- `simple` - Simple text concatenation after OCR with GooleOCR
- `latin` - Method as introduced by [Wang et al.](https://arxiv.org/abs/2306.00526)
- `sft` - DeepOpinion internal only
- `vision` - If images should directly be used. Requires a model with vision capabilities e.g. gpt-4-vision

5. The final ANLS* is shown on the console. 



## How to Execute all Unit Tests
To run all unit tests simply execute `pytest`


## Packaging
See https://packaging.python.org/en/latest/tutorials/packaging-projects/