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

### Returning the closest match
The `anls_score` function can also be used to return the object which best matches the prediction and can be derived from the ground truth by re-ordering lists, selecting options from tuples etc. by setting the `return_gt` argument to `True` (default is `False`).

As an example:
```python
gt = {'a': ('hello', 'world'), 'b': ['this', 'is', 'a', 'test']}
pred = {'a': 'hello!', 'b': ['a', 'test', 'this', 'be']}
score, closest_gt = anls_score(gt, pred, return_gt=True)
# score = 0.766...
# closest_gt = {'a': 'hello', 'b': ['a', 'test', 'this', 'is']}
```

This result can then be used e.g. with the [deepdiff](https://pypi.org/project/deepdiff/) package for further analysis:
```python
from deepdiff import DeepDiff
diff = DeepDiff(closest_gt, pred)
# diff = {'values_changed': {"root['a']": {'new_value': 'hello!', 'old_value': 'hello'},
#                            "root['b'][3]": {'new_value': 'be', 'old_value': 'is'}}}
```

## Supported Types
Simply copy this file to your project and import the `anls_score` function from it. Then call the function with the ground truth and the predictions. 

The following types (and all combinations of it) are supported:
- `String`: To compare strings against each other using the normalized Levenshtein similarity.
- `None`: Sometimes questions are not answerable. With this type it can be checked, whether the model does not answer. Any answer other than None will be penalized. On the other hand, if a model generates e.g. a None key in a dictionary that is not in the ground truth, ANLS* ignores it rather than penalizing or rewarding it.
- `Tuple`: Compare the given answer with each element in the tuple and select the element that produces the maximum ANLS* score. This is also provided by the classical ANLS metric.
- `List`: Sometimes it is required to information in the form of lists from a document. For example, extracting all purchased items found in an invoice. While the order is not important, the list should contain all items. Note that the same item can occur multiple times in lists. Hungarian matching is used to compare the ground truth and the predicted list against each other. Both missing elements as well as hallucinated elements are penalized as previously introduced.
- `Dict`: For document information extraction it is usually required to extract key-value pairs. For example, when extracting the date and total value from an invoice. Missing keys as well as hallucinated keys are penalized.

## Benchmarks

The following table shows the ANLS* score for the different models and prompt methods on different datasets. Note that we evaluate the models and prompt methods on 100 samples for single page datasets and 20 samples for multi page datasets in order to reduce the execution time and costs. Note that the provided validation set is used for the report.

<!-- Note that we link here to the github repo -- then values in pypi are automatically updated without the need to release a new package -->
<img src="https://raw.githubusercontent.com/deepopinion/anls_star_metric/main/assets/table.png" alt="table" width="1000"/>


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

Note that we always benchmark the latest version of each model and report those values in the table above. In the paper, we additionally report the performance of intermediate versions of each model such as gpt-4-1106-preview and 
gpt-4-turbo-2024-04-09.


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
