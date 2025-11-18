# Faithful & Plausible Visual Grounding in VQA

Scripts for calculating the FPVG metric, introduced in ["Measuring Faithful and Plausible Visual Grounding in VQA", Reich et al.](https://aclanthology.org/2023.findings-emnlp.206/).

## Prerequisites
To calculate FPVG with the scripts provided here you'll need:
- [GQA annotations](https://cs.stanford.edu/people/dorarad/gqa/download.html).
- Visual Features for GQA images (e.g. from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [VinVL](https://github.com/pzzhang/VinVL/blob/main/DOWNLOAD.md))
- A VQA model to evaluate (that can be tested with GQA and object-based visual feature representation)

### VQA model code modifications

The main changes you'll need to make to your model's code are changes in the data/feature loading pipeline such that you can: 
1. assign individual image input to each question (each question has different relevant/irrelevant objects)
2. drop objects from the visual feature input for each question (which objects to drop will be given by a python dict, see below (get_object_relevance.py) how to create)
3. (padding the reduced feature input to the expected input size with zeros - need for this step depends on how (2) was implemented)

Note that in some models you'll also be required to explicitly set the new number of valid objects in your input for *each question* (note: not just each image!), which will have changed from the original image representation after dropping objects. This is needed, as some models will otherwise interpret the 0-paddings as valid input which might impact results. 
Also note that for simplicity, we always evaluate the same set of test questions in each of the three test runs. Appropriate filtering (disregarding certain questions that e.g. did not have any objects removed) is handled in calculate_FPVG.py.

#### Format of results output file
The model's output file storing the test results should contain at least the questionId and the model's answer prediction. Default format for input to calculate_FPVG.py is a json or pickle file with contents structured as a list as follows:
```
[
  {
    'questionId': 'q_id_##',
    'prediction': 'house'
  },
  ...
]
```
- questionId: Beware: This absolutely needs to be (and always stay) a string. Some q_ids in GQA are prefixed with zeros which are lost when converting to int, which in turn renders their id ambiguous in the data set. If your evaluations return an unexpected (reduced) number of data points in evaluations, this might be a cause.
- prediction: This should be an answer (in words) that can be compared to the answers in GQA annotations. 

## get_object_relevance.py:
### Purpose
This script determines relevant (and irrelevant) objects per question, based on overlaps of bounding boxes detected by your object detector with bounding boxes annotated as relevant in your data set annotations (here: GQA's Q/A file). Detected bounding boxes are usually different across object detectors / visual features, so this step is required whenever new visual features are used for a VQA model.

The resulting file is a pickle file containing a python dict of form 
```
{
  'img_id_##': {
                'q_id_##': [obj_idx_##, ...],
                ...
                },
  ...
} 
```
The output file can be used to look up which objects (object idx in visual feature file) to keep for relevant / irrelevant testing in your VQA model's data loading pipeline.

### Examples 
Files for relevant (IoU>0.5) and irrelevant objects (overlap<0.25) are generated individually. Paths to features are currently hard-coded, so replace with paths to your local files.

For generating relevant objects:
```
python get_object_relevance.py --sg_input val_sceneGraphs.json \
                               --qa_input val_balanced_questions.json \
                               --matching_method iou \
                               --threshold 0.5 \
                               --feature_source gqa \
                               --num_processes 4 \
                               --output gqa_relevant_iou_50pct.pkl 
```

For generating irrelevant objects:
```
python get_object_relevance.py --sg_input val_sceneGraphs.json \
                               --qa_input val_balanced_questions.json \
                               --matching_method neg_overlap \
                               --threshold 0.25 \
                               --feature_source gqa \
                               --num_processes 4 \
                               --output gqa_irrelevant_neg_overlap_25pct.pkl 
```



###

## calculate_FPVG.py:
### Purpose
This script calculates the FPVG metric from the paper, given the results of three test runs (each with different visual objects in the input).

The contained FPVG function can also be used e.g. in a python interactive session to extract and investigate more detailed results, like evaluations per GQA question category (query, exist, ...) and in-depth answering behavior between the three test runs.

### Examples 
Paths to files generated by get_object_relevance.py are currently hard-coded, so replace with paths to your local files. They are accessed based on --features and are used to filter out questions that don't apply for FPVG calculation. Note that each of the three input result files are expected to contain results for the same set of questionIds.

For calculating FPVG metric with test results from three test runs:
```
python calculate_FPVG.py --test_all prediction_none_original.pkl \
                         --test_rel prediction_select_relevant_iou_50pct.pkl \
                         --test_irrel prediction_select_irrelevant_neg_overlap_25pct.pkl \
                         --features gqa
```
                         
                         


