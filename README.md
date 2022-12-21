# Faithful & Plausible Visual Grounding in VQA

Scripts for calculating the FPVG metric, introduced in ["Faithful & Plausible Visual Grounding in VQA", Reich et al.](http://arxiv.org/).

## Prerequisites
You'll need:
1. GQA annotations
2. Visual Features for GQA images
3. A VQA model to evaluate (that can be tested with GQA and object-based visual feature representation)

The main changes you'll need to make to your model's code are changes in the data/feature loading pipeline such that you can: 
1. assign individual image input to each question (each question has different relevant/irrelevant objects)
2. drop objects from the visual feature input for each question (which objects to drop can be given by a python dict, see below how to create)
3. (padding the reduced feature input to the expected input size with zeros - need for this step depends on how (2) was implemented)

Note that in some models you'll also be required to explicitly set the new number of valid objects in your input for *each question* (note: not just each image!), which will have changed from the original image representation. This is needed, as some models will otherwise interpret the 0-paddings as valid input which might impact results. 


## get_object_relevance.py:
### Purpose of get_object_relevance.py:
This script determines relevant (and irrelevant) objects per question, based on overlaps of bounding boxes detected by your object detector with bounding boxes annotated as relevant in your data set annotations (here: GQA's Q/A file). Detected bounding boxes are usually different across object detectors / visual features, so this step is required whenever new visual features are used for a VQA model.

The resulting files is a pickle file containing a python dict of form {'img_id_##': {'q_id_##': [obj_idx_##, obj_idx_##, ...]}}. This file can be used to tell your VQA model's data loading pipeline which objects (object idx in visual feature file) to keep for relevant / irrelevant testing.

###

## calculate_FPVG.py:
### Purpose of calculate_FPVG.py:
This script calculates the FPVG metric from the paper, given the results of three test runs (each with different visual objects in the input).

The contained FPVG function can also be used e.g. in a python interactive session to extract and investigate more detailed results, like evaluations per GQA question category (query, exist, ...) and in-depth answering behavior between the three test runs.

###




