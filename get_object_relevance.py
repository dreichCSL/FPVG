import os
import argparse
import copy
import time
import multiprocessing
import numpy as np

import pickle
import json
import h5py



def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict, bounding box of object
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict, bounding box of area box
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    if not (bb1['x1'] < bb1['x2']) \
            or not (bb1['y1'] < bb1['y2']) \
            or not (bb2['x1'] < bb2['x2']) \
            or not (bb2['y1'] < bb2['y2']):
        return 0.0
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = int((x_right - x_left)) * int((y_bottom - y_top))
    # compute the area of both AABBs
    bb1_area = int((bb1['x2'] - bb1['x1'])) * int((bb1['y2'] - bb1['y1']))
    bb2_area = int((bb2['x2'] - bb2['x1'])) * int((bb2['y2'] - bb2['y1']))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ## change: we want to know the area of the object's bbox that is covered by the absolute position area box
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #iou = intersection_area / float(bb1_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_overlap(bb1, bb2):
    """
    Calculate the overlap percentage of bb1 with bb2 (how much area of bb1 does bb2 cover?).
    Parameters
    ----------
    bb1 : dict, bounding box of object
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict, bounding box of area box
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    if not (bb1['x1'] < bb1['x2']) \
            or not (bb1['y1'] < bb1['y2']) \
            or not (bb2['x1'] < bb2['x2']) \
            or not (bb2['y1'] < bb2['y2']):
        return 0.0
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = int((x_right - x_left)) * int((y_bottom - y_top))
    # compute the area of both AABBs
    bb1_area = int((bb1['x2'] - bb1['x1'])) * int((bb1['y2'] - bb1['y1']))
    # bb2_area = int((bb2['x2'] - bb2['x1'])) * int((bb2['y2'] - bb2['y1']))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ## change: we want to know the area of the object's bbox that is covered by the absolute position area box
    overlap_rate = intersection_area / float(bb1_area)
    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    #iou = intersection_area / float(bb1_area)
    assert overlap_rate >= 0.0
    assert overlap_rate <= 1.0
    return overlap_rate


def get_relevant_object_indices_h5_mod(sg, qa, threshold=0.5, matching_method='iou',
                                       relevant_objects='path', feature_source="detectron", img_id_list_given=[]):
    # matching_method: overlap or iou
    # relevant_objects: path or final

    # get final object bbox for each question per img
    relevant_objects_per_img_and_question = {}

    # get reference/annotated relevant object ids per question (and image)
    for q_id in list(qa):
        img_id = qa[q_id]['imageId']
        if relevant_objects == 'path':
            object_list = list(qa[q_id]['annotations']['answer'].values())
            object_list.extend(list(qa[q_id]['annotations']['fullAnswer'].values()))
            object_list.extend(list(qa[q_id]['annotations']['question'].values()))
            object_list = list(set(object_list))
        elif relevant_objects == 'final':
            object_list = list(qa[q_id]['annotations']['answer'].values())

        if len(object_list):
            dict_entry = relevant_objects_per_img_and_question.get(img_id, {})
            dict_entry.update({q_id: object_list})
            relevant_objects_per_img_and_question[img_id] = dict_entry

    # now find overlap with detectron output objects
    img_id_list = list(relevant_objects_per_img_and_question.keys())

    # loading tsv/h5 & info_json for box location info
    if feature_source == "bottomup":
        import csv, sys, base64
        csv.field_size_limit(sys.maxsize)
        FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
                  "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
        # TODO: replace hard-coded path
        img_rep_gqa_feats = open("/home/{}/Data/VG_lxmert/vg_gqa_imgfeat/vg_gqa_obj36.tsv".format(USERNAME), 'r')

        tmp_dict_img_data = {}
        reader = csv.DictReader(img_rep_gqa_feats, FIELDNAMES, delimiter="\t")
        for item in reader:
            # if in subset, process, otherwise skip
            if relevant_objects_per_img_and_question.get(item['img_id'], None) is None:
                continue
            else:
                # process as reading from buffer file, not regular file
                num_boxes = int(item['num_boxes'])
                item['boxes'] = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32)
                item['boxes'] = item['boxes'].reshape((num_boxes, 4))
                tmp_dict_img_data[item['img_id']] = item['boxes']

    # TODO: replace hard-coded path
    elif feature_source == "detectron":
        img_rep_gqa_json = json.load(open('/home/{}/Data/GQA_mac/output_gqa_detectron_objects_info.json'.format(USERNAME), 'r'))
        img_rep_gqa_feats = h5py.File('/home/{}/Data/GQA_mac/output_gqa_detectron_objects.h5'.format(USERNAME), 'r')

    elif feature_source == "gqa":
        img_rep_gqa_json = json.load(open('/home/{}/Data/GQA/objects/gqa_objects_info.json'.format(USERNAME), 'r'))
        img_rep_gqa_feats = [h5py.File("/home/{}/Data/GQA/objects/gqa_objects_{}.h5".format(USERNAME, h5_file_idx), 'r') for h5_file_idx in range(16)]

    elif feature_source == "vinvl":
        img_rep_gqa_json = json.load(open('/home/{}/Data/GQA_mac/output_gqa_vinvl_objects_features_info.json'.format(USERNAME), 'r'))
        img_rep_gqa_feats = h5py.File('/home/{}/Data/GQA_mac/output_gqa_vinvl_objects_features.h5'.format(USERNAME), 'r')

    if len(img_id_list_given) != 0:
        new_relevant_objects_per_img_and_question = {}
        img_id_list = list(set(img_id_list) & set(img_id_list_given))
        for img_id in img_id_list:
            new_relevant_objects_per_img_and_question[img_id] = copy.deepcopy(relevant_objects_per_img_and_question[img_id])
        relevant_objects_per_img_and_question = new_relevant_objects_per_img_and_question

    for c,img_id in enumerate(img_id_list):
        if c % 1000 == 0:
            print(c)

        # feature files processing
        try:
            if feature_source == "bottomup":
                in_bboxes = tmp_dict_img_data[str(img_id)]
                num_bboxes = len(tmp_dict_img_data[str(img_id)])
            elif feature_source == 'gqa':
                img_index_h5 = img_rep_gqa_json[str(img_id)]['idx']
                img_h5_file = img_rep_gqa_json[str(img_id)]['file']
                in_bboxes = img_rep_gqa_feats[img_h5_file]['bboxes'][img_index_h5]
                num_bboxes = img_rep_gqa_json[str(img_id)]['objectsNum']
            else:
                img_index_h5 = img_rep_gqa_json[str(img_id)]['index']
                in_bboxes = img_rep_gqa_feats['bboxes'][img_index_h5]
                num_bboxes = img_rep_gqa_json[str(img_id)]['objectsNum']
        except:
            print("File for image {} not found, skipping.".format(img_id))
            continue

        # first get all ref boxes in a dict
        q_for_img = relevant_objects_per_img_and_question[img_id]
        q_id_list = list(q_for_img.keys())
        ref_box_match = {}
        for q_id in q_id_list:
            for obj_id in q_for_img[q_id]:
                #ref_box_match[obj['obj']] = None
                ref_box_match[obj_id] = None

        # then find the iou match among ref boxes (min 0.5). if not found, skip
        for obj_id in list(ref_box_match.keys()):
            best_obj_idx = []
            ref_o = sg[str(img_id)]['objects'][obj_id]
            ref_box = {'x1': ref_o['x'], 'y1': ref_o['y'], 'x2': ref_o['x'] + ref_o['w'], 'y2': ref_o['y'] + ref_o['h']}
            for box_idx, obj in enumerate(in_bboxes):
                x1, y1, x2, y2 = list(in_bboxes[box_idx])  # (4,) x,y,x2,y2
                # if boxes have no coordinates, they are empty: skip
                if np.sum([x1, y1, x2, y2]) == 0: continue
                hyp_box = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                if matching_method == 'iou':
                    overlap = get_iou(ref_box, hyp_box)
                elif matching_method == 'neg_overlap':
                    overlap = get_overlap(hyp_box, ref_box)  # how much of hyp box is relevant

                if overlap > threshold:
                    best_obj_idx.append(box_idx)
            # getting matching box index for obj_id in img_id
            if best_obj_idx is not None:
                ref_box_match[obj_id] = best_obj_idx

        # now store the found matching indices of detected objects in a dict for all questions in this image
        for q_id in list(q_for_img.keys()):
            dict_entry = q_for_img[q_id]
            try:
                box_idx_list = []
                for i in dict_entry:
                    box_idx_list.extend(ref_box_match[i])
                if matching_method == 'neg_overlap':
                    # only keep those that did not have any matches with any of the ref_boxes
                    box_idx_list = set([_ for _ in range(num_bboxes)]) - set(box_idx_list)
                q_for_img[q_id] = sorted(list(set(box_idx_list)))
            except:
                # answer object was not found with sufficient iou/overlap match among detected objects
                q_for_img[q_id] = []
        relevant_objects_per_img_and_question[img_id] = q_for_img

    return relevant_objects_per_img_and_question


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # paths to test results with all / rel / irrel objects
    parser.add_argument('--sg_input', type=str, required=True, help='path to annotated GQA scene graph')
    parser.add_argument('--qa_input', type=str, required=True, help='path to GQA Q/A annotations ')
    parser.add_argument('--threshold', type=float, default=0.5, help='overlap/iou threshold for determining match')

    parser.add_argument('--relevant_objects', type=str, default='path', choices=['path', 'final'],
                        help='which relevant objects to consider, entire inference path or only those in final step')
    parser.add_argument('--matching_method', type=str, default='iou', choices=['iou', 'neg_overlap'],
                        help='use IoU (rel) or negative overlap (irrel) to determine matching objects')
    #
    parser.add_argument('--feature_source', type=str, default='detectron', help='which object detector features to use for '
                                                                          'matching')
    parser.add_argument('--img_ids', type=list, default=[], help='only process these images in sg_input')

    parser.add_argument('--num_processes', type=int, default=1, help='number of processes in multiprocessing')
    parser.add_argument('--output', type=str, required=True, help='path to output location')

    args = parser.parse_args()


    print("Loading qa and sg annotation files.")
    sg = json.load(open(args.sg_input))
    qa = json.load(open(args.qa_input))
    # qa = json.load(open('/home/{}/Data/GQA/questions/val_balanced_questions.json'.format(USERNAME)))
    # sg = json.load(open('/home/{}/Data/GQA/sceneGraphs/val_sceneGraphs.json'.format(USERNAME)))
    print("Done.")

    if len(args.img_ids):
        img_ids = sorted(list(set(list(sg)) & set(args.img_ids)))
    else:
        img_ids = sorted(list(sg))

    # helper function for multi-processing
    def multiprocessing_func(j, img_ids, return_dict, args):
        return_dict[j] = get_relevant_object_indices_h5_mod(sg, qa, threshold=args.threshold, relevant_objects=args.relevant_objects,
                                                                    matching_method=args.matching_method, feature_source=args.feature_source,
                                                                    img_id_list_given=img_ids)


    print("Getting object relevance for {} images.".format(len(img_ids)))
    print("Processing in {} process(es).".format(args.num_processes))

    starttime = time.time()
    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    split_size = len(img_ids) // args.num_processes
    for j in range(0, args.num_processes):
        p = multiprocessing.Process(target=multiprocessing_func,
                                    args=(j, img_ids[j * split_size:(j + 1) * split_size], return_dict, args))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('That took {} seconds'.format(time.time() - starttime))
    # converting into familiar format
    rt = dict(return_dict)
    out = {}
    for i in range(len(rt)):
        out.update(rt[i])

    # write output dict[imgid][qid][list_of_rel_objects_idx] to pickle file
    pickle.dump(out, open(args.output, 'wb'), protocol=4)






