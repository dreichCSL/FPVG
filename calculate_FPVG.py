import os
import json
import pickle
import numpy as np
import argparse


def load_file(infile, framework=''):
    # check if input parameters are file paths (pickle/json) or contents and return accordingly
    if os.path.isfile(infile):
        try:
            f = json.load(open(infile, 'r'))
        except UnicodeDecodeError:  # pickle file?
            f = pickle.load(open(infile, 'rb'))
    else:  # e.g. if infile is given as already loaded contents in python interactive session
        f = infile

    if framework == 'visfis':
        if features == 'bottomup':
            # original visFIS release uses bottomup features (same as LXMERT)
            # also: class label2ans is different here for reduced data in GQA-101k
            ans_dicts = pickle.load(open('/share/data/VQA/visfis/gqacp/processed/trainval_label2ans.pkl', 'rb'))
        else:
            # full balanced train data used, class label2ans are different than GQA-101k
            ans_dicts = pickle.load(
                open('/share/data/VQA/negative_analysis_of_grounding/data/processed/trainval_label2ans.pkl', 'rb'))
        output = []
        for idx, qid in enumerate(f['qid']):
            # contents in file needs to be mapped from class label to actual answer
            # at the same time, reformat as list instead of original pandas format
            output.append({'questionId': qid, 'prediction': ans_dicts[f['pred_answers'][idx]]})
    elif framework == 'vlr':
        output = [{'questionId': q[0], 'prediction': q[4]} for q in f[3]]
    elif framework == 'dfol':
        # DFOL needs special processing when loading results;
        # DFOL's output doesn't match official answers in these two cases, prepare to remap them
        results_mapping_dfol = {'to the left of': 'left', 'to the right of': 'right'}
        # DFOL needs special processing when loading results
        output = []
        for q in f:
            if isinstance(q['prediction'], list):
                pred = results_mapping_dfol.get(q['prediction'][0], q['prediction'][0])
            else:
                pred = results_mapping_dfol.get(q['prediction'], q['prediction'])
            q['prediction'] = pred
            output.append(q)
    else:
        output = f

    return output


def create_helper_dicts(qa, test_all, test_rel, test_irrel):

    # prepare dicts for test_all result
    test_all_dict = {}
    test_all_result_dict = {}
    for q in test_all:
        qid = q['questionId']
        if not len(qa.get(qid, {})):
            continue
        else:
            pred = q['prediction']
        # dict for predicted answer
        test_all_dict[qid] = pred
        # dict for correctness of predicted answer
        test_all_result_dict[qid] = pred == qa[qid]['answer']

    # prepare dicts for test_rel result
    test_rel_dict = {}
    test_rel_result_dict = {}
    for q in test_rel:
        qid = q['questionId']
        if not len(qa.get(qid, {})):
            continue
        else:
            pred = q['prediction']
        # dict for predicted answer]
        test_rel_dict[qid] = pred
            # dict for correctness of predicted answer
        test_rel_result_dict[qid] = (pred == qa[qid]['answer'])

    # prepare dicts for test_irrel result
    test_irrel_dict = {}
    test_irrel_result_dict = {}
    for q in test_irrel:
        qid = q['questionId']
        if not len(qa.get(qid, {})):
            continue
        else:
            pred = q['prediction']
        # dict for predicted answer
        test_irrel_dict[qid] = pred
        # dict for correctness of predicted answer
        test_irrel_result_dict[qid] = (pred == qa[qid]['answer'])

    return test_all_dict, test_all_result_dict, \
           test_rel_dict, test_rel_result_dict, \
           test_irrel_dict, test_irrel_result_dict


def calculate_FPVG(test_all_input, test_rel_input, test_irrel_input, qa_input='', verbose=2,
                   framework='regular', test_case='balanced_val', features='detectron',
                   qa_subcats=False, return_details=False):

    test_all = load_file(test_all_input, framework=framework)
    test_rel = load_file(test_rel_input, framework=framework)
    test_irrel = load_file(test_irrel_input, framework=framework)

    # load qa if not given
    if qa_input == '':
        if test_case == 'balanced_val':
            qa = json.load(open('/home/dreich/Data/GQA/questions/val_balanced_questions.json'))
        elif test_case == 'gqa101k_id':  # special mix of GQA balanced val/train questions
            qa = json.load(open('/home/dreich/Data/GQA/questions/gqa101k_id_questions.json'))
        elif test_case == 'gqa101k_ood':  # special mix of GQA balanced val/train questions
            qa = json.load(open('/home/dreich/Data/GQA/questions/gqa101k_ood_questions.json'))
    else:
        # if given as path
        if os.path.isfile(qa_input):
            qa = load_file(qa_input)
        else:
            # if qa_input already loaded, eg in python interactive session, make deepcopy to avoid modifying original
            qa = copy.deepcopy(qa_input)

    # create helper dicts for further processing
    test_all_dict, test_all_result_dict, \
    test_rel_dict, test_rel_result_dict, \
    test_irrel_dict, test_irrel_result_dict = create_helper_dicts(qa, test_all, test_rel, test_irrel)

    if verbose >= 1:
        # Original accuracy before filtering non-grounding questions
        print("Number of questions before FPVG-required filtering: ", len(qa))
        print("ACC. all objects: {:.2f} {}".format(np.average(list(test_all_result_dict.values()))*100, len(test_all_result_dict)))
        print("ACC. only relevant objects: {:.2f} {}".format(np.average(list(test_rel_result_dict.values()))*100, len(test_rel_result_dict)))
        print("ACC. only irrelevant objects: {:.2f} {}".format(np.average(list(test_irrel_result_dict.values()))*100, len(test_irrel_result_dict)))

    if features == 'vinvl':
        filter_objects_rel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_vinvl_val_relevant_objects_path_iou_50pct.pkl','rb'))
        filter_objects_irrel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_vinvl_val_irrelevant_objects_path_neg_overlap_25pct_2.pkl','rb'))
    elif features in ['gqa', 'swapmix']:
        filter_objects_rel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_gqa_val_relevant_objects_path_iou_50pct.pkl','rb'))
        filter_objects_irrel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_gqa_val_irrelevant_objects_path_neg_overlap_25pct_2.pkl','rb'))
    elif features == 'bottomup':
        filter_objects_rel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_bottomup_val_relevant_objects_path_iou_50pct.pkl','rb'))
        filter_objects_irrel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_bottomup_val_irrelevant_objects_path_neg_overlap_25pct_2.pkl','rb'))
        if test_case in ['gqa101k_id', 'gqa101k_ood']:
            # visfis GQA-101k tests (id/ood) contain questions from both train and val set, so need to load those as well for evaluation
            filter_objects_rel.update(pickle.load(open(
                '/share/documents/dreich/Data/VG_experiments/GQA_bottomup_train_relevant_objects_path_iou_50pct.pkl',
                'rb')))
            filter_objects_irrel.update(pickle.load(open(
                '/share/documents/dreich/Data/VG_experiments/GQA_bottomup_train_irrelevant_objects_path_neg_overlap_25pct.pkl',
                'rb')))
    elif features == 'detectron':
        filter_objects_rel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_detectron_val_relevant_objects_path_iou_50pct.pkl','rb'))
        filter_objects_irrel = pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_detectron_val_irrelevant_objects_path_neg_overlap_25pct_2.pkl','rb'))
        if test_case in ['gqa101k_id', 'gqa101k_ood']:
            # visfis GQA-101k tests (id/ood) contain questions from both train and val set, so need to load those as well for evaluation
            filter_objects_rel.update(pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_detectron_train_relevant_objects_path_iou_50pct_2.pkl','rb')))
            filter_objects_irrel.update(pickle.load(open('/share/documents/dreich/Data/VG_experiments/GQA_detectron_train_irrelevant_objects_path_neg_overlap_25pct_2.pkl','rb')))

    # remove questions from FPVG eval based on available rel/irrel anno matches in object detector-provided features
    for qid in list(qa):
        try:
            # check if we found both relevant and irrelevant objects for a question, exclude from eval if not
            if len(filter_objects_rel[qa[qid]['imageId']][qid]) > 0 \
                    and len(filter_objects_irrel[qa[qid]['imageId']][qid]) > 0:
                if framework == 'dfol':
                    # This step is necessary for DFOL only, because DFOL can't handle certain questions
                    # (scene/global in particular), so there's no result. Remove these for FPVG eval.
                    if not test_all_dict.get(qid, 0):
                        del (qa[qid])
            else:  # exclude questions that have no matching relevant / non-relevant objects in the image
                del (qa[qid])

        except KeyError:
            del (qa[qid])

    # create helper dicts for further processing once again to arrive at FPVG-eval set after above filtering
    test_all_dict, test_all_result_dict, \
    test_rel_dict, test_rel_result_dict, \
    test_irrel_dict, test_irrel_result_dict = create_helper_dicts(qa, test_all, test_rel, test_irrel)


    ################################
    ####### FPVG CALCULATION #######
    ################################

    qa_subcategories = {"ALL": list(qa)}
    # prepare to additionally calculate and output FPVG per question category in GQA (if requested)
    if qa_subcats:
        for qid in list(qa):
            tmp_entry = qa_subcategories.get(qa[qid]['types']['structural'], [])
            tmp_entry.append(qid)
            qa_subcategories[qa[qid]['types']['structural']] = tmp_entry

    qid_lists = {}
    for qa_scname in list(qa_subcategories):
        qa_subcat = qa_subcategories[qa_scname]
        print("--------------", qa_scname, "---------------")

        # count changes
        c2c_rel = []
        c2i_rel = []
        i2c_rel = []
        i2i_rel = []

        c2c_irrel = []
        c2i_irrel = []
        i2c_irrel = []
        i2i_irrel = []

        for qid in list(qa_subcat):
            # c2c
            if (test_all_result_dict[qid] == test_rel_result_dict[qid] == True): c2c_rel.append(qid)
            if (test_all_result_dict[qid] == test_irrel_result_dict[qid] == True): c2c_irrel.append(qid)
            # c2i
            if (test_all_result_dict[qid] == True) and (test_rel_result_dict[qid] == False): c2i_rel.append(qid)
            if (test_all_result_dict[qid] == True) and (test_irrel_result_dict[qid] == False): c2i_irrel.append(qid)
            # i2c
            if (test_all_result_dict[qid] == False) and (test_rel_result_dict[qid] == True): i2c_rel.append(qid)
            if (test_all_result_dict[qid] == False) and (test_irrel_result_dict[qid] == True): i2c_irrel.append(qid)
            # c2c
            if (test_all_result_dict[qid] == False) and (test_rel_result_dict[qid] == False): i2i_rel.append(qid)
            if (test_all_result_dict[qid] == False) and (test_irrel_result_dict[qid] == False): i2i_irrel.append(qid)

        change_dict = {'rel': {'c2c': c2c_rel, 'c2i': c2i_rel, 'i2c': i2c_rel, 'i2i': i2i_rel},
                       'irrel': {'c2c': c2c_irrel, 'c2i': c2i_irrel, 'i2c': i2c_irrel, 'i2i': i2i_irrel}}
        if verbose == 1:
            print("Single evaluations.")
            print("Rel CC: ", len(change_dict['rel']['c2c']))
            print("Rel CI: ", len(change_dict['rel']['c2i']))
            print("Rel IC: ", len(change_dict['rel']['i2c']))
            print("Rel II: ", len(change_dict['rel']['i2i']))
            print("Irrel CC: ", len(change_dict['irrel']['c2c']))
            print("Irrel CI: ", len(change_dict['irrel']['c2i']))
            print("Irrel IC: ", len(change_dict['irrel']['i2c']))
            print("Irrel II: ", len(change_dict['irrel']['i2i']))
            print("Single evaluations DONE.")


        if verbose >= 1:
            #  Adjusted accuracy after filtering out non-grounding questions (ie only FPVG-relevant questions)
            print("Number of questions after FPVG-required filtering: ", len(qa_subcat))
            eval_tmp_list = [test_all_result_dict[eval_qid] for eval_qid in qa_subcat]
            print("ACC. all objects: {:.2f} {}".format(np.average(eval_tmp_list) * 100, len(eval_tmp_list)))
            eval_tmp_list = [test_rel_result_dict[eval_qid] for eval_qid in qa_subcat]
            print("ACC. only relevant objects: {:.2f} {}".format(np.average(eval_tmp_list) * 100, len(eval_tmp_list)))
            eval_tmp_list = [test_irrel_result_dict[eval_qid] for eval_qid in qa_subcat]
            print("ACC. only irrelevant objects: {:.2f} {}".format(np.average(eval_tmp_list) * 100, len(eval_tmp_list)))

        # CC
        # H3 cc, prove -> indeed well grounded answers (proves H3 cc)
        CC_1_list = list(set(change_dict['rel']['c2c']) & set(change_dict['irrel']['c2i']))
        CC_1 = len(CC_1_list)
        # H3 cc, disprove -> not well grounded answers (proves H1 cc)
        CC_2_list = list(set(change_dict['rel']['c2c']) & set(change_dict['irrel']['c2c']))
        CC_2 = len(CC_2_list)

        if verbose == 1:
            print("CC. Number of CCs: ", len(set(change_dict['rel']['c2c'])))
            print("CC. Good grounding and correct questions: ", CC_1)
            print("CC. Dubious grounding but correct questions: ", CC_2)

        # CI
        # H2 ci, prove:
        CI_1_list = list(set(change_dict['rel']['c2i']) & set(change_dict['irrel']['c2i']))
        CI_1 = len(CI_1_list)
        CI_1_sub1_list = list([i for i in CI_1_list if test_rel_dict[i] == test_irrel_dict[i]])
        CI_1_sub1 = len(CI_1_sub1_list)
        CI_1_sub2_list = list([i for i in CI_1_list if test_rel_dict[i] != test_irrel_dict[i]])
        CI_1_sub2 = len(CI_1_sub2_list)
        # H2 ci, disprove
        CI_2_list = list(set(change_dict['rel']['c2i']) & set(change_dict['irrel']['c2c']))
        CI_2 = len(CI_2_list)

        if verbose == 1:
            print("CI. Number of CIs: ", len(set(change_dict['rel']['c2i'])))
            print("CI. Reacts to random changes: ", CI_1)
            print("CI. CI_1_sub1. Any changes impact answer, but result in same changed answer: ", CI_1_sub1)
            print("CI. CI_1_sub2. Any changes impact answer, and both answers are different: ", CI_1_sub2)
            print("CI. Reacts to relevant changes: ", CI_2)

        # IC
        IC_1_list = list(set(change_dict['rel']['i2c']) & set(change_dict['irrel']['i2c']))
        IC_1 = len(IC_1_list)
        IC_2_list = list(set(change_dict['rel']['i2c']) & set(change_dict['irrel']['i2i']))
        IC_2 = len(IC_2_list)
        IC_2_sub1_list = list([i for i in IC_2_list if test_all_dict[i] == test_irrel_dict[i]])
        IC_2_sub1 = len(IC_2_sub1_list)
        IC_2_sub2_list = list([i for i in IC_2_list if test_all_dict[i] != test_irrel_dict[i]])
        IC_2_sub2 = len(IC_2_sub2_list)

        if verbose == 1:
            print("IC. Number of ICs: ", len(set(change_dict['rel']['i2c'])))
            print("IC. Reacts to random changes: ", IC_1)
            print("IC. Reacts to relevant changes: ", IC_2)
            print("IC. IC_2_sub1. Relevant changes impact incorrect answer, irrelevant changes don't: ", IC_2_sub1)
            print("IC. IC_2_sub2. Any changes impact incorrect answer: ", IC_2_sub2)

        # II
        # H1 ii, prove -> model not looking at V, no grounding
        II_1_list = list(set(change_dict['rel']['i2i']) & set(change_dict['irrel']['i2i']))
        II_1 = len(II_1_list)
        II_1_sub1_list = list([i for i in II_1_list if test_all_dict[i] == test_rel_dict[i] == test_irrel_dict[i]])
        II_1_sub1 = len(II_1_sub1_list)
        II_1_sub2_list = list([i for i in II_1_list if test_all_dict[i] != test_rel_dict[i] and test_all_dict[i] == test_irrel_dict[i]])
        II_1_sub2 = len(II_1_sub2_list)
        II_1_sub3_list = list([i for i in II_1_list if test_all_dict[i] == test_rel_dict[i] and test_all_dict[i] != test_irrel_dict[i]])
        II_1_sub3 = len(II_1_sub3_list)
        II_1_sub4_list = list([i for i in II_1_list if test_all_dict[i] != test_rel_dict[i] and test_all_dict[i] != test_irrel_dict[i] and test_rel_dict[i] == test_irrel_dict[i]])
        II_1_sub4 = len(II_1_sub4_list)
        II_1_sub5_list = list([i for i in II_1_list if test_all_dict[i] != test_rel_dict[i] and test_all_dict[i] != test_irrel_dict[i] and test_rel_dict[i] != test_irrel_dict[i]])
        II_1_sub5 = len(II_1_sub5_list)
        II_2_list = list(set(change_dict['rel']['i2i']) & set(change_dict['irrel']['i2c']))
        II_2 = len(II_2_list)
        II_2_sub1_list = list([i for i in II_2_list if test_all_dict[i] == test_rel_dict[i]])
        II_2_sub1 = len(II_2_sub1_list)
        II_2_sub2_list = list([i for i in II_2_list if test_all_dict[i] != test_rel_dict[i]])
        II_2_sub2 = len(II_2_sub2_list)

        if verbose == 1:
            print("II. Number of IIs: ", len(set(change_dict['rel']['i2i'])))
            print("II. Does not react to changes: ", II_1)
            print("II. II_1_sub1. Does not react to changes, incorrect answer stays same: ", II_1_sub1)
            print("II. II_1_sub2. Relevant changes impact incorrect answer, irrelevant changes don't: ", II_1_sub2)
            print("II. II_1_sub3. Relevant changes don't impact incorrect answer, irrelevant changes do: ", II_1_sub3)
            print("II. II_1_sub4. Any changes impact incorrect answer, but result in same changed answer: ", II_1_sub4)
            print("II. II_1_sub5. Any changes impact incorrect answer, and all answers are different: ", II_1_sub5)
            print("II. Reacts to random changes: ", II_2)
            print("II. II_2_sub1. Relevant changes don't impact incorrect answer, irrelevant changes do: ", II_2_sub1)
            print("II. II_2_sub2. Relevant and irrelevant changes impact answer: ", II_2_sub2)

        good_grounding_correct = CC_1
        good_grounding_wrong = II_1_sub3 + II_2_sub1
        gg = good_grounding_correct + good_grounding_wrong
        bad_grounding_correct = CC_2 + CI_2 + CI_1_sub1 + CI_1_sub2
        bad_grounding_wrong = IC_1 + IC_2 + II_2_sub2 + II_1_sub1 + II_1_sub2 + II_1_sub4 + II_1_sub5
        bg = bad_grounding_correct + bad_grounding_wrong

        # always print FPVG results
        print("GGC: {:.2f} ({})".format(good_grounding_correct*100 / len(qa_subcat), good_grounding_correct))
        print("GGW: {:.2f} ({})".format(good_grounding_wrong*100 / len(qa_subcat), good_grounding_wrong))
        print("BGC: {:.2f} ({})".format(bad_grounding_correct*100 / len(qa_subcat), bad_grounding_correct))
        print("BGW: {:.2f} ({})".format(bad_grounding_wrong*100 / len(qa_subcat), bad_grounding_wrong))
        print("Good grounding: {:.2f} ({})".format(gg*100 / len(qa_subcat), gg))
        print("Bad grounding: {:.2f} ({})".format(bg*100 / len(qa_subcat), bg))

        # all lists
        tmp_list = [
                    [CC_1_list],
                    [II_1_sub3_list, II_2_sub1_list],
                    [CC_2_list, CI_2_list, CI_1_sub1_list, CI_1_sub2_list],
                    [IC_1_list, IC_2_sub2_list, IC_2_sub1_list, II_2_sub2_list, II_1_sub1_list, II_1_sub2_list, II_1_sub4_list, II_1_sub5_list]
                    ]
        qid_lists[qa_scname] = tmp_list

    if return_details:
        return change_dict, test_all_dict, test_rel_dict, test_irrel_dict, qid_lists
    else:
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # paths to test results with all / rel / irrel objects
    parser.add_argument('--test_all', type=str, required=True, help='results with all objects')
    parser.add_argument('--test_rel', type=str, required=True, help='results with rel objects')
    parser.add_argument('--test_irrel', type=str, required=True, help='results with irrel objects')

    # Q/A annotation file
    parser.add_argument('--qa_input', type=str, default='', help='path to Q/A annotations (GQA: val_balanced_questions.json). '
                                                     'Specify path or a hard-coded one will be loaded.')
    # test name
    parser.add_argument('--test_case', type=str, default='balanced_val', help='test set to use for eval',
                        choices=['balanced_val', 'gqa101k_id', 'gqa101k_ood'])

    # related to input file format and processing
    parser.add_argument('--framework', type=str, default='', help='VQA framework / model source of test results file '
                                                                  'for handling different result input formattings')

    # influences which qids are evaluated for FPVG
    parser.add_argument('--features', type=str, default='detectron', help='selects (hard-coded) feature dependent '
                                                                          'rel/irrel object files for qid filtering. '
                                                                          'Needed files can be generated with '
                                                                          'get_object_relevance.py ')

    # for printing additional results
    parser.add_argument('--qa_subcats', action='store_true', help='for loading hard-coded feature-dependent '
                                                                          'rel/irrel object files for qid filtering')
    parser.add_argument('--verbose', type=int, default=2, help='verbosity for print messages. 0 prints least details, '
                                                               '1 prints most details, 2 default amount')

    # returns detailed lists/dict for investigation in python interactive session
    parser.add_argument('--return_details', action='store_true', help='Use in interactive session: '
                                                                      'returns qid lists and intermediate result dicts')

    args = parser.parse_args()


    out = calculate_FPVG(args.test_all, args.test_rel, args.test_irrel, qa_input=args.qa_input, verbose=args.verbose,
                         framework=args.framework, test_case=args.test_case, features=args.features,
                         qa_subcats=args.qa_subcats, return_details=False)
