from dataset import load_utf16le_data_to_list, get_ner_labels_from_file, NER_LabelEncode
from constant import RAW_DATA
from typing import List, Dict
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score
from copy import deepcopy
from collections import namedtuple, Counter
import pandas as pd

# CWS Function

# from segment to evaluation-able format
# e.g.
# Gold: 計算機 總是 有問題  => (1, 4) (4, 6) (6, 9)
# Predict: 計算機 總 是 有問題 => (1, 4) (4, 5) (5, 6) (6, 9)
# correct: (1, 4) (6, 9) error: (4, 5) (5, 6)


def _toSegEvalFormat(string_list: List[str]):
    eval_format_list = []
    word_count = 1  # start from 1
    for string in string_list:
        start = word_count
        word_count += len(string)
        end = word_count
        eval_format_list.append((start, end))
    return eval_format_list


def _scorerSingle(pred_eval: List[str], gold_eval: List[str]):
    e = 0
    c = 0
    N = len(gold_eval)

    for pred_start_end in pred_eval:
        if pred_start_end in gold_eval:
            c += 1
        else:
            e += 1

    return e, c, N


def _scorer(pred_eval_list: List[List[str]], gold_eval_list: List[List[str]]):
    N = 0  # gold segment words number
    e = 0  # wrong number of word segment
    c = 0  # correct number of word segment

    for pred_eval, gold_eval in zip(pred_eval_list, gold_eval_list):
        temp_e, temp_c, temp_N = _scorerSingle(pred_eval, gold_eval)

        N += temp_N
        e += temp_e
        c += temp_c

    R = c/N
    P = c/(c+e)
    F1 = (2*P*R)/(P+R)
    ER = e/N

    return R, P, F1, ER


def wordSegmentEvaluaiton(pred_seg_list: List[str], gold_seg_list: List[str]):

    pred_eval_list = []
    gold_eval_list = []
    for pred_string, gold_string in zip(pred_seg_list, gold_seg_list):
        pred_list = pred_string.split()
        gold_list = gold_string.split()
        pred_eval_list.append(_toSegEvalFormat(pred_list))
        gold_eval_list.append(_toSegEvalFormat(gold_list))

    P, R, F1, ER = _scorer(pred_eval_list, gold_eval_list)

    print('=== Evaluation reault of word segment ===')
    print('F1: %.2f%%' % (F1*100))
    print('P : %.2f%%' % (P*100))
    print('R : %.2f%%' % (R*100))
    print('ER: %.2f%%' % (ER*100))
    print('=========================================')

# NER Function


NOT_NER_TAG = 'N'  # or 'O' in different dataset


def per_token_eval(pred_ner_labels: List[str], gold_ner_labels: List[str], labels: List[str]):
    sorted_labels = sorted(labels, key=lambda name: (
        name[1:], name[0]))  # group B and I results
    return flat_classification_report(pred_ner_labels, gold_ner_labels, labels=sorted_labels, digits=4)


Entity = namedtuple("Entity", "e_type start_offset end_offset")


def collect_named_entities(tokens: List[str]):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.
    :param tokens: a list of labels
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == NOT_NER_TAG:
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(
                    Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities


def compute_metrics(true_named_entities: List[Entity], pred_named_entities: List[Entity], labels: List[str]):
    eval_metrics = {'correct': 0, 'incorrect': 0,
                    'partial': 0, 'missed': 0, 'spurious': 0}

    # ['PER', 'LOC', 'ORG']
    target_tags_no_schema = list(
        Counter([label[2:] for label in labels]).keys())

    # overall results
    evaluation = {'strict': deepcopy(eval_metrics),
                  'ent_type': deepcopy(eval_metrics),
                  'partial': deepcopy(eval_metrics),
                  'exact': deepcopy(eval_metrics)}

    # results by entity type
    evaluation_agg_entities_type = {e: deepcopy(
        evaluation) for e in target_tags_no_schema}

    true_which_overlapped_with_pred = []  # keep track of entities that overlapped

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif _find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        found_overlap = True
                        break

                    # Scenario VI: Entities overlap, but the entity type is
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True
                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:
                # overall results
                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['ent_type']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['partial']['spurious'] += 1
                evaluation_agg_entities_type[pred.e_type]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = _compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:

            evaluation_agg_entities_type[entity_type][eval_type] = _compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def _find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    Examples:
    >>> _find_overlap((1, 2), (2, 3))
    2
    >>> _find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def _compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def _compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.
    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the _compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: _compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: _compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results


def result_over_all_sentences(gold_ner_labels: List[List[str]], pred_ner_labels: List[List[str]], labels: List[str]):
    metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                       'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0}

    # ['PER', 'LOC', 'ORG']
    target_tags_no_schema = list(
        Counter([label[2:] for label in labels]).keys())

    # overall results
    results = {'strict': deepcopy(metrics_results),
               'ent_type': deepcopy(metrics_results),
               'partial': deepcopy(metrics_results),
               'exact': deepcopy(metrics_results)
               }

    # results aggregated by entity type
    evaluation_agg_entities_type = {e: deepcopy(
        results) for e in target_tags_no_schema}

    for true_ents, pred_ents in zip(gold_ner_labels, pred_ner_labels):

        # compute results for one message
        tmp_results, tmp_agg_results = compute_metrics(
            collect_named_entities(
                true_ents), collect_named_entities(pred_ents), labels
        )

        # print(tmp_results)

        # aggregate overall results
        for eval_schema in results.keys():
            for metric in metrics_results.keys():
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

        # Calculate global precision and recall

        results = compute_precision_recall_wrapper(results)

        # aggregate results by entity type

        for e_type in target_tags_no_schema:

            for eval_schema in tmp_agg_results[e_type]:

                for metric in tmp_agg_results[e_type][eval_schema]:

                    evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

            # Calculate precision recall at the individual entity level

            evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                evaluation_agg_entities_type[e_type])

    return results, evaluation_agg_entities_type


def _print_metric_result_dict(metric_result_dict: Dict[str, dict]):
    for key, value in metric_result_dict.items():
        temp_to_print_dict = {}
        value['f1'] = 2*value['precision'] * value['recall'] / \
            (value['precision']+value['recall'])

        for to_print in ['precision', 'recall', 'f1']:
            temp_to_print_dict[to_print] = [value[to_print]]

        print("statistics of:", key)
        print(pd.DataFrame(temp_to_print_dict))


def namedEntityEvaluation(pred_ner_labels: List[List[str]], gold_ner_labels: List[List[str]]):
    classes = list(NER_LabelEncode.keys())
    classes.remove(NOT_NER_TAG)
    print("Performance per label type per token")
    print(per_token_eval(pred_ner_labels, gold_ner_labels, classes))

    print("Performance over full named-entity")
    results, evaluation_agg_entities_type = result_over_all_sentences(
        gold_ner_labels, pred_ner_labels, classes)

    # print(results)
    # print(evaluation_agg_entities_type)

    print("  Over all result")
    _print_metric_result_dict(results)
    print("  Per type result")
    for entity_type, result_dict in evaluation_agg_entities_type.items():
        print("Type:", entity_type)
        _print_metric_result_dict(result_dict)

    # flat_pred_ner_labels = [
    #     label for sentence in pred_ner_labels for label in sentence]
    # flat_gold_ner_labels = [
    #     label for sentence in gold_ner_labels for label in sentence]
    # gold_entities = collect_named_entities(flat_gold_ner_labels)
    # pred_entities = collect_named_entities(flat_pred_ner_labels)
    # print(compute_metrics(gold_entities, pred_entities, classes))


if __name__ == "__main__":
    print("test cws evaluation function")
    test_list = load_utf16le_data_to_list(RAW_DATA.CWS)
    wordSegmentEvaluaiton(test_list, test_list)

    print("test ner evaluation function")
    # test the first 100 sentences
    test_labels = get_ner_labels_from_file(
        RAW_DATA.NER, use_utf16_encoding=True)[:100]
    namedEntityEvaluation(test_labels, test_labels)
