import json
import pickle
import os
import re
from collections import OrderedDict
import random
import sys
from operator import itemgetter
import copy

import numpy as np
from tqdm import tqdm

HOLLE_RANDOM_SEED = 12345
_PUNCS_RE = re.compile(r'[^\w\s]')

_PLOT = 0
_REVIEW = 1
_COMMENTS = 2
_FACT_TABLE = 3
LABEL_ID2STR = {
    _PLOT: 'plot',
    _REVIEW: 'review',
    _COMMENTS: 'comments',
    _FACT_TABLE: 'fact_table'
}
_MAX_NUM_MULTI = 14


def _remove_duplicate(a_list):
    return list(OrderedDict.fromkeys(a_list))


def _f1_score(true_set, pred_set, eps=sys.float_info.epsilon):
    precision = len(true_set.intersection(pred_set)) / (float(len(pred_set)) + eps)
    recall = len(true_set.intersection(pred_set)) / (float(len(true_set)) + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score


def _check_continuity(bool_list):
    """Check if all matches are adjoint"""
    matched_indices = [idx for idx, is_match in enumerate(bool_list) if is_match]
    return all(a + 1 == b for a, b in zip(matched_indices[:-1], matched_indices[1:])), matched_indices


class HolleDatasetReader:
    def __init__(self, raw_data, datatype='test', multi_responses=None):
        self._sent_tok = self._set_sent_tok()
        # Load raw dataset
        episodes = raw_data
        mode = datatype
        self.raw_episodes = episodes
        self.mode = mode
        self.multi_responses = multi_responses
        self.episodes = None
        self.title_cache = {}
        self.sent_cache = {}

    def build(self):
        if self.mode != 'test':
            self.episodes = self._to_wow_format(self.raw_episodes, self.mode)
        else:
            self.episodes = self._to_wow_format_multi(self.raw_episodes, self.multi_responses, self.mode)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        return self.episodes[item]

    def _set_sent_tok(self):
        import spacy
        sent_tok = spacy.load('en_core_web_sm')
        return sent_tok

    def _example(self, raw_episode, episode_idx, example_idx, mode):
        chosen_topic = raw_episode['movie_name']
        response = raw_episode['chat'][example_idx + 1]
        checked_title, checked_sentence, raw_knowledge_candidates = self._get_knowledge_sentences(
            raw_episode,
            episode_idx,
            example_idx,
            mode
        )

        knowledge_candidates = {}
        for key, value in raw_knowledge_candidates.items():
            if len(value) > 32:
                print(key, len(value))
                block_num = len(value) // 32 + 1
                block_size = len(value) // block_num + 1
                for block_id in range(block_num):
                    new_key = f'{key} {block_id}'
                    knowledge_candidates[new_key] = value[block_id * block_size:(block_id + 1) * block_size]
                    if checked_sentence in knowledge_candidates[new_key]:
                        checked_title = new_key
            else:
                knowledge_candidates[key] = value

        title = 'no_passages_used' if checked_sentence == 'no_passages_used' else checked_title

        dialog_context = []
        d = raw_episode['chat']
        for bef in range(example_idx + 1):
            if f'{episode_idx}_{bef}' in self.title_cache:
                dialog_context.append({'speaker': bef % 2,
                                       'text': d[bef],
                                       'title': self.title_cache[f'{episode_idx}_{bef}'],
                                       'checked_sentence': self.sent_cache[f'{episode_idx}_{bef}']})
            else:
                dialog_context.append({'speaker': bef % 2, 'text': d[bef]})

        example = {'text': raw_episode['chat'][example_idx], 'chosen_topic': chosen_topic, 'title': title,
                   'episode_num': episode_idx, 'example_num': example_idx // 2, 'checked_sentence': checked_sentence,
                   'knowledge': knowledge_candidates, 'context': dialog_context, 'labels': [response]}

        self.title_cache[f'{episode_idx}_{example_idx + 1}'] = title
        self.sent_cache[f'{episode_idx}_{example_idx + 1}'] = checked_sentence

        return example

    def example(self, episode_idx, example_idx=0):
        raw_episode = self.raw_episodes[episode_idx]
        if example_idx + 1 >= len(raw_episode['chat']):
            return None
        return self._example(raw_episode, episode_idx, example_idx, self.mode)

    def _to_wow_format(self, raw_episodes, mode):
        print("Convert holle dataset to wow format")
        episodes = []
        for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
            if episode_idx == 5958 and 'train' in mode:
                continue
            episode = []
            for example_idx in range(0, len(raw_episode['chat']), 2):
                if example_idx + 1 < len(raw_episode['chat']):
                    example = self._example(raw_episode, episode_idx, example_idx, mode)
                    episode.append(example)
                if example_idx + 2 < len(raw_episode['chat']):
                    missing_example = self._example(raw_episode, episode_idx, example_idx + 1, mode)
                    if mode == 'train':  # add missing turns
                        episode.append(missing_example)
            episodes.append(episode)
        return episodes

    def _to_wow_format_multi(self, raw_episodes, multi_responses, mode):
        print("Convert holle test dataset to wow format")
        episodes = []
        for episode_idx, raw_episode in enumerate(tqdm(raw_episodes, ncols=70)):
            episode = []
            multi_cnt = 0
            for example_idx in range(0, len(raw_episode['chat']), 2):
                if example_idx + 1 < len(raw_episode['chat']):
                    example = self._example(raw_episode, episode_idx, example_idx, mode)
                    response = example['labels'][0]
                    checked_sentence = example['checked_sentence']
                    knowledge = example['knowledge']
                    knowledge_sentences = [x for k, v in knowledge.items() for x in v]
                    # add multiple responses
                    example['multi_eval_labels'] = [response]
                    example['multi_checked_sentences'] = [checked_sentence]
                    if multi_cnt < len(raw_episode['chat']) // 2:
                        if f'ts_{episode_idx}_{multi_cnt}' in multi_responses.keys():
                            multi_response_id = f'ts_{episode_idx}_{multi_cnt}'
                            for multi_idx in range(len(multi_responses[multi_response_id]['responses'])):
                                raw_multi_response = multi_responses[multi_response_id]['responses'][multi_idx]
                                raw_multi_span = multi_responses[multi_response_id]['spans'][multi_idx]
                                if raw_multi_response != response:
                                    multi_response = _PUNCS_RE.sub('', str(raw_multi_response))
                                    multi_span = _PUNCS_RE.sub('', str(raw_multi_span))
                                    multi_knowledge_sentences = [_PUNCS_RE.sub('', str(x)) for x in knowledge_sentences]
                                    multi_knowledge_idx = self._get_best_match_idx(multi_span,
                                                                                   multi_knowledge_sentences,
                                                                                   multi_response)
                                    example['multi_eval_labels'].append(raw_multi_response)
                                    example['multi_checked_sentences'].append(knowledge_sentences[multi_knowledge_idx])
                            multi_cnt += 1
                    episode.append(example)
                if example_idx + 2 < len(raw_episode['chat']):
                    missing_example = self._example(raw_episode, episode_idx, example_idx + 1, mode)
            episodes.append(episode)
        return episodes

    def _get_knowledge_sentences(self, raw_episode, episode_idx, example_idx, mode):
        # Handle special case
        if episode_idx == 5958 and mode == 'train':
            if example_idx in [0, 1, 2, 3]:
                return 'no_passages_used', 'no_passages_used', {'plot': 'Transformers: Aget of Extinction'}
            else:
                return 'plot', 'Transformers: Aget of Extinction', {'plot': 'Transformers: Aget of Extinction'}

        # Make GT and candidates
        knowledge_candidates = self._get_knowledge_candidates(raw_episode, example_idx)
        gt_title, gt_knowledge, knowledge_candidates = self._get_gt_knowledge(
            raw_episode, knowledge_candidates, example_idx
        )
        for key, value in knowledge_candidates.items():
            knowledge_candidates[key] = _remove_duplicate(value)

        return gt_title, gt_knowledge, knowledge_candidates

    def _get_knowledge_candidates(self, raw_episode, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        doc = raw_episode['documents']

        plot = self.validate_spacy_sentences(self._sent_tok(doc['plot']))
        review = self.validate_spacy_sentences(self._sent_tok(doc['review']))
        comments = doc['comments']
        fact_table = self._extract_fact_table(doc['fact_table'])
        knowledge_candidates = {
            'plot': plot,
            'review': review,
            'comments': comments,
            'fact_table': fact_table
        }

        return knowledge_candidates

    def _get_gt_knowledge(self, raw_episode, knowledge_candidates, example_idx):
        label = raw_episode['labels'][example_idx + 1]
        label_str = LABEL_ID2STR.get(label, 'none')
        raw_gt_span = raw_episode['spans'][example_idx + 1]
        gt_span = _PUNCS_RE.sub('', raw_gt_span)
        raw_response = raw_episode['chat'][example_idx + 1]
        response = _PUNCS_RE.sub('', raw_response)

        # Find GT knowledge sentence
        if label_str == 'none':
            gt_knowledge = 'no_passages_used'
            gt_knowledge_idx = -1
        else:
            raw_label_candidates = knowledge_candidates[label_str]
            if label_str not in ['plot', 'review']:
                raw_label_candidates = _remove_duplicate(raw_label_candidates)
            label_candidates = [_PUNCS_RE.sub('', x) for x in raw_label_candidates]
            is_gt_in_cand = [gt_span in x for x in label_candidates]
            is_cand_in_gt = [x in gt_span for x in label_candidates]

            num_gt_in_cand = sum(is_gt_in_cand)
            num_cand_in_gt = sum(is_cand_in_gt)

            # Find matched candidate index
            if num_gt_in_cand == 1:  # Exact match
                gt_knowledge_idx = is_gt_in_cand.index(True)
            elif num_gt_in_cand > 1 or label in [_COMMENTS, _FACT_TABLE] or num_cand_in_gt == 0:
                # Find best match
                gt_knowledge_idx = self._get_best_match_idx(gt_span, label_candidates, response)
            elif num_cand_in_gt == 1:  # Inverse exact match
                gt_knowledge_idx = is_cand_in_gt.index(True)
            else:  # Span can exist over multiple sentences
                is_continue, matched_indices = _check_continuity(is_cand_in_gt)
                matched_words = ' '.join([label_candidates[idx] for idx in matched_indices])

                if is_continue and len(gt_span) > len(matched_words):
                    add_front = gt_span.split()[-1] == matched_words.split()[-1]
                    add_rear = gt_span.split()[0] == matched_words.split()[0]
                    index_to_add_front = [] if matched_indices[0] == 0 else [matched_indices[0] - 1]
                    if matched_indices[-1] + 1 == len(label_candidates):
                        index_to_add_rear = []
                    else:
                        index_to_add_rear = [matched_indices[-1] + 1]

                    if add_front:
                        matched_indices = index_to_add_front + matched_indices
                    elif add_rear:
                        matched_indices = matched_indices + index_to_add_rear
                    else:  # Add front & rear
                        matched_indices = index_to_add_front + matched_indices + \
                                          index_to_add_rear
                    gt_knowledge_idx = matched_indices
                elif is_continue:
                    gt_knowledge_idx = matched_indices
                else:
                    gt_knowledge_idx = self._get_best_match_idx(
                        gt_span, label_candidates, response)

            # Get GT knowledge
            if isinstance(gt_knowledge_idx, int):
                gt_knowledge = raw_label_candidates[gt_knowledge_idx]
                gt_knowledge_idx = [gt_knowledge_idx]
            elif isinstance(gt_knowledge_idx, list):
                gt_knowledge = ' '.join(itemgetter(*gt_knowledge_idx)(raw_label_candidates))
            else:
                raise ValueError()

            new_label_candidates = []
            catch = False
            for raw_idx in range(len(raw_label_candidates)):
                if raw_idx in gt_knowledge_idx:
                    if not catch:
                        new_label_candidates.append(gt_knowledge)
                        catch = True
                else:
                    new_label_candidates.append(raw_label_candidates[raw_idx])

            knowledge_candidates[label_str] = new_label_candidates

        return label_str, gt_knowledge, knowledge_candidates

    def _extract_fact_table(self, fact_table):
        if len(fact_table.keys()) == 2:
            return []

        awards = self.validate_sentences(fact_table['awards'])
        taglines = self.validate_sentences(fact_table['taglines'])
        similar_movies = self.validate_sentences(fact_table['similar_movies'])
        box_office = fact_table['box_office']
        if isinstance(box_office, str):
            box_office = [box_office if len(box_office) > 0 else []]
        else:
            box_office = []

        return awards + taglines + similar_movies + box_office

    def _get_best_match_idx(self, gt_span, label_candidates, response):
        gt_span_words = set(gt_span.split())
        response_words = set(response.split())
        label_words_candidates = [
            set(x.split()) for x in label_candidates
        ]

        f1_scores = []
        for label_words_candidate in label_words_candidates:
            f1_scores.append(_f1_score(gt_span_words, label_words_candidate))

        if sum(f1_scores) == 0.0:
            f1_scores = []
            for label_words_candidate in label_words_candidates:
                f1_scores.append(_f1_score(response_words, label_words_candidate))

        if len(f1_scores) == 0:
            f1_scores = [0]

        max_idx = f1_scores.index(max(f1_scores))

        return max_idx

    def validate_spacy_sentences(self, spacy_sentences):
        def _validate_sent(sent):
            if len(_PUNCS_RE.sub('', sent.text).strip()) > 1:
                return True
            else:
                False

        return [sent.text for sent in spacy_sentences.sents if _validate_sent(sent)]

    def validate_sentences(self, sentences):
        return [sent for sent in sentences if len(sent) > 0]


def main():
    raw_data = json.load(open('dataset/holle/test_data.json'))
    # multi_ref = json.load(open('dataset/holle/multi_reference_test.json'))
    agent = HolleDatasetReader(raw_data, datatype='valid')

    agent.build()
    data = []
    for session in agent.episodes:

        for turn in session:
            data.append(turn)
    print(len(data))
    # json.dump(data, open('dataset/holle/test.json', 'w'))

    # for i in range(100):
    #     for j in range(0, 100):
    #         example = agent.example(i, j)
    #         if example is None:
    #             break
    #         print(json.dumps(example, indent=4))
    #         input('>')


if __name__ == '__main__':
    main()
