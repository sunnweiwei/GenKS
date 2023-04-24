import json
from collections import defaultdict

TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
TOKEN_LABEL = '__label__'
TOKEN_END_LABEL = '__endlabel__'


def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (
                    cand_title1
                    and cand_title1 in k_dict
                    and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


class WizardDialogKnowledgeTeacher:
    def __init__(self, raw_data, datatype='test'):
        self.raw_data = raw_data
        self._init_attributes({})
        self.datatype = datatype

    def _init_attributes(self, opt):
        """
        Initialize teacher attributes.
        """
        self.add_missing_turns = opt.get('add_missing_turns', 'train')
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', True)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.chosen_topic_delimiter = opt.get('chosen_topic_delimiter', '\n')
        self.title_cache = {}
        self.sent_cache = {}

    def len_episode(self, ep):
        d = self.raw_data[ep]
        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        if wizard_first:
            if self.add_missing_turns == 'none':
                len_ep = (len(d['dialog']) - 1) // 2
            elif self.add_missing_turns == 'train' and self.datatype != 'train':
                len_ep = (len(d['dialog']) - 1) // 2
            else:
                len_ep = (len(d['dialog']) - 1) // 2 + 1
            return len_ep
        return len(d['dialog']) // 2

    def _format_example(self, episode_idx, entry_idx=0):
        d = self.raw_data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)

        wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        if idx >= len(d['dialog']):
            return None

        # first, get knowledge
        apprentice_ret_passages = wizard_ret_passages = {}

        if not wizard_first or idx != 0:
            apprentice_entry = d['dialog'][idx - 1]
            apprentice_ret_passages = apprentice_entry['retrieved_passages']
        if idx - 2 >= 0:
            wizard_prev_entry = d['dialog'][idx - 2]
            wizard_ret_passages = wizard_prev_entry['retrieved_passages']

        chosen_topic = d.get('chosen_topic', '')
        chosen_topic_passages = d['chosen_topic_passage']
        chosen_topic = d.get('chosen_topic', '')

        knowledge_dict = {chosen_topic: chosen_topic_passages}
        for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
            for passage in ret_passes:
                for k, v in passage.items():
                    if k not in knowledge_dict.keys():
                        knowledge_dict[k] = v

        # then, get text
        if idx == 0:
            # first message - only have the chosen topic
            text = chosen_topic
        elif idx == 1:
            # first response - only have the first message
            text = (
                f"{chosen_topic}{self.chosen_topic_delimiter}{apprentice_entry['text']}"
            )
        else:
            text = ''
            if self.label_type == 'chosen_sent':
                # if chosen_sent, add wizard response to dialog history
                text += '{}\n'.format(wizard_prev_entry['text'])
            text += apprentice_entry['text']

        # next, get label
        wizard_entry = d['dialog'][idx]
        if self.label_type == 'response':
            labels = [wizard_entry['text']]
        else:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            if self.knowledge_separator and title != TOKEN_NOCHOSEN:
                labels = ['{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)]
            else:
                labels = ['{} {}'.format(title, sentence)]

        # finally, get label_candidates
        label_cands = ['{} {}'.format(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)]
        knowledge_str = defaultdict(list)
        for title, passage in knowledge_dict.items():
            for p in passage:
                if self.knowledge_separator:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                else:
                    cand = '{} {}'.format(title, p)
                knowledge_str[title].append(p)
                label_cands.append(cand)
        if self.label_type == 'response':
            if 'train' in self.datatype:
                label_cands = []
            else:
                label_cands = wizard_entry.get('candidate_responses', [])

        dialog_context = []
        for bef in range(idx):
            if f'{episode_idx}_{bef}' in self.title_cache:
                dialog_context.append({'speaker': d['dialog'][bef]['speaker'],
                                       'text': d['dialog'][bef]['text'],
                                       'title': self.title_cache[f'{episode_idx}_{bef}'],
                                       'checked_sentence': self.sent_cache[f'{episode_idx}_{bef}']})
            else:
                dialog_context.append({'speaker': d['dialog'][bef]['speaker'], 'text': d['dialog'][bef]['text']})

        action = dict(
            {
                'id': 'WizardDialogKnowledgeTeacher',
                'text': text,
                'labels': labels,
                'chosen_topic': chosen_topic,
                'episode_done': episode_done,
                'label_candidates': label_cands,
                'context': dialog_context
            }
        )

        action['knowledge'] = knowledge_str
        title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
        action['title'] = title
        action['checked_sentence'] = sentence

        self.title_cache[f'{episode_idx}_{idx}'] = title
        self.sent_cache[f'{episode_idx}_{idx}'] = sentence

        return action

    def example(self, episode_idx, entry_idx=0):
        return self._format_example(episode_idx, entry_idx)


def main():
    raw_data = json.load(open('dataset/raw/test_random_split.json'))
    agent = WizardDialogKnowledgeTeacher(raw_data, datatype='train')
    total = 0
    num = 0
    num_b = 0
    turns = []
    turns_a = []
    turns_b = []
    print(len(raw_data))
    saved = []
    for i in range(len(raw_data)):
        parent = None
        for j in range(100):
            example = agent.example(i, j)
            if example is None:
                break
            if example['title'] != 'no_passages_used' and example['checked_sentence'] not in example['knowledge'][example['title']]:
                print(example)
            example['parent'] = parent
            example['dialog_id'] = i
            example['turn_id'] = j
            saved.append(example)
            parent = len(saved) - 1
            # print(json.dumps(example, indent=4))
            # input('>')
            if example['episode_done']:
                break
    print(len(saved))
    print(num)
    json.dump(saved, open('dataset/wizard/seen_full.json', 'w'))
    input('>>')

    for i in range(len(raw_data)):
        last_example = None
        for j in range(100):
            example = agent.example(i, j)
            if example is None:
                break
            print(json.dumps(example, indent=4))
            input('>')
            if example['checked_sentence'] not in example['knowledge'][example['title']] and \
                    example['checked_sentence'] != 'no_passages_used':
                print(json.dumps(example, indent=4))
                input('>')
            if j == 0:
                if example['title'] == example['chosen_topic']:
                    if example['knowledge'][example['title']].index(example['checked_sentence']) == 0:
                        num_b += 1
            if last_example is not None:
                total += 1
                turns.append(j)
                if example['title'] == last_example['title'] and example['checked_sentence'] != 'no_passages_used':
                    pos = example['knowledge'][example['title']].index(example['checked_sentence'])
                    last_pos = example['knowledge'][example['title']].index(last_example['checked_sentence'])
                elif last_example['title'] in example['knowledge']:
                    titles = [k for k in example['knowledge']]
                    pos = titles.index(example['title'])
                    last_pos = titles.index(last_example['title'])
                    # if last_pos <= pos:
                    #     num += 1
                    # else:
                    #     num_b += 1
                else:
                    turns_a.append(j)
            last_example = example
    print(total)
    print(num + num_b)
    print(num)
    print(num_b)

    print(sum(turns) / len(turns))
    print(sum(turns_a) / len(turns_a))


if __name__ == '__main__':
    main()
