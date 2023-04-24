import random
import sys

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, Adafactor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from accelerate import Accelerator
from utils.evaluation import eval_f1, eval_all
from utils.evaluation import f1_score
from utils.io import write_file, read_pkl
import torch.nn.functional as F
import nltk
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import copy
import torch
import math
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class GPTData(Dataset):
    def __init__(self, data, tokenizer, context_len=256, sent_len=64, max_length=1024, test=False, psg_filter=None,
                 psg_num=1, use_oracle=False, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False,
                 add_label=True, add_response=False, add_label_to_prefix=None, add_hyperlink=False,
                 use_pred_label=None, dialogue_first=True, knowledge_response=False, second_id=False, drop_null=True,
                 max_num_of_know=None):
        super(Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.sent_len = sent_len
        self.max_length = max_length
        self.test = test
        self.psg_filter = psg_filter
        self.psg_num = psg_num
        self.use_oracle = use_oracle
        self.shuffle_id = shuffle_id
        self.max_id = max_id
        self.add_aux_loss = add_aux_loss
        self.gpt_style = gpt_style
        self.add_response = add_response
        self.add_label = add_label
        self.response = [example['labels'][0] for example in self.data]
        self.add_label_to_prefix = add_label_to_prefix
        self.add_hyperlink = add_hyperlink
        self.use_pred_label = use_pred_label
        self.dialogue_first = dialogue_first
        self.knowledge_response = knowledge_response
        self.second_id = second_id
        self.drop_null = drop_null
        self.max_num_of_know = max_num_of_know

    def __getitem__(self, index):
        example = self.data[index]

        # =============================
        # Build knowledge
        # =============================

        knowledge = example['knowledge']
        if self.psg_filter is not None:
            positive = [k for k in knowledge if k == example['title']]
            titles = positive + [k for k in knowledge if k != example['title']]
            titles = [titles[pid] for pid in self.psg_filter[index]][:self.psg_num]
            if self.use_oracle and example['title'] != 'no_passages_used' and \
                    example['title'] in knowledge and example['title'] not in titles:
                titles = [example['title']] + titles[:-1]
            new_knowledge = OrderedDict()
            for k in titles:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge
        else:
            titles = [k for k in knowledge][:self.psg_num]
            if self.use_oracle and example['title'] != 'no_passages_used' and \
                    example['title'] in knowledge and example['title'] not in titles:
                titles = [example['title']] + titles[:-1]
            new_knowledge = OrderedDict()
            for k in titles:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge

        if self.drop_null and not self.test and example['title'] != 'no_passages_used':
            if example['title'] not in knowledge or example['checked_sentence'] not in knowledge[example['title']]:
                return self[np.random.randint(len(self))]

        id_map = [i for i in range(2, self.max_id)]
        if self.shuffle_id:
            np.random.shuffle(id_map)
        id_map = [0, 1] + id_map

        # =============================
        # Passage sequence
        # =============================

        sequence = []
        sent_id = 0
        label = f'<s{id_map[0]}>'
        sentence_to_id = {}

        sequence += self.tokenizer.encode('\nPassage information.\n',
                                          add_special_tokens=False)

        sentence = 'no_passages_used'
        sent_id += 1
        sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}<s{id_map[sent_id]}>\n',
                                          add_special_tokens=False)
        sentence_to_id[sentence] = sent_id
        if sentence == example['checked_sentence']:
            label = f'<s{id_map[sent_id]}>'

        second_best = ''
        second_best_score = 0
        for pid, (title, passage) in enumerate(knowledge.items()):
            sequence += self.tokenizer.encode(f'Passage {pid + 1}, Title: {title}\n', add_special_tokens=False)
            # np.random.shuffle(passage)
            for sentence in passage:
                if len(sequence) > self.max_length:
                    break
                sent_id += 1
                sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>{sentence}',
                                                  truncation=True, max_length=self.sent_len, add_special_tokens=False)
                sequence += self.tokenizer.encode(f'<s{id_map[sent_id]}>\n', add_special_tokens=False)
                sentence_to_id[sentence] = sent_id
                if sentence == example['checked_sentence']:
                    label = f'<s{id_map[sent_id]}>'
                elif self.second_id and \
                        f1_score(sentence + example['checked_sentence'], [example['labels'][0]]) > second_best_score:
                    second_best = f'<s{id_map[sent_id]}>'
                    second_best_score = f1_score(sentence + example['checked_sentence'], [example['labels'][0]])
                if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                    break
            if self.max_num_of_know is not None and sent_id >= self.max_num_of_know:
                break

        passage_sequence = copy.deepcopy(sequence)

        if self.second_id:
            label = label + second_best

        # =============================
        # Dialogue sequence
        # =============================

        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: ', 'user1': 'User1: ', 'user2': 'User2: '}
        context = ''
        for turn in example['context']:
            speaker = role.get(turn['speaker'], turn['speaker'])
            text = turn['text']
            kk = ''
            if self.add_hyperlink and 'title' in turn:
                kk = f"[{turn['title']}]"
                if turn['checked_sentence'] in sentence_to_id:
                    kk += f"<s{id_map[sentence_to_id[turn['checked_sentence']]]}>"
                kk += ' '
            context += f'{speaker}{kk}{text}\n'

        topic = 'Chosen topic: ' + example['chosen_topic'] + '\n'
        sequence = []
        sequence += self.tokenizer.encode('\nDialogue history.\n',
                                          add_special_tokens=False)
        sequence += self.tokenizer.encode(topic, add_special_tokens=False)
        sequence += self.tokenizer.encode(context, add_special_tokens=False)[-self.context_len:]
        sequence += self.tokenizer.encode('Predict the next knowledge sentence id and response of User1.\n',
                                          add_special_tokens=False)
        if self.add_label_to_prefix:
            if isinstance(self.add_label_to_prefix, list):
                pred_label = self.add_label_to_prefix[index]
                # pred_label = '<s5>'
                sequence += self.tokenizer.encode(f'Selected knowledge = {pred_label}\n', add_special_tokens=False)
            else:
                sequence += self.tokenizer.encode(f'Selected knowledge = {label}\n',
                                                  add_special_tokens=False)
        dialogue_sequence = copy.deepcopy(sequence)

        # =============================
        # Build input output sequence
        # =============================

        sequence = []
        passage_sequence = passage_sequence[:self.max_length - len(dialogue_sequence)]
        if self.dialogue_first:
            sequence += dialogue_sequence
            sequence += passage_sequence
        else:
            sequence += passage_sequence
            sequence += dialogue_sequence
        target = []
        if self.add_label:
            if isinstance(self.use_pred_label, list):
                target.append(self.use_pred_label[index][0])
                # target.append('<s5>')
            else:
                target.append(f'{label}')
        if self.add_response:
            if self.knowledge_response and example['checked_sentence'] != 'no_passages_used' and \
                    np.random.random() < self.knowledge_response:
                target.append(f"{example['checked_sentence']}")
            else:
                target.append(f"{example['labels'][0]}")
        target = ' '.join(target)

        if self.gpt_style:
            sequence += self.tokenizer.encode('</s>', add_special_tokens=False)
            labels = [-100] * len(sequence)
            sequence += self.tokenizer.encode(target, add_special_tokens=False)
            labels += self.tokenizer.encode(target, add_special_tokens=False)
        else:  # bart style
            sequence = sequence
            labels = self.tokenizer.encode(target, truncation=True, max_length=self.context_len,
                                           add_special_tokens=True)

        return torch.tensor(sequence), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


def main():
    accelerator = Accelerator(gradient_accumulation_steps=8)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 30
    batch_size = 2

    model_name = 'facebook/bart-base'
    ckpt_name = 'wow-bart-base'
    print(model_name, ckpt_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.add_tokens([f'<s{i}>' for i in range(128)] + ['<s>', '</s>', '<pad>', '<positive>', '<negative>'])
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token = '<pad>'

    data = json.load(open('dataset/wizard/train.json'))
    psg_filter = None
    print('Data: ', len(data))
    dataset = GPTData(data, tokenizer, psg_filter=psg_filter, context_len=128, sent_len=64, max_length=512,
                      psg_num=1, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False, use_oracle=True,
                      add_label=True, add_response=False, add_hyperlink=True, add_label_to_prefix=False,
                      dialogue_first=True, knowledge_response=0.0, second_id=False, drop_null=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_constant_schedule(optimizer,
                                      # num_warmup_steps=500, num_training_steps=epochs * len(data_loader)
                                      )
    scheduler = accelerator.prepare(scheduler)

    accelerator.print(tokenizer.decode(dataset[100][0]))
    accelerator.print('===>')
    accelerator.print(tokenizer.decode(dataset[100][1]))
    accelerator.print('++++++++++')
    accelerator.print(tokenizer.decode(dataset[10][0]))
    accelerator.print('===>')
    accelerator.print(tokenizer.decode(dataset[10][1]))

    print(model_name, ckpt_name)

    for epoch in range(epochs):
        accelerator.wait_for_everyone()
        accelerator.print(f'train epoch={epoch}')
        tk0 = tqdm(data_loader, total=len(data_loader))
        losses = []
        acc = []
        for batch_idx, batch in enumerate(tk0):
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                acc.append((output.logits.argmax(-1) == batch['labels'])[:, 1].float().mean().item())
                # acc.append((output.logits.argmax(-1)[:, :-1] == batch['labels'][:, 1:])[batch['input_ids'][:, :-1] == 50386].float().mean().item())
                losses.append(loss.item())
                tk0.set_postfix(loss=sum(losses) / len(losses), acc=sum(acc[:]) / len(acc))
                scheduler.step()

        os.makedirs(f'ckpt/{ckpt_name}', exist_ok=True)
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'ckpt/{ckpt_name}/{epoch}.pt')


def filter_data(data, psg_filter=None, turn=0):
    new_data = []
    new_psg_filter = []
    for example, psg in zip(data, psg_filter):
        if example['turn_id'] == turn:
            new_data.append(example)
            new_psg_filter.append(psg)
    return new_data, new_psg_filter


def split_id(collect):
    # return collect, [''] * len(collect)
    id_collect = []
    text_collect = []
    for line in collect:
        if '>' in line:
            id_collect.append(line[:line.index('>') + 1].strip())
            text_collect.append(line[line.index('>') + 1:].strip())
        else:
            id_collect.append('')
            text_collect.append(line)
    return id_collect, text_collect


def lower(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def test():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    batch_size = 16
    model_name = 'facebook/bart-base'
    ckpt_name = 'wow-bart-base'
    corpus_name = 'wizard'
    data_name = 'unseen'

    print(model_name, ckpt_name, corpus_name, data_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.add_tokens([f'<s{i}>' for i in range(128)] + ['<s>', '</s>', '<pad>', '<positive>', '<negative>'])
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.pad_token = '<pad>'
    add_label_to_prefix = False
    use_pred_label = None
    psg_filter = None


    data = json.load(open(f'dataset/{corpus_name}/{data_name}.json'))
    print('Data: ', len(data))

    dataset = GPTData(data, tokenizer, psg_filter=psg_filter, context_len=128, sent_len=64, max_length=128 + 32,
                      psg_num=1, shuffle_id=False, max_id=128, add_aux_loss=False, gpt_style=False, test=True,
                      use_oracle=False, add_label=True, add_response=False, add_hyperlink=True,
                      add_label_to_prefix=add_label_to_prefix, use_pred_label=use_pred_label, dialogue_first=True,
                      second_id=False, max_num_of_know=None)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, num_workers=8)

    model = model.cuda()

    trunc_from = 1

    print(tokenizer.decode(dataset[100][0]))
    print(tokenizer.decode(dataset[100][1]))
    cache = eval_acc(None, json.load(open(f'dataset/{corpus_name}/{data_name}.json')), compute_cache=True)
    # cache = None
    for epoch in range(0, 100, 1):
        if not os.path.exists(f'ckpt/{ckpt_name}/{epoch}.pt'):
            continue
        print(f'Test ckpt/{ckpt_name}/{epoch}.pt')
        model.load_state_dict(torch.load(f'ckpt/{ckpt_name}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        output_text_collect = []
        output_id_collect = []
        true_text_collect = []
        acc = []
        tok_num = 0
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}

                # output = model(**batch)
                # loss = F.cross_entropy(output.logits[:, trunc_from:, :].reshape(-1, output.logits.size(-1)),
                #                        batch['labels'][:, trunc_from:].reshape(-1), reduction='sum')
                # tok_num += (batch['labels'][:, trunc_from:] != -100).float().sum().item()
                # acc.append(loss.item())

                output = model.generate(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                    max_length=128,
                )
                output_id, output_text = split_id(tokenizer.batch_decode(output, skip_special_tokens=True))
                output_text_collect.extend(output_text)

                # output_id_collect.extend(output_id)
                # batch['labels'][batch['labels'] == -100] = 1
                # true_id, true_text = split_id(tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
                # true_text_collect.extend(true_text)
                # acc.extend([int(x == y) for x, y in zip(output_id, true_id)])

                # tk0.set_postfix(acc=sum(acc) / len(acc))
        # print(sum(acc) / len(acc))
        # print(math.exp(sum(acc) / tok_num))
        #
        true_text_collect = dataset.response
        print(eval_all(lower(output_text_collect), lower(true_text_collect)))
        print(eval_acc(output_text_collect, None, cache=cache))

        json.dump(output_id_collect, open(f'ckpt/{ckpt_name}/id.{corpus_name}.{data_name}.{epoch}.json', 'w'))
        json.dump(acc, open(f'ckpt/{ckpt_name}/acc.{corpus_name}.{data_name}.{epoch}.json', 'w'))
        write_file(output_text_collect, f'ckpt/{ckpt_name}/text.{corpus_name}.{data_name}.{epoch}.txt')
        #


def eval_acc(predictions, raw, cache=None, compute_cache=False):
    import spacy
    def lower(text):
        if isinstance(text, str):
            text = text.strip().lower()
            text = ' '.join(nltk.word_tokenize(text))
            return text.strip()
        return [lower(item) for item in text]

    nlp = spacy.load("en_core_web_sm")
    ent_f1 = []
    k_f1 = []
    sent_acc = []
    build_cache = []
    if cache is not None:
        raw = cache
    if compute_cache:
        predictions = tqdm(raw)

    for pred, example in zip(predictions, raw):
        if cache is not None:
            label_knowledge, label_response, label_ents, all_candidates = example
        else:
            if isinstance(example['title'], list):
                label_knowledge = [lower(f'{t} {s}') for t, s in zip(example['title'], example['checked_sentence'])]
            else:
                label_knowledge = [lower(example['title'] + ' ' + example['checked_sentence'])]
            label_response = lower(example['labels'][0])
            label_ents = [ent.text for ent in nlp(label_response).ents]
            all_candidates = [lower(f'{title} {sentence}') for title in example['knowledge'] for sentence in
                              example['knowledge'][title]]
        if compute_cache:
            build_cache.append([label_knowledge, label_response, label_ents, all_candidates])
        else:
            pred_response = lower(pred)
            pred_ents = [ent.text for ent in nlp(pred_response).ents]
            if len(label_ents) > 0:
                ent_f1.append(f1_score(' '.join(pred_ents), [' '.join(label_ents)]))
            if len(label_knowledge) == 0:
                k_f1.append(0)
            else:
                k_f1.append(f1_score(pred_response, label_knowledge))
            max_candidates_f1 = max([f1_score(sent, [pred_response]) for sent in all_candidates])
            sent_acc.append(int(max_candidates_f1 == k_f1[-1]))
    if compute_cache:
        return build_cache
    return {'KF1': sum(k_f1) / len(k_f1) * 100,
            'EntF1': sum(ent_f1) / len(ent_f1) * 100,
            'ACC': sum(sent_acc) / len(sent_acc) * 100}


if __name__ == '__main__':
    # Training
    main()
    # Evaluation
    test()
