import random
import sys

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from utils.evaluation import eval_f1, eval_all
from utils.evaluation import f1_score
from utils.io import write_file, read_pkl
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
import nltk
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import copy
import torch
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def add_speaker(session):
    speaker = ['User1: ', 'User2: ']
    topic = 'Topic: ' + session[0]
    response = session[-1]
    session = session[1:-1]
    for i in range(1, len(session) + 1):
        session[-i] = speaker[i % 2] + session[-i]
    return [topic] + session + [response]


def load_data(file):
    dialog = [line[:-1].split('\t') for line in open(f'{file}/en.txt')]
    dialog = [add_speaker(line) for line in dialog]
    corpus = [line[:-1] for line in open(f'{file}/0.txt')]
    pool = read_pkl(f'{file}/pool.pkl')
    return dialog, corpus, pool


class RankData(Dataset):
    def __init__(self, dialog, corpus, pool, tokenizer, context_len=256, response_len=128, neg_num=4,
                 pad_none=True):
        super(Dataset, self).__init__()
        self.dialog = dialog
        self.corpus = corpus
        self.pool = pool
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.neg_num = neg_num
        self.pad_none = pad_none

    def __getitem__(self, index):
        context = self.dialog[index][:-1]
        response = self.dialog[index][-1]
        knowledge = [self.corpus[kid] for kid in self.pool[index]]

        topic = self.tokenizer.encode(context[0])
        his = topic + self.tokenizer.encode(' '.join(context[1:]))[-(self.context_len - len(topic)):]

        neg = knowledge[1:]
        if self.pad_none:
            random.shuffle(neg)
            neg = neg + ['None.'] * (self.neg_num - len(neg))
        neg = neg[:self.neg_num]
        knowledge = [knowledge[0]] + neg
        response = self.tokenizer.encode(response, truncation=True, max_length=self.response_len)
        batch_context = []
        for k in knowledge:
            context = torch.tensor(his + self.tokenizer.encode(
                'Knowledge: ' + k, truncation=True, max_length=self.response_len))
            batch_context.append(context)
        return batch_context, response

    def __len__(self):
        return len(self.dialog)

    def collate_fn(self, data):
        padding_value = self.tokenizer.pad_token_id
        batch_context, response = zip(*data)
        batch_context = sum(batch_context, [])
        context = pad_sequence(batch_context, batch_first=True, padding_value=padding_value)
        return {
            'input_ids': context,
            'attention_mask': context.ne(padding_value),
        }


class PassageData(Dataset):
    def __init__(self, data, tokenizer, context_len=256, response_len=128, neg_num=4,
                 pad_none=True, inputs_len=512, shuffle=True, inference=False, add_label=True):
        super(Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.neg_num = neg_num
        self.pad_none = pad_none
        self.inputs_len = inputs_len
        self.shuffle = shuffle
        self.inference = inference
        self.add_label = add_label

    def __getitem__(self, index):
        example = self.data[index]
        if not self.inference and example['title'] not in example['knowledge']:
            return self[np.random.randint(len(self))]
        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: '}
        # context = '\n'.join([role[turn['speaker']] + turn['text'] for turn in example['context']])
        context = ''
        for turn in example['context']:
            speaker = role[turn['speaker']]
            text = turn['text']
            kk = ''
            if 'title' in turn and self.add_label:
                kk = f" [{turn['title']}]"
            context += f'{speaker}{text}{kk}\n'
        topic = 'Topic: ' + example['chosen_topic']

        positive = ['Title: ' + k + ' '.join(v) for k, v in example['knowledge'].items() if k == example['title']]
        negative = ['Title: ' + k + ' '.join(v) for k, v in example['knowledge'].items() if k != example['title']]

        topic = self.tokenizer.encode(topic)
        his = topic + self.tokenizer.encode(context)[-(self.context_len - len(topic)):]

        if self.shuffle:
            random.shuffle(negative)
        if self.pad_none:
            negative = negative + ['None.'] * (self.neg_num - len(negative))
        neg = negative[:self.neg_num]
        knowledge = positive + neg
        batch_context = []
        for k in knowledge:
            context = torch.tensor(his + self.tokenizer.encode(
                'Knowledge: ' + k, truncation=True, max_length=self.inputs_len - len(his)))
            batch_context.append(context)
        return batch_context, None

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data):
        padding_value = self.tokenizer.pad_token_id
        batch_context, response = zip(*data)
        batch_context = sum(batch_context, [])
        context = pad_sequence(batch_context, batch_first=True, padding_value=padding_value)
        return {
            'input_ids': context,
            'attention_mask': context.ne(padding_value),
        }


class SentenceData(PassageData):
    def __init__(self, *args, psg_filter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psg_filter = psg_filter

    def __getitem__(self, index):
        example = self.data[index]
        role = {'0_Wizard': 'User1: ', '1_Apprentice': 'User2: ', '0_Apprentice': 'User2: ', '1_Wizard': 'User1: ',
                0: 'User1: ', 1: 'User2: '}

        context = ''
        for turn in example['context']:
            speaker = role[turn['speaker']]
            text = turn['text']
            kk = ''
            if 'title' in turn and self.add_label:
                kk = f"[{turn['title']}] "
            context += f'{speaker}{kk}{text}\n'
        topic = 'Topic: ' + example['chosen_topic']

        knowledge = example['knowledge']
        if self.psg_filter is not None:
            positive = [k for k in knowledge if k == example['title']]
            titles = positive + [k for k in knowledge if k != example['title']]
            titles = [titles[pid] for pid in self.psg_filter[index]]
            new_knowledge = OrderedDict()
            for k in titles[:1]:
                new_knowledge[k] = knowledge[k]
            knowledge = new_knowledge

        if self.psg_filter is not None and (
                example['title'] not in knowledge or example['checked_sentence'] not in knowledge[example['title']]):
            positive = ['None.']
        else:
            if example['title'] not in knowledge or example['checked_sentence'] not in knowledge[example['title']]:
                if self.inference:
                    positive = ['None.']
                else:
                    positive = ['Title: no_passages_used | no_passages_used']
            else:
                positive = [
                    f"Title: {example['title']} | {knowledge[example['title']].index(example['checked_sentence'])} "
                    f"| {example['checked_sentence']}"]
        negative = [f'Title: {k} | {j} | {sent}' for k, v in knowledge.items() for j, sent in enumerate(v) if
                    sent != example['checked_sentence']]

        if example['checked_sentence'] != 'no_passages_used':
            negative += ['Title: no_passages_used | no_passages_used']

        topic = self.tokenizer.encode(topic)
        his = topic + self.tokenizer.encode(context)[-(self.context_len - len(topic)):]

        if self.shuffle:
            random.shuffle(negative)
        if self.pad_none:
            negative = negative + ['None.'] * (self.neg_num - len(negative))
        neg = negative[:self.neg_num]
        knowledge = positive + neg
        batch_context = []
        for k in knowledge:
            context = torch.tensor(his + self.tokenizer.encode(
                'Knowledge: ' + k, truncation=True, max_length=self.inputs_len - len(his)))
            batch_context.append(context)
        return batch_context, None


def main():
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 10
    batch_size = 2
    neg_num = 2
    model_name = 'distilbert-base-uncased'

    ckpt_name = 'wow-distilbert-psg-rank'
    print(model_name, ckpt_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = json.load(open('dataset/wizard/train.json'))
    print('Data: ', len(data))
    dataset = PassageData(data, tokenizer, context_len=128, inputs_len=512, neg_num=neg_num,
                           pad_none=True, shuffle=True, inference=False, add_label=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=8)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=epochs * len(data_loader))
    scheduler = accelerator.prepare(scheduler)

    for epoch in range(epochs):
        accelerator.wait_for_everyone()
        accelerator.print(f'train epoch={epoch}')
        tk0 = tqdm(data_loader, total=len(data_loader))
        losses = []
        for batch in tk0:
            output = model(**batch)
            logits = output.logits.view(-1, neg_num + 1)
            loss = F.cross_entropy(logits, torch.zeros((logits.size(0),)).long().cuda())
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            tk0.set_postfix(loss=sum(losses) / len(losses))

        os.makedirs(f'ckpt/{ckpt_name}', exist_ok=True)
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'ckpt/{ckpt_name}/{epoch}.pt')


def lower(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ' '.join(nltk.word_tokenize(text))
        return text.strip()
    return [lower(item) for item in text]


def filter_dialog(dialog, pool, turn=0):
    new_dialog = []
    new_pool = []
    for i in range(len(dialog)):
        if ' '.join(dialog[i]).count('User1') == turn:
            new_dialog.append(dialog[i])
            new_pool.append(pool[i])
    return new_dialog, new_pool


def filter_data(data, turn=0):
    new_data = []
    for example in data:
        if example['turn_id'] == turn:
            new_data.append(example)
    return new_data


def test():
    batch_size = 1
  
    model_name = 'distilbert-base-uncased'
    ckpt_name = 'wow-distilbert-psg-rank'
    corpus_name = 'wizard'
    data_name = 'seen'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = json.load(open(f'dataset/{corpus_name}/{data_name}.json'))

    psg_filter = None

    print(model_name, ckpt_name, corpus_name, data_name, len(data))
    dataset = PassageData(data, tokenizer, context_len=128, inputs_len=512, neg_num=999,
                          pad_none=False, shuffle=False, inference=True, add_label=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collate_fn, batch_size=batch_size, shuffle=False, num_workers=8)

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model = model.cuda()

    for epoch in range(0, 100, 1):
        if not os.path.exists(f'ckpt/{ckpt_name}/{epoch}.pt'):
            continue
        print(f'Test ckpt/{ckpt_name}/{epoch}.pt')
        model.load_state_dict(torch.load(f'ckpt/{ckpt_name}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        output_text_collect = []
        scores = defaultdict(list)
        metrics = [1, 2, 3, 5, 10, 20]
        model.eval()
        ranks = []
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model(**batch)
                logits = output.logits
                logits = logits.view(-1)
                logits = -logits
                rank = logits.argsort(-1).cpu().tolist()
                ranks.append(rank)
                for k in metrics:
                    scores[f'{k}'].append(int(0 in rank[:k]))
                tk0.set_postfix(**{k: sum(v) / len(v) * 100 for k, v in scores.items()})
        print({k: sum(v) / len(v) * 100 for k, v in scores.items()})
        data_name = data_name.replace('/', '.')
        json.dump(ranks, open(f'ckpt/{ckpt_name}/{corpus_name}.{data_name}.{epoch}.json', 'w'))


if __name__ == '__main__':
    # Training
    main()
    # Evaluation
    test()
