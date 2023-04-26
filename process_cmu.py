import os
import json
import spacy
from tqdm import tqdm


def all_file(dirname):
    fl = []
    for root, dirs, files in os.walk(dirname):
        for item in files:
            path = os.path.join(root, item)
            fl.append(path)
    return fl


def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return [line[:-1] for line in f]


nlp = spacy.load("en_core_web_sm")


def sent_tok(doc):
    doc = nlp(doc)
    out = []
    for sent in doc.sents:
        out.append(sent.text)
    return out


wiki = {}
passage_name = {'0': 'Introduction', '1': 'Scene 1', '2': 'Scene 2', '3': 'Scene 3'}
lll = []
for file in tqdm(all_file('datasets-CMU_DoG-master/WikiData')):
    item = json.load(open(file))
    intro = item['0']
    movie = intro['movieName']
    Intro = [f"{intro['movieName']} is a {intro['year']} {intro['genre']} film directed by {intro['director']}."] + \
            [f"Cast {i}: {s}" for i, s in enumerate(intro['cast'])] + ['Ratings: ' + '; '.join(intro['rating'])] + \
            sent_tok(intro['introduction']) + \
            [f"Reviews {i}: {s}" for i, s in enumerate(intro['critical_response'])]
    new_pool = {f'Scene {i}': sent_tok(item[i]) for i in ['1', '2', '3']}
    new_pool['Introduction'] = Intro
    new_pool['movieName'] = movie
    wiki[item['wikiDocumentIdx']] = new_pool
    lll.append(sum([len(v) for k, v in new_pool.items()]))

print(sum(lll) / len(lll))

saved = []
dialog_id = 0
for file in tqdm(all_file('datasets-CMU_DoG-master/Conversations/test')):
    # item = {'id':'CMU_DoG', }
    data = json.load(open(file))
    knowledge = wiki[data['wikiDocumentIdx']]
    movie = knowledge['movieName']
    context = []

    history = []
    last_user = ''
    for turn in data['history']:
        if turn['uid'] == last_user:
            history[-1]['text'] += ' ' + turn['text']
        else:
            history.append(turn)
        last_user = turn['uid']
    context = []
    last_text = movie
    parent = None
    turn_id = 0
    for turn in history:
        user_turn = {'speaker': turn['uid'], 'text': turn['text']}
        k = knowledge[passage_name[str(turn['docIdx'])]]
        item = {
            'id': 'CMU_DoG',
            'text': last_text,
            'chosen_topic': movie,
            'knowledge': {f'{movie} {k}': v for k, v in knowledge.items() if k != 'movieName'},
            'context': context,
            'labels': [turn['text']],
            'title': f"{movie} {passage_name[str(turn['docIdx'])]}",
            'checked_sentence': None,
            'dialog_id': dialog_id,
            'turn_id': turn_id
        }
        # print(json.dumps(item, indent=4))
        # input('>')
        saved.append(item)
        context.append(user_turn)
        last_text = turn['text']
        turn_id += 1
        if turn['uid'] in data['whoSawDoc']:
            saved.append(item)
    dialog_id += 1

json.dump(saved, open('test.json', 'w'))
