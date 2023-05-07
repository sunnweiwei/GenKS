# Generative Knowledge Selection for Knowledge-Grounded Dialogues

Code for [Generative Knowledge Selection for Knowledge-Grounded Dialogues](https://arxiv.org/abs/2304.04836), Weiwei Sun, Pengjie Ren, and Zhaochun Ren.

Knowledge selection for knowledge-grounded dialogue systems using generative language models. Benefit from better interaction between knowledge and between knowledge and multi-turn dialogues.

## Data pre-processing
Code for pre-process wizard-of-wikipedia, holl-e, and cmu-dog can be found at `process_wizard.py`, `process_holle.py`, and `process_cmu.py`.


## Training passage re-ranking model
Train a lightweight passage re-ranking model using contrastive objectives, and get predictions on the test set.
```bash
python psg_ranker.py
```

## Training knowledge selection and response generation model
Train GenKS model that sequentially selects knowledge snippets and generates dialogue response using generative model (BART).

```bash
python run_genks.py
```



## Cite
```
@inproceedings{Sun2023GenerativeKS,
  title={Generative Knowledge Selection for Knowledge-Grounded Dialogues},
  author={Weiwei Sun and Pengjie Ren and Zhaochun Ren}
  booktitle={Findings of EACL 2023},
  year={2023}
}
```
