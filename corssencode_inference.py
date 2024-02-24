from typing import List, Tuple, Any

import numpy as np
from transformers import AutoTokenizer
from bi_encoder import get_corpus, get_question

import torch

from model import CrossEncoderBert


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CrossEncoderBert()
model.model.resize_token_embeddings(len(model.tokenizer))
model.load_state_dict(torch.load('model/torch_model', map_location=torch.device(device)))
model.tokenizer = AutoTokenizer.from_pretrained('model/tokenizer')
model.to(device)


def get_range_answers(
                      context: str,
                      question: str,
                      num_answers: int = 5) -> list[str]:

    corpus = get_corpus(context, question)
    context_question = f'{context} [Cont_token] {question}'
    context_questions = [context_question] * len(corpus)
    tokenized_texts = model.tokenizer(
        context_questions,
        corpus,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        ce_scores = model(tokenized_texts['input_ids'],
                          tokenized_texts['attention_mask']).squeeze(-1)
        ce_scores = torch.sigmoid(ce_scores)

    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]
    best_answers = []
    for idx in scores_ix[:num_answers]:
        best_answers.append((scores[idx], corpus[idx]))

    best_answers = [str(x[1]) for x in best_answers]
    return best_answers


def get_best_answer(
        context: str,
        question: str
) -> str:
    return get_range_answers(context, question, 1)[0][1]
