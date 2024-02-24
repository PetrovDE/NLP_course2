import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")


def mean_pool(token_embeds: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool


def encode(input_texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: str = "cpu"
) -> torch.tensor:

    model.eval()
    tokenized_texts = tokenizer(input_texts, max_length=512,
                                padding='max_length', truncation=True, return_tensors="pt")
    token_embeds = model(tokenized_texts["input_ids"].to(device),
                         tokenized_texts["attention_mask"].to(device)).last_hidden_state
    pooled_embeds = mean_pool(token_embeds, tokenized_texts["attention_mask"].to(device))
    return pooled_embeds


with open('data/sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

with open('data/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)

df = pd.DataFrame.from_dict(sentences)
df['corpus'] = corpus


def get_question(context: str, question: str):
    cont_quest = f"{context} [Cont_token] {question}"
    pooled_embeds = encode(cont_quest, tokenizer, bert_model, "cpu")
    pooled_embeds = pooled_embeds.cpu().detach().numpy()
    return pooled_embeds


def cosine_sim(question, embed):
    return cosine_similarity(question, embed)[0][0]


def get_corpus(context: str, question: str):
    question_embed = get_question(context, question)
    df['cosine_similarity'] = df.apply(lambda x: cosine_sim(question_embed, x['embeds']), axis=1)
    corp = df.sort_values(by=['cosine_similarity'], ascending=False).head(10)['corpus'].tolist()
    return corp
