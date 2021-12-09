import string
import torch
import torch.nn as nn
import numpy as np

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D, Q_mask=None, D_mask=None, cls_Q_mask=None, cls_D_mask=None, token_overlap=None):
        Q, q_input_ids = self.query(*Q)
        D, d_input_ids = self.doc(*D)
        token_overlap = self.tensor_intersect(q_input_ids, d_input_ids)
        return self.score(Q, D, Q_mask, D_mask, cls_Q_mask, cls_D_mask, token_overlap)

    @staticmethod
    def tensor_intersect(Q, D):
        D = D[1].cpu().detach().numpy()
        Q = Q[1].cpu().detach().numpy()
        return 10*(len(np.intersect1d(Q, D)) / np.count_nonzero(D))

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2), input_ids

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D, input_ids

    def old_score(self, Q, D, Q_mask=None, D_mask=None):
        if self.similarity_metric == 'cosine':
            if Q_mask is not None:
                Q = Q[:, Q_mask, :]
            if D_mask is not None:
                D = D[:, D_mask, :]
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def score(self, Q, D, Q_mask=None, D_mask=None, cls_Q_mask=None, cls_D_mask=None, token_overlap=None):
        if self.similarity_metric == 'cosine':
            if Q_mask is not None:
                Q_tok = Q[:, Q_mask, :]
            if D_mask is not None:
                D_tok = D[:, D_mask, :]
            if cls_Q_mask is not None:
                Q_cls = Q[:, cls_Q_mask, :]
            if cls_D_mask is not None:
                D_cls = D[:, cls_D_mask, :]

            score_token = (Q_tok @ D_tok.permute(0, 2, 1)).max(2).values.sum(1)
            score_cls = (Q_cls @ D_cls.permute(0, 2, 1)).max(2).values.sum(1)

            return score_token, score_cls

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
