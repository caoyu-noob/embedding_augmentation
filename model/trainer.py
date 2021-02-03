import logging
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert import BertTokenizer

from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqTokenizer
from .loss import LabelSmoothingLoss
from .optim import Adam
from .optim import NoamOpt
from .utils import pad_sequence

SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<talker1_bos>', '<talker2_bos>', '<talker1_eos>', '<talker2_eos>',
                  '<info_bos>', '<info_eos>', '.', ',', '?', '!', ':']
MIX_IGNORE_TOKENS = ['.', ',', '!', '?', ';', ':', '-', '*', '=', ')', '(', '\'', '"', ]


class Trainer:
    def __init__(self, model, train_dataset, trainer_config, writer, logger=None, test_dataset=None, valid_dataset=None,
                 n_jobs=0, label_smoothing=0, device=torch.device('cuda'), evaluate_full_sequences=False,
                 ignore_idxs=[], full_input=False, max_length=511, max_y_length=80, new_dataset=False,
                 best_model_path='',
                 no_persona=False, mixup=False, mixup_mode='alternate', mixup_dataset=None,
                 mixup_ratio=0.15, bert_mixup=False, replace=False, pointer_gen=False):
        if logger is None:
            self.logger = logging.getLogger(__file__)
        else:
            self.logger = logger

        self.train_batch_size = trainer_config.train_batch_size
        self.test_batch_size = trainer_config.test_batch_size
        self.lr = trainer_config.lr
        self.lr_warmup = trainer_config.lr_warmup
        self.weight_decay = trainer_config.weight_decay
        self.batch_split = trainer_config.batch_split
        self.s2s_weight = 1
        self.single_input = True
        self.clip_grad = trainer_config.clip_grad
        self.n_epochs = trainer_config.n_epochs
        self.linear_schedule = trainer_config.linear_schedule
        self.patience = trainer_config.patience
        self.model_saving_interval = trainer_config.model_saving_interval
        self.device = device
        self.no_persona = no_persona
        self.evaluate_full_sequences = evaluate_full_sequences
        self.global_step = 0
        self.full_input = full_input
        self.max_length = max_length
        self.max_y_length = max_y_length
        self.new_dataset = new_dataset
        self.best_ppl = 1e5
        self.best_model_path = best_model_path
        self.mixup_mode = mixup_mode
        self.replace = replace
        self.mixup = mixup
        self.mixup_dataset = mixup_dataset
        self.mixup_ratio = mixup_ratio
        self.model_type = 'pretrain'
        self.patience_cnt = 0
        self.stop_training = False
        self.pointer_gen = pointer_gen

        self.model = model.to(device)

        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing,
                                            ignore_index=self.model.padding_idx).to(device)

        param_optimizer = list(self.model.named_parameters())
        # Here we should remove parameters which are not used during to avoid breaking apex with None grads
        self.loss_weight = None
        no_decay = ['bias', 'loss']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        base_optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)

        if not self.linear_schedule:
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, lr=self.lr,
                                     linear_schedule=False, loss_weight=self.loss_weight)
        else:
            total_steps = len(train_dataset) * self.n_epochs // self.train_batch_size
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, linear_schedule=True,
                                     lr=self.lr, total_steps=total_steps, loss_weight=self.loss_weight)

        train_sampler = RandomSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size // self.batch_split,
                                           sampler=train_sampler,
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        self.train_dataset = train_dataset  # used to sample negative examples
        if test_dataset is not None:  # only do evaluation on main process
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        if valid_dataset is not None:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.test_batch_size, shuffle=False,
                                               num_workers=n_jobs, collate_fn=self.collate_func)
        self.bert_mixup = bert_mixup
        if bert_mixup:
            self.bert_model = BertForMaskedLM.from_pretrained('./bert_model').to(device)
            self.bert_tokenizer = BertTokenizer.from_pretrained('./bert_model')

        self.vocab = train_dataset.vocab
        self.writer = writer

        if isinstance(self.model, TransformerSeq2Seq):
            self.model_type = 'seq2seq'

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step}

    def load_state_dict(self, state_dict):
        if state_dict.__contains__('model') and state_dict.__contains__('optimizer'):
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.global_step = state_dict['global_step']
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def collate_func(self, data):
        persona_info, h, y, distractors_batch = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
            contexts.append(persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            contexts.append(h)

        y_out = [torch.tensor(d, dtype=torch.long) for d in y]

        if self.no_persona:
            for c in contexts[1]:
                c[0][0] = self.vocab.bos_id
            y_out = [torch.cat(pieces, dim=0) for pieces in zip(*([contexts[1]] + [y_out]))]
            lengths = [(contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
            contexts = lengths
        else:
            y_out1 = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts))]
            lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in
                       range(len(y_out))]
            y_out = (y_out1, y_out)
            contexts = lengths

        # Pad now so we pad correctly when we have only a single input (context concatenated with y)
        if isinstance(y_out, tuple):
            y_out = (
            [y[-(self.max_length - 1):] for y in y_out[0]], [y[:(self.max_y_length - 1)] for y in y_out[1]])
        else:
            y_out = [y[-(self.max_length - 1):] for y in y_out]
        contexts = [c if c[1] <= self.max_length - 1 else (c[0] - (c[1] - self.max_length + 1), self.max_length - 1)
                    for c in contexts]
        if isinstance(y_out, tuple):
            y_out = (pad_sequence(y_out[0], batch_first=True, padding_value=self.model.padding_idx),
                     pad_sequence(y_out[1], batch_first=True, padding_value=self.model.padding_idx))
        else:
            y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)

        return contexts, y_out

    def _s2s_loss(self, targets, enc_contexts, negative_samples):
        hidden_state, padding_mask = None, None

        nexts = targets[:, 1:].contiguous() if targets.dim() == 2 else targets[:, 1:, 0].contiguous()
        outputs, _, _ = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        if self.full_input:
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i][j][1] == self.vocab.sent_dialog_id:
                        nexts[i][: j] = self.model.padding_idx
                        break

        outputs = outputs.view(-1, outputs.shape[-1]).float()
        nexts = nexts.view(-1)

        loss = self.criterion(F.log_softmax(outputs, dim=-1), nexts) if self.model.training \
            else self.lm_criterion(outputs, nexts)
        return loss, hidden_state, padding_mask

    def random_replace(self, ids, ratio, lengths, th=0.4, max_mix=5):
        def padding_ids(ori_ids, max_len):
            new_ids = [idx for idx in ori_ids]
            cur_idx = 0
            while len(new_ids) < max_len:
                new_ids.append(ori_ids[cur_idx])
                cur_idx += 1
                if cur_idx == len(ori_ids):
                    cur_idx = 0
            return new_ids

        res = []
        for i in range(ids.size(0)):
            if len(ids.size()) > 2:
                cur_ids_list = ids[i][:, 0].tolist()
            else:
                cur_ids_list = ids[i].tolist()
            decoded_tokens = [self.vocab.decode([x], skip_special_tokens=True) for x in cur_ids_list[:lengths[i]]]
            token_new_start = True
            candidate_encoded_ids = []
            cur_token_ids = (1, [])
            for j, token in enumerate(decoded_tokens):
                if isinstance(self.vocab, Seq2seqTokenizer):
                    if len(token) > 0 and token not in MIX_IGNORE_TOKENS:
                        candidate_encoded_ids.append((j, tuple([cur_ids_list[j]])))
                else:
                    if len(token) > 0:
                        if token in MIX_IGNORE_TOKENS:
                            token_new_start = True
                            continue
                        if token[0] == ' ':
                            token_new_start = True
                        if token_new_start:
                            if len(cur_token_ids[1]) > 0:
                                candidate_encoded_ids.append((cur_token_ids[0], tuple(cur_token_ids[1])))
                            cur_token_ids = (j, [])
                            cur_token_ids[1].append(cur_ids_list[j])
                            token_new_start = False
                        else:
                            cur_token_ids[1].append(cur_ids_list[j])
                    else:
                        token_new_start = True
            if len(cur_token_ids[1]) > 0:
                candidate_encoded_ids.append((cur_token_ids[0], tuple(cur_token_ids[1])))
            candidate_idxs = list(range(len(candidate_encoded_ids)))
            random.shuffle(candidate_idxs)
            mix_token_num = max(math.floor(float(len(candidate_encoded_ids)) * ratio), 1)
            if self.bert_mixup:
                cur_res = self.get_bert_replace(decoded_tokens, candidate_idxs, candidate_encoded_ids, mix_token_num)
            else:
                cur_res = self.get_fasttext_repalce(candidate_idxs, candidate_encoded_ids, mix_token_num, padding_ids)
            if self.replace:
                for candidate in cur_res:
                    ids[i][candidate[2]] = candidate[0][0]
            else:
                res.append(cur_res)
        if self.replace:
            return None, ids
        return res, ids

    def get_fasttext_repalce(self, candidate_idxs, candidate_encoded_ids, mix_token_num, padding_ids):
        cur_res = []
        for candidate_idx in candidate_idxs:
            if mix_token_num == 0:
                break
            neighbors = self.mixup_dataset.get_neighbors(candidate_encoded_ids[candidate_idx][1])
            if len(neighbors) > 0:
                start_pos = candidate_encoded_ids[candidate_idx][0]
                max_neighbor_len = max([len(n[0]) for n in neighbors])
                neighbors_ids = [n[0] if len(n[0]) == max_neighbor_len else padding_ids(n[0], max_neighbor_len)
                                 for n in neighbors]
                cur_res.append((
                    torch.tensor(neighbors_ids, dtype=torch.long),
                    torch.tensor([n[1] for n in neighbors], dtype=torch.float),
                    torch.tensor(list(range(start_pos, start_pos + len(candidate_encoded_ids[candidate_idx][1]))),
                                 dtype=torch.long)
                ))
                mix_token_num -= 1
        return cur_res

    def get_bert_replace(self, decoded_tokens, candidate_idxs, candidate_encoded_ids, mix_token_num, max_replace=5):
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(decoded_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        logits = self.bert_model(input_ids.unsqueeze(0))[0].squeeze(0)
        topk_logits, topk_token_ids = torch.topk(F.softmax(logits, dim=1), k=10)
        cur_res = []
        for candidate_idx in candidate_idxs:
            if mix_token_num == 0:
                break
            cur_logits = topk_logits[candidate_encoded_ids[candidate_idx][0]].tolist()
            cur_tokens = self.bert_tokenizer.convert_ids_to_tokens(
                topk_token_ids[candidate_encoded_ids[candidate_idx][0]].tolist())
            cur_encoded_ids = [self.vocab.encode([token]) for token in cur_tokens]
            cnt = 0
            tmp_replace_ids, tmp_prob = [], []
            for i in range(len(cur_logits)):
                if len(cur_encoded_ids[i]) != 0 and cur_encoded_ids[i][0] != candidate_encoded_ids[candidate_idx][1][0]:
                    tmp_replace_ids.append(cur_encoded_ids[i])
                    tmp_prob.append(cur_logits[i])
                    cnt += 1
                    if cnt >= max_replace:
                        break
            if len(tmp_replace_ids) > 0:
                prob_tensor = torch.tensor(tmp_prob, dtype=torch.float)
                cur_res.append((
                    torch.tensor(tmp_replace_ids, dtype=torch.long),
                    prob_tensor / torch.sum(prob_tensor),
                    torch.tensor([candidate_encoded_ids[candidate_idx][0]], dtype=torch.long)
                ))
                mix_token_num -= 1
        return cur_res

    def optimizer_step(self, lm_loss, risk_loss, hits_loss, s2s_loss, full_loss):
        if self.clip_grad is not None:
            for group in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        global_step = max(self.global_step, 0)
        self.writer.add_scalar("training/lm_loss", lm_loss, global_step=global_step)
        self.writer.add_scalar("training/risk_loss", risk_loss, global_step=global_step)
        self.writer.add_scalar("training/hits_loss", hits_loss, global_step=global_step)
        self.writer.add_scalar("training/s2s_loss", s2s_loss, global_step=global_step)
        self.writer.add_scalar("training/full_loss", full_loss, global_step=global_step)
        self.writer.add_scalar("training/lr", self.optimizer.get_lr(), global_step=global_step)

        self.global_step += 1

    def _eval_train(self, epoch, risk_func=None):  # add ppl and hits@1 evaluations
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        s2s_loss = 0
        lm_loss = 0
        risk_loss = 0
        hits_loss = 0
        mixup_train = False
        if self.mixup and self.mixup_mode == 'all':
            mixup_train = True
        for i, (contexts, targets) in enumerate(tqdm_data):
            input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
            input_ids_replace, labels_replace = None, None,
            if mixup_train:
                input_ids_replace, input_ids = self.random_replace(input_ids, self.mixup_ratio, [l[0] for l in lengths])
                labels_replace, labels = self.random_replace(labels, self.mixup_ratio, [l[1] for l in lengths])
                if not self.replace:
                    input_ids_replace = [
                        [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                        for x in input_ids_replace]
                    labels_replace = [
                        [(y[0].to(self.device), y[1].to(self.device), y[2].to(self.device)) for y in x]
                        for x in labels_replace]
            loss = self.model(input_ids, labels, input_ids_replace, labels_replace)
            if isinstance(loss, tuple):
                full_loss = (loss[0] / self.batch_split,)
                s2s_loss = (i * s2s_loss + loss[1].item()) / (i + 1)
            else:
                full_loss = (loss / self.batch_split,)
                s2s_loss = (i * s2s_loss + loss.item()) / (i + 1)
            tqdm_data.set_postfix({'s2s_loss': s2s_loss})

            # optimization
            full_loss = tuple(filter(lambda x: x.requires_grad, full_loss))
            full_loss = self.optimizer.backward(full_loss)
            if self.pointer_gen and (torch.isnan(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0]) or \
                                     torch.isinf(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0])):
                self.optimizer.zero_grad()
                self.logger.info('Abnormal gradient')
            if self.mixup and self.mixup_mode == 'random':
                if torch.randint(0, 10, (1,)).item() % 2 == 0:
                    mixup_train = True
                else:
                    mixup_train = False

            if (i + 1) % self.batch_split == 0:
                self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)
                if self.mixup and self.mixup_mode == 'alternate':
                    mixup_train = not mixup_train
        if (i + 1) % self.batch_split != 0:
            self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)

    def _get_eval_loss(self, contexts, targets, metrics, index):
        lengths, enc_contexts = None, []
        input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
        batch_s2s_loss = self.model(input_ids, labels)
        if isinstance(batch_s2s_loss, tuple):
            batch_s2s_loss = batch_s2s_loss[1]

        metrics['s2s_loss'] = (metrics['s2s_loss'] * index + batch_s2s_loss.item()) / (index + 1)
        metrics['s2s_ppl'] = (metrics['s2s_ppl'] * index + math.exp(batch_s2s_loss)) / (index + 1)
        return metrics, lengths, enc_contexts

    def _get_eval_predictions(self, contexts, targets, lengths, enc_contexts, metrics, metric_funcs,
                              external_metrics_func, index):
        string_references, string_predictions = [], []
        if self.evaluate_full_sequences:
            labels = targets[1]
            input_ids = targets[0].to(self.device)
            predictions = self.model.inference(input_ids)

            labels_lens = labels.ne(self.model.padding_idx).sum(dim=-1)
            labels = [t[: l - 1].tolist() for t, l in zip(labels, labels_lens)]

            for name, func in metric_funcs.items():
                score = func(predictions, labels)
                metrics[name] = (metrics[name] * index + score) / (index + 1)

            if external_metrics_func:
                # Store text strings for external metrics
                string_references = list(
                    self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                    labels)
                string_predictions = list(
                    self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                    predictions)
                string_predictions = [x.replace('\n', ' ') for x in string_predictions]
        return string_predictions, string_references

    def _eval_test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, is_best=False,
                   raw_entail_data=None):
        with torch.no_grad():
            self.model.eval()
            if epoch == -1:
                tqdm_data = tqdm(self.test_dataloader, desc='Test')
                self.logger.info('Starting testing on Test dataset')
            else:
                tqdm_data = tqdm(self.valid_dataloader, desc='Test')
                self.logger.info('Starting testing on Valid dataset')
            metrics = {name: 0 for name in
                       ('s2s_loss', 'lm_loss', 'hits_acc', 'lm_ppl', 's2s_ppl') + tuple(metric_funcs.keys())}
            full_predictions, full_references = [], []
            for i, (contexts, targets) in enumerate(tqdm_data):
                '''Get the loss, ppl for each batch'''
                metrics, lengths, enc_contexts = self._get_eval_loss(contexts, targets, metrics, i)
                # full sequence loss
                cur_predictions, cur_references = self._get_eval_predictions(contexts, targets, lengths, enc_contexts,
                                                                             metrics, metric_funcs,
                                                                             external_metrics_func, i)
                full_predictions.extend(cur_predictions)
                full_references.extend(cur_references)
                tqdm_data.set_postfix(dict(**metrics))
            if raw_entail_data is not None:
                external_metrics_func(full_predictions, raw_entail_data)
                return

            if external_metrics_func and self.evaluate_full_sequences:
                external_metrics = external_metrics_func(full_references, full_predictions, epoch, is_best)
                metrics.update(external_metrics)

            # logging
            global_step = max(self.global_step, 0)
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar("eval/{}".format(key), value, global_step=global_step)
            self.logger.info(metrics)

            if epoch != -1:
                if metrics['s2s_ppl'] < self.best_ppl:
                    self.logger.info('Current ppl BEATS the previous best one, previous best is %.5f', self.best_ppl)
                    self.best_ppl = metrics['s2s_ppl']
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.logger.info('Best model is saved on epoch %d', epoch)
                else:
                    self.patience_cnt += 1
                    self.logger.info('Current ppl CANNOT BEATS the previous best one, previous best is %.5f',
                                     self.best_ppl)
                    if self.patience > 0 and self.patience_cnt > self.patience:
                        self.stop_training = True
            if epoch % self.model_saving_interval == 0 and epoch >= self.model_saving_interval and \
                    self.model_type in ['seq2seq']:
                torch.save(self.model.state_dict(), self.best_model_path + '_' + str(epoch))

    def _clip_grad_norm(self, grads, max_norm, norm_type=2):
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(grad.data.abs().max() for grad in grads)
        else:
            total_norm = 0
            for grad in grads:
                grad_norm = grad.data.norm(norm_type)
                total_norm += grad_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.data.mul_(clip_coef)
        return total_norm

    def test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, raw_entail_data=None):
        if hasattr(self, 'valid_dataloader') or hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs, external_metrics_func, epoch, inference, raw_entail_data=raw_entail_data)
            if epoch == -1 and not inference:
                self.logger.info('Loading the best model...')
                state_dict = torch.load(self.best_model_path, map_location=self.device)
                if state_dict.__contains__('model'):
                    self.model.load_state_dict(state_dict['model'], strict=False)
                else:
                    self.model.load_state_dict(state_dict)
                self._eval_test(metric_funcs, external_metrics_func, epoch, inference, is_best=True)

    def train(self, after_epoch_funcs=[], risk_func=None):
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('===============================')
            self.logger.info('Start training on Epoch %d', epoch)
            self._eval_train(epoch, risk_func)
            # self._eval_test()

            for func in after_epoch_funcs:
                func(epoch)
            self.logger.info('End training on Epoch %d', epoch)
            self.logger.info('===============================')
            if self.stop_training:
                self.logger.info('Training will be STOPPED in advance due to exceeding patience number')
                break

        for func in after_epoch_funcs:
            func(-1)
