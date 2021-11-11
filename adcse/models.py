import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.init_weights()


"""AdCSE类与函数"""

class BertForAdCSE(BertPreTrainedModel):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        self.moco_m = self.model_args.moco_m
        self.moco_t = self.model_args.moco_t
        self.mem_m = self.model_args.mem_m
        self.mem_t = self.model_args.mem_t
        self.mem_lr = self.model_args.mem_lr
        self.mem_wd = self.model_args.mem_wd
        self.bank_size = self.model_args.neg_num
        self.sym = self.model_args.sym
        self.model_name_or_path = self.model_args.model_name_or_path
        
        self.bert = BertModel(config, add_pooling_layer=False)  # encoder_q
        self.encoder_k = BertModel(config, add_pooling_layer=False)
        self.adversary_model = Adversary_Negatives(self.bank_size, config.hidden_size)

        for param_q, param_k in zip(self.bert.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        cl_init(self, config)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.bert.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1. - self.moco_m)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                ):
        if sent_emb:  # evaluate及test阶段
            return adcse_sentemb_forward(self, self.bert,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict,
                                   )
        else:  # train阶段
            return adcse_forward(self,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              )


class Adversary_Negatives(nn.Module):
    def __init__(self, bank_size, dim):
        super(Adversary_Negatives, self).__init__()
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self, q, init_mem=False):
        memory_bank = self.W  # (dim, bank_size)
        memory_bank = nn.functional.normalize(memory_bank, dim=0)  # (dim, bank_size)
        logit=torch.einsum('nc,ck->nk', [q, memory_bank])  # (batch_size, dim)*(dim, bank_size)->(batch_size, bank_size)
        return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())


def adcse_forward(self,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               labels=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               ):
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    input_ids_q = input_ids[:, 0, :]
    input_ids_k = input_ids[:, 1, :]
    attention_mask_q = attention_mask[:, 0, :]
    attention_mask_k = attention_mask[:, 1, :]
    if token_type_ids is not None:
        token_type_ids_q = attention_mask[:, 0, :]
        token_type_ids_k = attention_mask[:, 1, :]
    else:
        token_type_ids_q = None
        token_type_ids_k = None
    if position_ids is not None:
        position_ids_q = position_ids[:, 0, :]
        position_ids_k = position_ids[:, 1, :]
    else:
        position_ids_q = None
        position_ids_k = None
    if head_mask is not None:
        head_mask_q = head_mask[:, 0, :]
        head_mask_k = head_mask[:, 1, :]
    else:
        head_mask_q = None
        head_mask_k = None
    

    if 'bert' in self.model_name_or_path:
        cur_encoder = self.bert
    else:
        raise NotImplementedError

    if not self.sym:
        '''计算q, k'''
        # Encoding
        output, q = encode_sent(self,
            encoder=cur_encoder,
            input_ids=input_ids_q,
            attention_mask=attention_mask_q,
            token_type_ids=token_type_ids_q,
            position_ids=position_ids_q,
            head_mask=head_mask_q,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions)

        # Normalize
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():  # no gradient to keys
            # if update_key_encoder:
            self._momentum_update_key_encoder()  # update the key encoder

            _, k = encode_sent(self,
                encoder=self.encoder_k,
                input_ids=input_ids_k,
                attention_mask=attention_mask_k,
                token_type_ids=token_type_ids_k,
                position_ids=position_ids_k,
                head_mask=head_mask_k,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions)
            k = nn.functional.normalize(k, dim=1)
            
            k = k.detach()
        
        '''update network'''
        l_pos = torch.einsum('nc,ck->nk', [q, k.T])
        d_norm, d, l_neg = self.adversary_model(q)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.moco_t
        cur_batch_size = logits.shape[0]
        labels = torch.arange(0, cur_batch_size, dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss()
        total_bsize=logits.shape[1]-self.bank_size
        loss = criterion(logits, labels)
        
        '''更新Adversary_Negative的参数'''
        with torch.no_grad():
            logits_d = torch.cat([l_pos, l_neg], dim=1) / self.mem_t
            p_qd = nn.functional.softmax(logits_d, dim=1)[:,total_bsize:]
            g = torch.einsum('cn,nk->ck', [q.T, p_qd])/logits_d.shape[0] - torch.mul(torch.mean(torch.mul(p_qd,l_neg),dim=0),d_norm)
            g = -torch.div(g, torch.norm(d, dim=0)) / self.mem_t  # c*k
            self.adversary_model.v.data = self.mem_m * self.adversary_model.v.data + g + self.mem_wd * self.adversary_model.W.data
            self.adversary_model.W.data = self.adversary_model.W.data - self.mem_lr * self.adversary_model.v.data

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            )
    else:
        '''计算q, k, q_pred, k_pred'''
        q_output, q = encode_sent(self,
            encoder=cur_encoder,
            input_ids=input_ids_q,
            attention_mask=attention_mask_q,
            token_type_ids=token_type_ids_q,
            position_ids=position_ids_q,
            head_mask=head_mask_q,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions)
        
        q = nn.functional.normalize(q, dim=1)
        q_pred=q
        k_output, k_pred = encode_sent(self,
            encoder=cur_encoder,
            input_ids=input_ids_k,
            attention_mask=attention_mask_k,
            token_type_ids=token_type_ids_k,
            position_ids=position_ids_k,
            head_mask=head_mask_k,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions)
        k_pred = nn.functional.normalize(k_pred, dim=1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            _, q = encode_sent(self,
                encoder=self.encoder_k,
                input_ids=input_ids_q,
                attention_mask=attention_mask_q,
                token_type_ids=token_type_ids_q,
                position_ids=position_ids_q,
                head_mask=head_mask_q,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions)
            q = nn.functional.normalize(q, dim=1)
            q = q.detach()


            _, k = encode_sent(self,
                encoder=self.encoder_k,
                input_ids=input_ids_k,
                attention_mask=attention_mask_k,
                token_type_ids=token_type_ids_k,
                position_ids=position_ids_k,
                head_mask=head_mask_k,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions)

            k = nn.functional.normalize(k, dim=1)
            k = k.detach()
            
        '''update_sym_network'''
        l_pos1 = torch.einsum('nc,ck->nk', [q_pred, k.T])
        l_pos2=torch.einsum('nc,ck->nk', [k_pred, q.T])
        
        d_norm1, d1, l_neg1 = self.adversary_model(q_pred)
        d_norm2, d2, l_neg2 = self.adversary_model(k_pred)

        logits1 = torch.cat([l_pos1, l_neg1], dim=1)
        logits1 /= self.moco_t
        logits2 = torch.cat([l_pos2, l_neg2], dim=1)
        logits2 /= self.moco_t

        cur_batch_size = logits1.shape[0]
        labels = torch.arange(0, cur_batch_size, dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss()
        loss = 0.5*criterion(logits1, labels) + 0.5*criterion(logits2, labels)
        
        '''更新Adversary_Negative的参数'''
        with torch.no_grad():
            # update memory bank
            logits1 = torch.cat([l_pos1, l_neg1], dim=1)
            logits1 /= self.mem_t

            logits2 = torch.cat([l_pos2, l_neg2], dim=1)
            logits2 /= self.mem_t
            total_bsize = logits1.shape[1] - self.bank_size
            p_qd1 = nn.functional.softmax(logits1, dim=1)[:, total_bsize:]
            g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
                torch.mean(torch.mul(p_qd1, l_neg1), dim=0), d_norm1)
            p_qd2 = nn.functional.softmax(logits2, dim=1)[:, total_bsize:]
            g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
                torch.mean(torch.mul(p_qd2, l_neg2), dim=0), d_norm1)
            g = -0.5*torch.div(g1, torch.norm(d1, dim=0)) / self.mem_t - 0.5*torch.div(g2,
                                                                            torch.norm(d1, dim=0)) / self.mem_t  # c*k
            
            self.adversary_model.v.data = self.mem_m * self.adversary_model.v.data + g + self.mem_wd * self.adversary_model.W.data
            self.adversary_model.W.data = self.adversary_model.W.data - self.mem_lr * self.adversary_model.v.data

        logits1 = torch.softmax(logits1, dim=1)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits1,
            hidden_states=q_output.hidden_states,
            attentions=q_output.attentions,
            )


def adcse_sentemb_forward(
        cls,
        encoder,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    # 这里encoder就是bert
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


def encode_sent(self, encoder,
               input_ids=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               output_attentions=None):
    
    # Encoding queries
    # input_ids: (bs, num_sent, len)
    output = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # Pooling
    pooler = self.pooler(attention_mask, output)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if self.pooler_type == "cls":
        pooler = self.mlp(pooler)

    return  output, pooler  
