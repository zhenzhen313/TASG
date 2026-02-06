
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel ,AutoTokenizer
from model.attention import DotProductAttention,Attention
from model.GCN import GCN
import numpy as np


class CustomModel(nn.Module):
    def __init__(self, src, catr, tokenizer):
        super().__init__()
        self.scr = src
        self.catr = catr
        self.bert = self.build_model(self.src.model,tokenizer)
        self.dropout = nn.Dropout(self.src.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        self.att = DotProductAttention(self.src.dropout)
        self.mul_att = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=self.src.dropout)
        self.aspect_att = Attention(4,768,768)
        self.gcn_dropout = 0
        self.GCN_on = True
        self.context_gcn=GCN(768,768,768,dropout=self.gcn_dropout)
        self.before_output_mean = True
        self.dep_mode = 'text_anp_sim'
        self.project = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self._init_weights(self.out)
        self._init_weights(self.project)
        self.linear_aspect=nn.Linear(768*2,1)
       
    def _init_weights(self, module):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def build_model(self,
                    bert_model,
                    tokenizer
                    ):
        model = RobertaModel.from_pretrained(bert_model)
        num_tokens, embed_size = model.embeddings.word_embeddings.weight.shape 

        new_tokens = tokenizer.unique_no_split_tokens
        model.resize_token_embeddings(
            len(new_tokens) + num_tokens 
        ) 

        embedding_layer = model.embeddings.word_embeddings
        padding_idx =  self.src.pad_token_id
        embedding_layer.padding_idx = padding_idx


        _tokenizer =AutoTokenizer.from_pretrained("bertweet-base")

 
        for token in new_tokens:
            if token.startswith('<<') and token.endswith('>>'):
                core_token = token[2:-2]
                core_token_ids = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(core_token))

                if len(core_token_ids) == 0:
                    raise RuntimeError(f"Core token '{core_token}' could not be tokenized.")
                
                token_ids = tokenizer.convert_tokens_to_ids(tokenizer._base_tokenizer.tokenize(token))
                
                if len(token_ids) > 1:
                    raise RuntimeError(f"Token '{token}' splits into multiple token IDs.")
                new_token_id = token_ids[0]

                assert new_token_id >= num_tokens, (new_token_id, num_tokens, token)
                
                embed = torch.zeros(embed_size)
                for idx in core_token_ids:
                    embed += embedding_layer.weight.data[idx]
                embed /= len(core_token_ids)
                
                embedding_layer.weight.data[new_token_id] = embed
            
        return model
              
    def forward(self,samples,caption,c_mask,input_ids,attention_mask,ce_loss,input_ids_tt,attention_mask_tt,input_ids_at,attention_mask_at,
                dependency_input_ids=None, dependency_attention_mask=None, dependency_aspect_mask=None, dependency_adj_mask = None, dependency_adv_mask = None, dependency_matrix=None,text_mask=None,
                is_training=True):
        sample_k = self.src.sample_k
        cap_len = self.src.max_caption_len
        pad_token_id = self.src.pad_token_id
        end_token_id = self.src.end_token_id
        bs = caption.shape[0]
        self.device = caption.device
      
        origin_len = attention_mask.sum(dim=-1)
 
        end_token_embedding = self.bert.embeddings.word_embeddings(torch.tensor(end_token_id, device=self.device))
      
        _, caption_out, _, cap_mask, finished = self.catr(samples,caption,c_mask,cap_len,sample_k,end_token_id,pad_token_id)
        sorted_out_id = torch.argsort(caption_out, dim=-1, descending=True)  
        sample_out_id = sorted_out_id[:, :, :sample_k]
        sample_out = torch.zeros((bs, cap_len, sample_k),dtype=torch.float,device=self.device)
        for i in range(bs):
            for j in range(cap_len):
                sample_out[i][j] = caption_out[i, j, sample_out_id[i][j]]
        sample_prob = F.softmax(sample_out, dim=-1)
        sample_prob = sample_prob.unsqueeze(3)
        sample_caption_embedding = self.bert.embeddings.word_embeddings(sample_out_id)
        sample_caption_embedding = (sample_prob*sample_caption_embedding).sum(dim=2) 
 
        for i in range(bs):
            if finished[i]:
                cap_mask[i, finished[i]] = True
        inputs_embeds = self.bert.embeddings.word_embeddings(input_ids).clone()
        inputs_embeds_tt = self.bert.embeddings.word_embeddings(input_ids_tt).clone()
        inputs_embeds_at = self.bert.embeddings.word_embeddings(input_ids_at).clone()
        target_len = attention_mask_at.sum(dim=-1)

        cap_mask = (~cap_mask).int()
        caption_len = cap_mask.sum(dim=-1)
        
        inputs_embeds_tt,_ = self.att(inputs_embeds_tt, inputs_embeds_tt, inputs_embeds_tt, attention_mask_tt.sum(dim=-1))
        att_output_at, _ = self.att(inputs_embeds_at, inputs_embeds_tt, inputs_embeds_tt, attention_mask_tt.sum(dim=-1))
        inputs_embeds_at = (inputs_embeds_at + att_output_at) / 2 
        sample_caption_embedding,_ = self.att(sample_caption_embedding, sample_caption_embedding,sample_caption_embedding, caption_len)
        att_output_ai, _ = self.att(inputs_embeds_at, sample_caption_embedding, sample_caption_embedding, caption_len)
        attention_mask = attention_mask.clone()
        inputs_embeds = inputs_embeds.clone()
        
        senti_feature = self.senti_matrix(dependency_input_ids, dependency_attention_mask, dependency_aspect_mask,dependency_adj_mask, dependency_adv_mask, dependency_matrix,text_mask)

        for i in range(bs):
            attention_mask[i, origin_len[i]: origin_len[i]+target_len[i]] = 1
            inputs_embeds[i, origin_len[i]: origin_len[i]+target_len[i]] = att_output_ai[i, :target_len[i]]
            inputs_embeds[i, origin_len[i]+target_len[i]] = end_token_embedding

        outputs = self.bert(inputs_embeds=inputs_embeds,attention_mask=attention_mask)
        outputs = outputs.pooler_output
        outputs_unsqueezed = outputs.unsqueeze(1)
        x_final = torch.cat([outputs_unsqueezed, senti_feature], dim = 1)
        if self.before_output_mean:
            x_final = x_final.mean(dim=1)  # [8, 768]

        outputs = self.out(self.dropout(x_final))
        outputs_kl = self.out(self.dropout(x_final))

        return outputs,outputs_kl
    
    def get_aspect_embed(self, feature, aspect_mask):
        aspect_mask = aspect_mask.cpu()
        aspect_num = [x.numpy().tolist().count(1) for x in aspect_mask] 
        aspect_position=[np.where(np.array(x)==1)[0].tolist() for x in aspect_mask]
        for i,x in enumerate(aspect_position):
            assert len(x)==aspect_num[i]
        max_aspect_num = max(aspect_num)

        for i,x in enumerate(aspect_position):
            if len(x) < max_aspect_num:
                aspect_position[i]  +=  [0]*(max_aspect_num-len(x))
        aspect_position = torch.tensor(aspect_position).to(self.device)
        aspect_embed = torch.zeros(feature.shape[0], max_aspect_num,feature.shape[-1]).to(self.device)
        for i in range(len(feature)):
            aspect_embed[i] = torch.index_select(feature[i],dim=0, index=aspect_position[i])
            aspect_embed[i, aspect_num[i]:] = torch.zeros(max_aspect_num-aspect_num[i],feature.shape[-1])
        return aspect_embed

    def aspect_attention(self, feature, aspect_embed):
        att_feautures = self.aspect_att(feature,aspect_embed,aspect_embed)
        alpha = torch.sigmoid(self.linear_aspect(torch.cat([feature, att_feautures], dim=-1)))
        alpha = alpha.repeat(1,1,768)
        encoder_outputs = torch.mul(1-alpha, feature)+torch.mul(alpha,att_feautures)
        return encoder_outputs


    def multimodal_Graph(self, aspect_encoder_outputs, dependency_matrix, attention_mask, aspect_mask, dependency_adj_mask, dependency_adv_mask, text_mask):
       
        new_dependency_matrix = torch.zeros([aspect_encoder_outputs.shape[0], aspect_encoder_outputs.shape[1],aspect_encoder_outputs.shape[1]],dtype=torch.float).to(aspect_encoder_outputs.device)
        if self.dep_mode == 'text_anp_sim':
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            anp_mask_expanded = 1.0 - text_mask_expanded 
           
            text_feature = aspect_encoder_outputs * text_mask_expanded
            anp_feature = aspect_encoder_outputs * anp_mask_expanded

            anp_feature_extend = anp_feature.unsqueeze(1)  
            text_feature_extend = text_feature.unsqueeze(2) 
            sim = torch.cosine_similarity(text_feature_extend, anp_feature_extend, dim=-1) 

            aspect_mask_expanded = aspect_mask.unsqueeze(-1)
            sim = sim * aspect_mask_expanded  
            new_dependency_matrix = sim + sim.transpose(1, 2)

           
            text_feat1 = text_feature.unsqueeze(1) 
            text_feat2 = text_feature.unsqueeze(2) 
            text_sim = torch.cosine_similarity(text_feat1, text_feat2, dim=-1) 

            dependency_adj_mask_expanded = dependency_adj_mask.unsqueeze(-1)
            dependency_adv_mask_expanded = dependency_adv_mask.unsqueeze(-1)
            adj_sim = text_sim*dependency_adj_mask_expanded*aspect_mask_expanded
            adv_sim = text_sim*dependency_adv_mask_expanded*aspect_mask_expanded

            new_dependency_matrix = new_dependency_matrix * ((~text_mask).unsqueeze(2).float()) * ((~text_mask).unsqueeze(1).float()) 
            new_dependency_matrix = new_dependency_matrix + dependency_matrix * text_sim  
            new_dependency_matrix = new_dependency_matrix + (adj_sim + adj_sim.transpose(1, 2))*1.5
            new_dependency_matrix = new_dependency_matrix + (adv_sim + adv_sim.transpose(1, 2))*1.5

        elif self.dep_mode == 'sim_together':
            aspect_encoder_outputs1 = aspect_encoder_outputs.unsqueeze(1) 
            aspect_encoder_outputs2 = aspect_encoder_outputs.unsqueeze(2) 
            sim_together = torch.cosine_similarity(aspect_encoder_outputs1, aspect_encoder_outputs2, dim=-1)
            new_dependency_matrix = dependency_matrix * sim_together

        for i in range(new_dependency_matrix.shape[1]):
                new_dependency_matrix[:,i,i]=1

        context_dependency_matrix = new_dependency_matrix.clone().detach()
        if self.GCN_on:
            context_feature=self.context_gcn(aspect_encoder_outputs,context_dependency_matrix,attention_mask)
        
        return 0.65*context_feature + aspect_encoder_outputs



    def senti_matrix(self, dependency_input_ids,dependency_attention_mask, dependency_aspect_mask,dependency_adj_mask, dependency_adv_mask,dependency_matrix, text_mask):
        encoded = self.bert(input_ids = dependency_input_ids,attention_mask=dependency_attention_mask)
        last_hidden_state = encoded[0]
        aspect_embed = self.get_aspect_embed(last_hidden_state, dependency_aspect_mask) 
        aspect_encoder_outputs = self.aspect_attention(last_hidden_state, aspect_embed)

        if self.GCN_on:
            mix_feature = self.multimodal_Graph(aspect_encoder_outputs,dependency_matrix, dependency_attention_mask, dependency_aspect_mask,dependency_adj_mask, dependency_adv_mask,text_mask)

        return mix_feature





