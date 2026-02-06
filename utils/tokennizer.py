import spacy
from spacy.symbols import ORTH
import numpy as np
import json
import torch
import emoji
from transformers import AutoTokenizer,RobertaModel
import csv

nlp = spacy.load("en_core_web_sm")

def matrix_pad(mat, pad_cl):
    assert mat.shape[0] == mat.shape[1]
    mat_len = mat.shape[0]
    indice = list(np.arange(0, mat_len)) 
    indice.insert(pad_cl + 1, 0) 

    mat = mat[:, indice] 
    mat[:, pad_cl + 1] = 0 
    mat = mat[indice, :]
    mat[pad_cl + 1, :] = 0
    return mat

def preprocess_data(data_file_path):
    texts = []
    image_ids = []
    aspects_list = []
    aspect_positions_list = []
    texts1 = []

    with open(data_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None) 
        for row in reader:
            if len(row) < 5:
                    continue
          
            image_id = row[2]
            text_with_T = row[3]
            aspect_str = row[4]

            text_with_T = emoji.replace_emoji(text_with_T, '')
            aspect_str = emoji.replace_emoji(aspect_str, '')
            tokens = text_with_T.split()
            aspect_tokens = aspect_str.split()
        
            if '$T$' not in tokens:
                continue

            t_index = tokens.index('$T$')
            aspect_length = len(aspect_tokens)
            asp_start = t_index
            asp_end = t_index + aspect_length

            new_tokens = tokens[:t_index] + ["<ASPECT>"]+aspect_tokens + ["</ASPECT>"] + tokens[t_index+1:]
            final_text = " ".join(new_tokens)
            new_tokens1 = tokens[:t_index] + aspect_tokens + tokens[t_index+1:]
            final_text1 = " ".join(new_tokens1)

            texts.append(final_text)
            texts1.append(final_text1)
            image_ids.append(image_id)
            aspects_list.append(aspect_tokens)
            aspect_positions_list.append((asp_start, asp_end))  #

        output = {
                    'texts': texts,
                    'texts1':texts1,
                    'image_ids':image_ids,
                    'aspects_list':aspects_list,
                    'aspect_positions_list':aspect_positions_list}

        return output

class SentiTokenizer:
    def __init__(
        self,
        pretrained_model_name,
        path_ANP,
        begin_ANP = "<<ANP>>",
        end_ANP = "<</ANP>>"
    ):
        self._base_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.path_ANP = path_ANP

        self.additional_special_tokens = [begin_ANP,end_ANP]
        unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.unique_no_split_tokens = unique_no_split_tokens + self.additional_special_tokens
        self.unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.add_tokens(self.additional_special_tokens)
        self.begin_ANP = begin_ANP
        self.end_ANP = end_ANP
        self.begin_ANP_id = self.convert_tokens_to_ids(begin_ANP)
        self.end_ANP_id = self.convert_tokens_to_ids(end_ANP)

        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.eos_token_id = self._base_tokenizer.sep_token_id
        self.eos_token = self._base_tokenizer.sep_token
        self.bos_token_id = self._base_tokenizer.bos_token_id
        self.bos_token = self._base_tokenizer.bos_token

        
        

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)
    

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)
    
    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)
    
    def hete_sent_matrix(self, text, text1, image_id, aspect, aspect_positions):
       
        if not isinstance(image_id, list):
            image_id = [image_id]
        if not isinstance(text, list):
            text = [text]
        if not isinstance(text1, list):
            text1 = [text1]
        if not isinstance(aspect,list):
            aspect = [aspect]

        with open(self.path_ANP, 'r') as f:
            anp_data = json.load(f) 
   
        aspect_positions = []
        special_aspect_tokens = ["<ASPECT>","</ASPECT>"]
        for token in special_aspect_tokens:
            nlp.tokenizer.add_special_case(token, [{ORTH: token}])
        pos_doc = [nlp(x) for x in text]
        pos_doc1 = [nlp(x) for x in text1]
       
        text_split = []
        for i, split in enumerate(pos_doc):
            a_text = []
            for token in pos_doc[i]:
                a_text.append(str(token))
            text_split.append(a_text)
            
            asp_start = 0
            asp_end = 0
            for j in range(len(pos_doc[i])):
                if str(pos_doc[i][j]) =="<ASPECT>":
                    asp_start = j+1
                elif str(pos_doc[i][j]) == "</ASPECT>":
                    asp_end = j

            if asp_start is None or asp_end is None:
                raise ValueError(f"Text at index {i} is missing <ASPECT> or </ASPECT> tags.")
            
            
            del text_split[i][asp_end]

            del text_split[i][asp_start-1]
            
            aspect_token = text_split[i][asp_start-1:asp_end-1]
            asp_end = asp_start-1+len(aspect_token)

            aspect_positions.append((asp_start-1, asp_end))
            assert len(pos_doc1[i]) == len(text_split[i])

        adj_list = ['JJ', 'JJR', 'JJS']
        adv_list = ['RB', 'RBR', 'RBS']
        adj_positions = []
        adv_positions = []
        for i, split in enumerate(pos_doc1):
            adj_position = []
            adv_position = []
            for j in range(len(pos_doc1[i])):
                if pos_doc1[i][j].tag_ in adj_list:
                    adj_position.append(j)
                if pos_doc1[i][j].tag_ in adv_list:
                    adv_position.append(j)
            adj_positions.append(adj_position)
            adv_positions.append(adv_position)

        dependency_matrix = [torch.zeros([len(text_split[i]), len(text_split[i])]) for i in range(len(text_split))]
        for i, split in enumerate(text_split):
            for t in pos_doc1[i]:
                dependency_matrix[i][t.i][t.i] = 5  
                for child in t.children: 
                    dependency_matrix[i][t.i][child.i] = 1
                    dependency_matrix[i][child.i][t.i] = 1

                    for cchild in child.children: 
                        dependency_matrix[i][t.i][cchild.i] = 1
                        dependency_matrix[i][cchild.i][t.i] = 1

        assert len(text_split) == len(pos_doc1)

        aspect_masks = []
        input_sentence_tokens = []
        text_lens = []
        token_index = [np.arange(0,len(x)) for x in text_split]

        adj_masks = []
        adv_masks = []

        for i,split in enumerate(text_split):

            aspect_mask = [0]
            adj_mask = [0]
            adv_mask = [0]
            word_bpes = [self.bos_token_id]
            dependency_matrix[i] = matrix_pad(dependency_matrix[i], -1)
            dependency_matrix[i][0][0] = 1 

            asp_start, asp_end = aspect_positions[i]

            for j, word in enumerate(split):
                bpes = self._base_tokenizer.tokenize(word)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)

                if j >= asp_start and j < asp_end:
                            aspect_mask += [1] * len(bpes)
                else:
                    aspect_mask += [0] * len(bpes)
                
                if j in adj_positions[i]:
                    adj_mask += [1]*len(bpes)
                else:
                    adj_mask += [0]*len(bpes)
                
                if j in adv_positions[i]:
                    adv_mask += [1]*len(bpes)
                else:
                    adv_mask += [0]*len(bpes)

                if len(bpes) > 1:
                    for d_i in range(len(bpes) - 1): 
                        pad_index = token_index[i][j] + d_i 
                        dependency_matrix[i] = matrix_pad(dependency_matrix[i], pad_index)
                        dependency_matrix[i][pad_index + 1][pad_index + 1] = 5
                        have_arc = torch.nonzero(
                            dependency_matrix[i][token_index[i][j]] == 1
                        ).squeeze().numpy().tolist() 
                        if isinstance(have_arc,int):
                            have_arc = [have_arc]
                        for arc_x in have_arc:
                            if arc_x != token_index[i][j]:
                                dependency_matrix[i][pad_index+1][arc_x] = 1
                                dependency_matrix[i][arc_x][pad_index+1] = 1

                    for d_j in range(j + 1, len(split)):
                            token_index[i][d_j] += len(bpes)
                            token_index[i][d_j] -= 1    

                word_bpes.extend(bpes)

            word_bpes.append(self.eos_token_id)
            dependency_matrix[i] = matrix_pad(dependency_matrix[i], dependency_matrix[i].shape[0]-1)
            dependency_matrix[i][-1][-1] = 1
            aspect_mask +=[0]
            adj_mask += [0]
            adv_mask += [0]

            text_len = len(word_bpes)
            text_lens.append(text_len)
            
            cur_image_id = image_id[i]
            if str(cur_image_id) in anp_data:
                anp_dict = anp_data[str(cur_image_id)]
            else:
                anp_dict = []
            

            if isinstance(anp_dict, dict):
                sorted_anps = sorted(anp_dict.items(), key=lambda x: x[1], reverse=True)
                top_anps = sorted_anps[:5]
                top_anps_dict = {key.replace('_', ' '): value for key, value in top_anps}
            for anp_node,anp_value in top_anps_dict.items():
                anp_bpes = self._base_tokenizer.tokenize(anp_node)
                anp_bpes = self._base_tokenizer.convert_tokens_to_ids(anp_bpes)

                word_bpes.append(self.begin_ANP_id)
                aspect_mask += [0]
                adj_mask += [0]
                adv_mask += [0]
                dependency_matrix[i] = matrix_pad(dependency_matrix[i], dependency_matrix[i].shape[0]-1)
                dependency_matrix[i][-1][-1] = 1
                
                for _ in anp_bpes:
                    dependency_matrix[i] = matrix_pad(dependency_matrix[i], dependency_matrix[i].shape[0]-1) 
                    dependency_matrix[i][-1][-1] = 5  
                
                aspect_mask += [0]*len(anp_bpes)
                adj_mask += [0]*len(anp_bpes)
                adv_mask += [0]*len(anp_bpes)
                word_bpes.extend(anp_bpes)

                word_bpes.append(self.end_ANP_id)
                aspect_mask += [0]
                adj_mask += [0]
                adv_mask += [0]
                dependency_matrix[i] = matrix_pad(dependency_matrix[i], dependency_matrix[i].shape[0]-1)
                dependency_matrix[i][-1][-1] = 1

            
            input_sentence_tokens.append(word_bpes)
            aspect_masks.append(aspect_mask)
            adj_masks.append(adj_mask)
            adv_masks.append(adv_mask)

        input_sentence_tokens, input_sentence_mask, aspect_masks_padded, adj_masks_padded, adv_masks_padded, dependency_matrix_padded, text_mask = self.pad_tokens(
                    input_sentence_tokens, aspect_masks, adj_masks, adv_masks, dependency_matrix, text_lens)
        encoded = {
        'input_ids': input_sentence_tokens,
        'attention_mask': input_sentence_mask,
        'aspect_mask': aspect_masks_padded,
        'adj_mask':adj_masks_padded,
        'adv_mask':adv_masks_padded,
        'dependency_matrix': dependency_matrix_padded,
        'text_mask':text_mask
        }

        return encoded

                 
    def pad_tokens(self, tokens, aspect_masks = None, adj_masks = None, adv_masks = None, dependency_matrix = None, text_lens = None):

        
        max_len = max([len(x) for x in tokens])

        pad_result = torch.full((len(tokens), max_len), self.pad_token_id) 
        mask = torch.zeros(pad_result.size(),dtype=bool)
        text_mask = torch.zeros(pad_result.size(), dtype = bool)


        for i,x in enumerate(tokens):
            pad_result[i,:len(x)] = torch.tensor(tokens[i], dtype=torch.long)
            mask[i, :len(x)] = True
            text_mask[i, :text_lens[i]] = True
                
        if aspect_masks is not None:
                aspect_mask = torch.zeros((len(tokens),max_len), dtype=torch.bool)
                for i,x in enumerate(tokens):
                    aspect_mask[i, :len(x)] = torch.tensor(aspect_masks[i], dtype=torch.bool)

        if adj_masks is not None:
            adj_mask = torch.zeros((len(tokens),max_len), dtype=torch.bool)
            for i,x in enumerate(tokens):
                adj_mask[i, :len(x)] = torch.tensor(adj_masks[i], dtype=torch.bool)
        elif adj_masks is None:
            adj_mask = None
        
        if adv_masks is not None:
            adv_mask = torch.zeros((len(tokens),max_len),dtype= torch.bool)
            for i,x in enumerate(tokens):
                adv_mask[i, :len(x)] = torch.tensor(adv_masks[i], dtype=torch.bool)
        elif adv_masks is None:
            adv_mask = None
            
        if dependency_matrix is not None:
                ret_dependency_matrix = torch.zeros([len(tokens), max_len, max_len], dtype=torch.float)
                for i in range(len(tokens)):
                    dim = dependency_matrix[i].shape[0]
                    ret_dependency_matrix[i, :dim, :dim] = dependency_matrix[i]
            
        if aspect_masks is None:
                aspect_mask = None
        
            
        if dependency_matrix is None:
                ret_dependency_matrix = None

        

        return pad_result, mask, aspect_mask, adj_mask, adv_mask, ret_dependency_matrix, text_mask
        
    

