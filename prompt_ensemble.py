import imp
import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn as nn

from AnomalyCLIP_lib.pointnet2_sem_seg import get_model as Pointnet
from AnomalyCLIP_lib.neg_prompt import DefectPromptGenerator

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def encode_text_with_prompt_ensemble(model, texts, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence)
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details):
        super().__init__()
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]
        self.state_anomaly_list = [
            "{}",
        ]

        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        # pointnet
        self.pointnet = Pointnet(13)
        self.load_pointnet_weights('/root/autodl-tmp/GHP-CLIP/pretrained_weights/best_model.pth')
        self.point_feature_to_prompt_pos = nn.Sequential(
                nn.Linear(512, ctx_dim),
                nn.GELU(),
                nn.LayerNorm(ctx_dim),
            )
        self.point_feature_to_prompt_neg = nn.Sequential(
                nn.Linear(512, ctx_dim),
                nn.GELU(),
                nn.LayerNorm(ctx_dim),
            )
        self.point_feature_to_neg_point = DefectPromptGenerator(
                point_feat_dim = 128,
                text_embed_dim = 768,
                num_defect_tokens = 1,
                num_normal_prototypes = 32,
                top_k_outliers = 12,
                nhead = 4,
        )

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            print("Initializing class-specific contexts")
            ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
            ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape) # 4，768
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)


        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        _c_pos = torch.empty_like(ctx_vectors_pos)
        _c_neg = torch.empty_like(ctx_vectors_neg)
        nn.init.normal_(_c_pos, std = 0.02)
        nn.init.normal_(_c_neg, std = 0.02)
        self.ctx_pos_child = nn.Parameter(_c_pos)
        self.ctx_neg_child = nn.Parameter(_c_neg)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]


        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
     
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)


        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 2 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 2 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

        self.state_normal_list_child = ["", ]
        self.state_anomaly_list_child = ["a", ]

        prompts_pos_child = [prompt_prefix_pos + " " + _t + "."
                             for _t in self.state_normal_list_child]
        prompts_neg_child = [prompt_prefix_neg + " " + _t + "."
                             for _t in self.state_anomaly_list_child]
        tok_pos_child = torch.cat([tokenize(p) for p in prompts_pos_child])
        tok_neg_child = torch.cat([tokenize(p) for p in prompts_neg_child])
        with torch.no_grad():
            emb_pos_child = clip_model.token_embedding(tok_pos_child).type(dtype)
            emb_neg_child = clip_model.token_embedding(tok_neg_child).type(dtype)
            _, l_c, d_c = emb_pos_child.shape
            emb_pos_child = emb_pos_child.reshape(1, self.n_cls, l_c, d_c).permute(1, 0, 2, 3)
            _, l_c, d_c = emb_neg_child.shape
            emb_neg_child = emb_neg_child.reshape(1, self.n_cls, l_c, d_c).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos_child", emb_pos_child[:, :, :1, :])
        self.register_buffer("token_suffix_pos_child",
                             emb_pos_child[:, :, 1 + n_ctx_pos:, :])       # k=0
        self.register_buffer("token_prefix_neg_child", emb_neg_child[:, :, :1, :])
        self.register_buffer("token_suffix_neg_child",
                             emb_neg_child[:, :, 2 + n_ctx_neg:, :])       # k=1

        _, d_c = tok_pos_child.shape
        self.register_buffer("tokenized_prompts_pos_child",
                             tok_pos_child.reshape(1, self.n_cls, d_c).permute(1, 0, 2))
        _, d_c = tok_neg_child.shape
        self.register_buffer("tokenized_prompts_neg_child",
                             tok_neg_child.reshape(1, self.n_cls, d_c).permute(1, 0, 2))

    def load_pointnet_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu') 
        state_dict_to_load = (checkpoint['model_state_dict'])
        self.pointnet.load_state_dict(state_dict_to_load)

    def forward(self, pointcloud_input, cls_id =None):
        
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_pos_child = self.ctx_pos_child
        ctx_neg_child = self.ctx_neg_child

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        points_feat, l4_points = self.pointnet(pointcloud_input)  # (b,N,128), (b,512,16)
        pn_global_feat = torch.max(l4_points, dim=2)[0]
        pointnet_prompt_pos = self.point_feature_to_prompt_pos(pn_global_feat) # (B, D_text_embed)
        b = pointnet_prompt_pos.shape[0]
        pointnet_prompt_pos = pointnet_prompt_pos.unsqueeze(1).unsqueeze(1) # (b,1,1,768)
        
        pointnet_prompt_neg = self.point_feature_to_prompt_neg(pn_global_feat)
        pointnet_prompt_neg = pointnet_prompt_neg.unsqueeze(1).unsqueeze(1)

        prompt_neg_point = self.point_feature_to_neg_point(points_feat)
        prompt_neg_point = prompt_neg_point.unsqueeze(1)  # GDDM defect token (b,1,1,768)

        # parent normal
        prompts_pos = torch.cat(
            [
                prefix_pos.repeat(b,1,1,1),  # b,1,1,768
                ctx_pos.repeat(b,1,1,1),     # b,1,n_ctx_pos,768
                pointnet_prompt_pos,
                suffix_pos.repeat(b,1,1,1),  # b,1,-1,768
            ],
            dim=2,
        ) # b,1,77,768

        # parent anomalous: same layout as parent normal; ctx_pos / ctx_neg separate them
        prompts_neg = torch.cat(
            [
                prefix_neg.repeat(b,1,1,1),
                ctx_neg.repeat(b,1,1,1),
                pointnet_prompt_pos,
                suffix_neg.repeat(b,1,1,1),
            ],
            dim=2,
        )

        # child normal: no point-cloud token
        prompts_pos_child = torch.cat(
            [
                self.token_prefix_pos_child.repeat(b,1,1,1),
                ctx_pos_child.repeat(b,1,1,1),
                self.token_suffix_pos_child.repeat(b,1,1,1),
            ],
            dim=2,
        )
        # child anomalous: GDDM defect token
        prompts_neg_child = torch.cat(
            [
                self.token_prefix_neg_child.repeat(b,1,1,1),
                ctx_neg_child.repeat(b,1,1,1),
                prompt_neg_point,
                self.token_suffix_neg_child.repeat(b,1,1,1),
            ],
            dim=2,
        )

        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        _, _, l, d = prompts_pos_child.shape
        prompts_pos_child = prompts_pos_child.reshape(-1, l, d)
        _, _, l, d = prompts_neg_child.shape
        prompts_neg_child = prompts_neg_child.reshape(-1, l, d)
        # order: parent N / parent A / child N / child A; tokenized_prompts must match
        prompts = torch.cat([prompts_pos, prompts_neg,
                             prompts_pos_child, prompts_neg_child], dim=0)  # b*4,77,768

        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)  # 1,77
        tokenized_prompts_pos = tokenized_prompts_pos.repeat(b,1) # b,77
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts_neg = tokenized_prompts_neg.repeat(b,1)
        _, _, d_t = self.tokenized_prompts_pos_child.shape
        tok_pos_c = self.tokenized_prompts_pos_child.reshape(-1, d_t).repeat(b, 1)
        _, _, d_t = self.tokenized_prompts_neg_child.shape
        tok_neg_c = self.tokenized_prompts_neg_child.reshape(-1, d_t).repeat(b, 1)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg,
                                       tok_pos_c, tok_neg_c), dim = 0)  # b*4,77

        return prompts, tokenized_prompts, self.compound_prompts_text
