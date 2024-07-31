import math
import numpy as np
import pandas as pd


import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ASSIST2017_PID_DIFF(Dataset):
    def __init__(self, max_seq_len, subject=None, config=None, dataset_dir=None, q2idx=None, pid2idx=None) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.subject = subject
        self.q2idx  = q2idx
        self.pid2idx = pid2idx

        # 추가
        self.config = config
        self.q_seqs, self.r_seqs, self.q_list, self.r_list, \
            self.q2idx, self.pid2idx, \
            self.pid_seqs,  self.diff_seqs, self.pid_list, self.diff_list = self.preprocess()

        self.num_q = len(self.q2idx) #self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = len(self.pid2idx) #self.pid_list.shape[0]
        self.num_diff = self.diff_list.shape[0]

        self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.diff_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df= pd.read_csv(self.dataset_dir, encoding="ISO-8859-1") if type(self.dataset_dir) == str  else self.dataset_dir
        # max_dif = max(df['difficulty'])
        df['difficulty'] = df['difficulty'].apply(lambda x: round(x,2))
    
        q_list = np.unique(df["skill_id"].values)
        r_list = np.unique(df["correct"].values)
        pid_list = np.unique(df["item_id"].values)
        diff_list = np.unique(df['difficulty'].values)

        q2idx = {q: idx for idx, q in enumerate(q_list)}  if self.q2idx is None else {float(k):float(self.q2idx[k]) for k in self.q2idx}
        pid2idx = {pid: idx for idx, pid in enumerate(pid_list)} if self.pid2idx is None else {float(k):float(self.pid2idx[k]) for k in self.pid2idx}

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        diff_seqs = []

        for idx, u in enumerate( np.unique(df["user_id"].values)):
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])
            diff_seq =  df_u["difficulty"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            diff_seqs.append(diff_seq)

        return q_seqs, r_seqs, q_list, r_list, q2idx, pid2idx, pid_seqs, diff_seqs, pid_list, diff_list #끝에 두개 추가

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, diff_seqs, max_seq_len, pad_val=-1):
        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_diff_seqs = []

        for q_seq, r_seq, pid_seq, diff_seq in zip(q_seqs, r_seqs, pid_seqs, diff_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                proc_diff_seqs.append(diff_seq[i:i + max_seq_len])

                i += max_seq_len

            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_pid_seqs.append(
                np.concatenate(
                    [
                        pid_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_diff_seqs.append(
                np.concatenate(
                    [
                        diff_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )

        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_diff_seqs
    

# For Test MLM : The last of seq will be changed to the <MASK>
def Mlm4BertTest(r_seqs, mask_seqs):
    #|r_seqs| = (bs, n)

    mlm_r_seqs = []
    mlm_idxs = []

    for r_seq, mask_seq in zip(r_seqs, mask_seqs):
        r_len = r_seq.size(0)

        real_r_seq = torch.masked_select(r_seq, mask_seq).cpu()
        real_r_seq_len = real_r_seq.size(0)

        # last index of real_r_seq
        mlm_idx = real_r_seq_len - 1
        # last index get a <MASK>, <MASK> is 2
        real_r_seq[mlm_idx] = 2

        pad_len = r_len - real_r_seq_len
        pad_seq = torch.full((1, pad_len), 3).squeeze(0) # <PAD> is 3
        pad_r_seq = torch.cat((real_r_seq, pad_seq), dim=-1)
        mlm_r_seqs.append(pad_r_seq)

        mlm_zeros = np.zeros(shape=(r_len, ))
        mlm_zeros[mlm_idx] = 1
        mlm_idxs.append(mlm_zeros)

    mlm_r_seqs = torch.stack(mlm_r_seqs)
    mlm_idxs = torch.BoolTensor(mlm_idxs)

    return mlm_r_seqs, mlm_idxs
    # |mlm_r_seqs| = (bs, n)
    # |mask_seqs| = (bs, n)

def pid_diff_collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    pid_seqs = []
    diff_seqs = []

    for q_seq, r_seq, pid_seq, diff_seq in batch:
        q_seqs.append(torch.Tensor(q_seq))
        r_seqs.append(torch.Tensor(r_seq))
        pid_seqs.append(torch.Tensor(pid_seq))
        diff_seqs.append(torch.Tensor(diff_seq))

    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    pid_seqs = pad_sequence(
        pid_seqs, batch_first=True, padding_value=pad_val
    )
    diff_seqs = pad_sequence(
        diff_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (q_seqs != pad_val)

    q_seqs, r_seqs, pid_seqs, diff_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs, pid_seqs * mask_seqs, diff_seqs * mask_seqs

    return q_seqs, r_seqs, pid_seqs, diff_seqs, mask_seqs


# For inference
def model_infer(model, data_loader, device=torch.device("cpu")):
    y_scores, items, skills, diffs = [], [], [], []
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            q_seqs, r_seqs, pid_seqs, diff_seqs, mask_seqs = data

            q_seqs = q_seqs.to(device)
            r_seqs = r_seqs.to(device)
            pid_seqs = pid_seqs.to(device)
            diff_seqs = diff_seqs.to(device)
            mask_seqs = mask_seqs.to(device)

            mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)
            mlm_r_seqs, mlm_idxs = mlm_r_seqs.to(device), mlm_idxs.to(device)

            y_hat = model(
                q_seqs.long(),
                mlm_r_seqs.long(),
                pid_seqs.long(),
                diff_seqs.long(),
                mask_seqs.long()
            ).to(device).squeeze()

            items   += [torch.masked_select(pid_seqs,  mlm_idxs)]
            skills  += [torch.masked_select(q_seqs,  mlm_idxs)]
            diffs   += [torch.masked_select(diff_seqs, mlm_idxs)]
            y_scores+= [torch.masked_select(y_hat, mlm_idxs)]

    y_scores = torch.cat(y_scores).detach().cpu().numpy()
    items    = torch.cat(items).detach().cpu().numpy()
    skills   = torch.cat(skills).detach().cpu().numpy()
    diffs    = torch.cat(diffs).detach().cpu().numpy()
    return y_scores, items, skills, diffs

#############################################################################
#                     _____    ____    ____                                 #
#           /\  /\   |     |  |    \  |      |                              #
#          /  \/  \  | -+- |  |     | |----  |                              #
#         /        \ |_____|  |____/  |____  |____                          #
#                                                                           #
#############################################################################

class HookHelper(nn.Module):
    '''custom Module wrapper for register_forward_hook
    '''
    def forward(self, x):
        return x

# SeparableConv1D
class SeparableConv1D(nn.Module):
    def __init__(self, input_filters, output_filters, kernel_size):
        super().__init__()

        # input_filters = 512 <- hidden_size
        # output_filters = 256 <- all_attn_head_size

        self.depthwise = nn.Conv1d(input_filters, input_filters, kernel_size=kernel_size, groups=input_filters, padding=kernel_size //2, bias = False)
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        self.depthwise.weight.data.normal_(mean=0.0, std=0.02)
        self.pointwise.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, hidden_states):
        # |hidden_states| = (bs, hs, n)

        x = self.depthwise(hidden_states)
        # |x| = (bs, hs, n)

        x = self.pointwise(x)
        # |x| = (bs, hs/2(all_attn_h_size), n)

        x += self.bias
        # |x| = (bs, hs/2(all_attn_h_size), n)
        return x

# Thank for the Huggingface and Author of AKT
# Combined the Monotonic Attention and Span Dynamic Convolutional Attention
class MonotonicConvolutionalMultiheadAttention(nn.Module):
    # hidden % n_splits == 0
    def __init__(self, hidden_size, n_splits, dropout_p, head_ratio=2, conv_kernel_size=9):
        super().__init__()
        # default: n_splits = 16, head_ratio = 2

        new_num_attention_heads = n_splits // head_ratio
        self.num_attention_heads = new_num_attention_heads
        # default: self.new_num_attention_heads = 8

        self.head_ratio = head_ratio
        # default: self.head_ratio = 2

        self.conv_kernel_size = conv_kernel_size
        # default: self.conv_kernel_size = 9

        self.attention_head_size = hidden_size // n_splits
        # default: self.attention_head_size = 512//16 = 32

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # default: self.all_head_size = 32 * 8 = 256

        # linear layers for query, key, value
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=False) # 512 -> 256

        # layers for span dynamic convolutional attention
        self.key_conv_attn_layer = SeparableConv1D(
            hidden_size, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = nn.Linear(self.all_head_size,
                                        self.num_attention_heads * self.conv_kernel_size # 8 * 9 = 72
                                        )
        self.conv_out_layer = nn.Linear(hidden_size, self.all_head_size)

        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0]
        )

        # this is for the distance function
        self.gammas = nn.Parameter(torch.zeros(self.num_attention_heads, 1, 1))
        self.dropout = nn.Dropout(dropout_p)

        self.filters = None
        self.attn_scores = None
        self.conv_value_vectors = None

        self.conv_hook_helper = HookHelper()
        self.attn_hook_helper = HookHelper()

    def forward(self, Q, K, V, mask=None):
        # |Q| = |K| = |V| = (bs, n, hs)
        # |mask| = (bs, n)

        batch_size = Q.size(0)

        mixed_query_layer = self.query(Q)
        mixed_key_layer = self.key(K)
        mixed_value_layer = self.value(V)
        # |mixed_query_layer| = |mixed_key_layer| = |mixed_value_layer| = (bs, n, hs/2(all_attn_h_size))

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(
            K.transpose(1, 2) # |hidden_states.transpose(1, 2)| = (bs, hs, n)
        )
        # |mixed_key_conv_attn_layer| = (bs, hs/2(all_attn_h_size), n)
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        # |mixed_key_conv_attn_layer| = (bs, n, hs/2(all_attn_h_size))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # |query_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)

        ##############
        # conv layer #
        ##############
        # element-wise multiply of conv key and query
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        # |conv_attn_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        # |conv_kernel_layer| = (bs, n, (n_attn_h * conv_kernel_size) = (64, 100, 8 * 9) = (64, 100, 72)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        # |conv_kernel_layer| = (51200, 9, 1)
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        # |conv_kernel_layer| = (51200, 9, 1), 각 head별 확률값들을 도출하는 듯

        # save the filter pattern
        self.filters = conv_kernel_layer.detach().clone()

        # q X k is matmul with v
        conv_out_layer = self.conv_out_layer(V)
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size))
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        # |conv_out_layer| = (bs, hs/2(all_attn_h_size), n, 1)

        self.conv_value_vectors = conv_out_layer.detach().clone()

        # unfold 참고 -> #https://www.facebook.com/groups/PyTorchKR/posts/1685133764959631/
        conv_out_layer = nn.functional.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        # |conv_out_layer| = (64, 2304, 100)
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        # |conv_out_layer| = (bs, n, hs/2(all_attn_h_size), conv_kernal_size)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        # |conv_out_layer|, default = (51200, 32, 9)
        # matmul(q X k, v)
        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        # |conv_out_layer|, default = (51200, 32, 1)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        # |conv_out_layer|, default = (6400, 256)
        conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        # |conv_out| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)

        ###################
        # self_attn layer #
        ###################
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # |attention_scores| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # |attention_scores| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)

        #####################
        # distance function #
        #####################
        dist_scores = self.dist_func(attention_scores, mask)
        # |dist_scores| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)
        m = nn.Softplus()
        # gamma is learnable decay rate parameter
        gamma = -1.0 * m(self.gammas).unsqueeze(0)
        # Now after do exp(gamma * distance) and then clamp to 1e-5 to 1e-5
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )
        # |total_effect| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)

        attention_scores = attention_scores * total_effect
        # |attention_scores| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)

        # |mask| = (bs, n)
        attention_mask = self.get_extended_attention_mask(mask)
        # |attention_mask| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)
        attention_scores = attention_scores.masked_fill_(attention_mask==0, -1e8)
        # |attention_scores| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)
        attention_probs = self.dropout(attention_probs)
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        # save the attention scores
        self.attn_scores = attention_probs.detach().clone()

        context_layer = torch.matmul(attention_probs, value_layer)
        # |context_layer| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # |context_layer| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)

        ###############
        # hook helper #
        ###############
        context_layer = self.attn_hook_helper(context_layer)
        conv_out = self.conv_hook_helper(conv_out)

        #########################################
        # concat with conv and self_attn values #
        #########################################
        context_layer = torch.cat([context_layer, conv_out], 2)
        # |context_layer| = (bs, n, n_attn_head * 2, attn_head_size) = (64, 100, 16, 32)
        new_context_layer_shape = context_layer.size()[:-2] + \
             (self.head_ratio * self.all_head_size,)
        # new_context_layer_shape = (bs, n, hs)
        context_layer = context_layer.view(*new_context_layer_shape)
        # |context_layer| = (bs, n, hs)

        outputs = context_layer
        # |context_layer| = (bs, n, hs)
        # if you need attention_probs, add the return
        # |attention_probs| = (bs, n_attn_head, n, n) = (64, 8, 100, 100)

        # |outputs| = (bs, n, hs)
        return outputs

    # Thanks for the AKT's author and Upstage
    # this is the distance function, this function don't use grad
    @torch.no_grad()
    def dist_func(self, attention_scores, mask):

        scores = attention_scores
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

        x1 = torch.arange(seqlen).expand(seqlen, -1)
        x2 = x1.transpose(0, 1).contiguous()

        attention_mask = self.get_extended_attention_mask(mask)

        scores_ = scores.masked_fill_(attention_mask == 0, -1e32)

        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * attention_mask.float()

        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        """
        >>> x1-x2
            tensor([[ 0,  1,  2,  3,  4],
                    [-1,  0,  1,  2,  3],
                    [-2, -1,  0,  1,  2],
                    [-3, -2, -1,  0,  1],
                    [-4, -3, -2, -1,  0]])

        >>> torch.abs(x1-x2)
            tensor([[0, 1, 2, 3, 4],
                    [1, 0, 1, 2, 3],
                    [2, 1, 0, 1, 2],
                    [3, 2, 1, 0, 1],
                    [4, 3, 2, 1, 0]])
        """
        device = distcum_scores.get_device()

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(
            torch.FloatTensor
        )
        # |position_effect| = (1, 1, seqlen, seqlen)
        # print("line 278 device =",device)
        if device >= 0:
            position_effect  = position_effect.to(device)

        # dist_score => d(t, tau)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )

        dist_scores = dist_scores.sqrt().detach()

        # |dist_scores| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)
        return dist_scores

    # this is for attention mask
    @torch.no_grad()
    def get_extended_attention_mask(self, mask):
        # |mask| = (bs, n)
        mask_shape = mask.size() + (mask.size(1), self.num_attention_heads)
        # mask_shape = (bs, n, n, n_attn_head)
        mask_enc = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), mask.size(1) * self.num_attention_heads).bool()
        # |mask_enc| = (bs, n, n * n_attn_head)

        mask_enc = mask_enc.view(*mask_shape)
        # |mask_enc| = (bs, n, n, n_attn_head), default = (64, 100, 100, 8)

        return mask_enc.permute(0, 3, 2, 1)
        # |mask_enc| = (bs, n_attn_head, n, n), default = (64, 8, 100, 100)

    # for attention, last dim will be divied to n_attn_head, and get a new shape
    def transpose_for_scores(self, x):
        # |x| = (bs, n, hs/2(all_attn_h_size))

        # 마지막 차원을 n_attn_head의 수만큼으로 나눔
        new_x_shape = x.size()[:-1] + \
             (self.num_attention_heads, self.attention_head_size)
        # |x.size()[:-1]| = (bs, n)
        # self.new_num_attention_heads = 8
        # self.attention_head_size = 32
        # |new_x_shape| = (bs, n, new_num_attention_heads, attention_head_size)

        x = x.view(*new_x_shape)
        # |x| = (bs, n, n_attn_head, attn_head_size) = (64, 100, 8, 32)

        return x.permute(0, 2, 1, 3)
        # |x| = (bs, n_attn_head, n, attn_head_size) = (64, 8, 100, 32)


class EncoderBlock(nn.Module):

    def __init__(self, hidden_size, n_splits, use_leakyrelu, max_seq_len, dropout_p=.1, **kwargs):
        super().__init__()

        self.use_leakyrelu = use_leakyrelu

        self.attn = MonotonicConvolutionalMultiheadAttention(hidden_size, n_splits, dropout_p)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            # if you want to use gelu, then you have to change config option
            nn.LeakyReLU() if self.use_leakyrelu else self.gelu(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x| = (bs, n, emb_size), torch.float32 & |mask| = (bs, n, n)

        # Pre-LN: |z| = (bs, n, emb_size)
        z = self.attn_norm(x)

        # x+ means redisual connection = |z| = (bs, n, hs)
        z = x + self.attn_dropout(self.attn(Q=z,K=z,V=z,mask=mask))

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        return z, mask

    # Thanks for the upstage
    # upstage's gelu
    def gelu(x):
        """Upstage said:
            Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Thanks for the Kihyeon Kim
class MySequential(nn.Sequential):
    # New Sequential function
    # this can handle the tuple also
    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

        for module in self._modules.values():
            x = module(*x)

        return x

# This is the main model
class MonaCoBERT_CTT(nn.Module):

    def __init__(
        self, num_q, num_r, num_pid, num_diff,
        hidden_size, output_size,
        num_head, num_encoder,
        max_seq_len, device, use_leakyrelu, dropout_p=.1,):
        self.num_q = num_q
        self.num_r = num_r + 2 # '+2' is for 1(correct), 0(incorrect), <PAD>, <MASK>
        self.num_pid = num_pid
        self.num_diff = 101 # hard coding

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_encoder = num_encoder
        self.max_seq_len = max_seq_len
        self.device = device
        self.use_leakyrelu = use_leakyrelu
        self.dropout_p = dropout_p

        super().__init__()
        # question embedding
        self.emb_q = nn.Embedding(self.num_q, self.hidden_size).to(self.device)
        # response embedding
        self.emb_r = nn.Embedding(self.num_r, self.hidden_size).to(self.device)
        # item embedding
        self.emb_pid = nn.Embedding(self.num_pid, self.hidden_size).to(self.device)
        # diffical embedding
        self.emb_diff = nn.Embedding(self.num_diff, self.hidden_size).to(self.device)
        # positional embedding
        self.emb_p = nn.Embedding(self.max_seq_len, self.hidden_size).to(self.device)
        self.emb_dropout = nn.Dropout(self.dropout_p)

        # Using MySequential
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                num_head,
                self.use_leakyrelu,
                self.max_seq_len,
                dropout_p,
              ) for _ in range(num_encoder)],
        )

        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer.
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid() # Binary
        )

    # Learnable Positional embedding
    def _positional_embedding(self, q):
        # |q| = (bs, n)
        # |r| = (bs, n)
        seq_len = q.size(1)
        # seq_len = (n,)
        pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(q).to(self.device)
        # |pos| = (bs, n)
        pos_emb = self.emb_p(pos)
        # |emb| = (bs, n, hs)

        return pos_emb

    def forward(self, q, r, pid, diff, mask):
        # |q| = (bs, n), |r| = (bs, n), |mask| = (bs, n)

        emb = self.emb_q(q) + self.emb_r(r) + self.emb_pid(pid) + self.emb_diff(diff) + self._positional_embedding(q)
        # |emb| = (bs, n, emb_size)
        # print(emb.shape)
        z = self.emb_dropout(emb)
        # |z| = (bs, n, emb_size)
        # print(z.shape)

        # |mask_enc| = (bs, n, n)
        # |z| = (bs, n, emb_size)
        z, _ = self.encoder(z, mask)
        # |z| = (bs, n, hs)

        y_hat = self.generator(z)
        #|y_hat| = (bs, n, output_size=1)

        return y_hat