#@title dataloader
import numpy as np
import pandas as pd

from torch.utils.data import Dataset



class ASSIST2017_PID_DIFF(Dataset):
    def __init__(self, max_seq_len, subject, config=None, dataset_dir="", q2idx=None, pid2idx=None) -> None:
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
        self.num_r = 2 #self.r_list.shape[0]
        self.num_pid = len(self.pid2idx) #self.pid_list.shape[0]
        self.num_diff = self.diff_list.shape[0] # not necessary

        self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.diff_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df= pd.read_csv(self.dataset_dir, encoding="ISO-8859-1")
        df = df[(df["correct"] == 0) | (df["correct"] == 1)]
        if "subject_id" in df.columns:
            df = df[df['subject_id'] == self.subject]
        # max_dif = max(df['difficulty'])
        # df['difficulty'] = df['difficulty'].apply(lambda x: round(x, 2) * 100)
        print(df.head())
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