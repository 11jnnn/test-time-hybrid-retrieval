import numpy as np
import torch
import torch.nn.functional as F

from dataset_configs import DataSplit, RagDataset
from embedding_configs import EncoderConfig
from utils import get_device, slice_sparse_coo_tensor


def score_multi_vector(
    emb_queries: torch.Tensor | list[torch.Tensor],
    emb_passages: torch.Tensor | list[torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """
    Evaluate the similarity scores using the MaxSim scoring function.

    Inputs:
        - emb_queries: List of query embeddings, each of shape (n_seq, emb_dim).
        - emb_passages: List of document embeddings, each of shape (n_seq, emb_dim).
        - batch_size: Batch size for the similarity computation.
    """
    if len(emb_queries) == 0:
        raise ValueError("No queries provided")
    if len(emb_passages) == 0:
        raise ValueError("No passages provided")

    if emb_queries[0].device != emb_passages[0].device:
        raise ValueError("Queries and passages must be on the same device")

    if emb_queries[0].dtype != emb_passages[0].dtype:
        raise ValueError("Queries and passages must have the same dtype")

    scores: list[torch.Tensor] = []

    for i in range(0, len(emb_queries), batch_size):
        batch_scores = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            emb_queries[i: i + batch_size],
            batch_first=True,
            padding_value=0,
        )
        for j in range(0, len(emb_passages), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                emb_passages[j: j + batch_size],
                batch_first=True,
                padding_value=0,
            )
            batch_scores.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        batch_scores = torch.cat(batch_scores, dim=1)
        scores.append(batch_scores)

    return torch.cat(scores, dim=0)

#计算 query 与所有文档的相似度，再为每个 query 取分数最高的 k 个文档。
def get_topk(d_embs, q_embs, k, is_multi=False):
    device = get_device()
    #多向量模型用多个向量表示一个 query 或一篇文档，因此需要专门的评分方法。
    #这个仓库的 score_multi_vector() 使用 MaxSim：
    if is_multi:
        similarity = score_multi_vector(q_embs, d_embs, batch_size=8)
    else:
        #这行代码本身计算的是点积；如果 query 和文档向量都已经做了 L2 归一化，它就等价于余弦相似度。
        similarity = torch.matmul(q_embs.to(device), d_embs.T.to(device))
    #如果相似度矩阵是稀疏张量，转成普通稠密张量
    if similarity.is_sparse:
        similarity = similarity.to_dense()
    #对每个 query，选出最高的 k 个分数
    top_vals, top_idx = torch.topk(similarity, k=k, dim=-1)
    return top_vals, top_idx


class Retriever:
    def __init__(self, encoder_config: EncoderConfig):
        file_suffix = encoder_config.embedder.file_suffix
        self.id = encoder_config.model_id.split("/")[-1] + file_suffix.replace("_embeddings", "")
        self.modality = encoder_config.modality
        self.is_multi = encoder_config.embedder.is_multi_vector
        self.is_sparse = encoder_config.embedder.is_sparse

        self.d: torch.Tensor | None = None
        self.q_dict: dict[DataSplit, torch.Tensor] | None = None
        self._emb_cache = {}

    def load_embs(self, dataset: RagDataset) -> tuple[torch.Tensor, dict[DataSplit, torch.Tensor]]:
        key = (str(dataset.path.resolve()), self.id)
        if key in self._emb_cache:
            self.d, self.q_dict = self._emb_cache[key]
            return self.d, self.q_dict

        def load_to_torch(embs_obj: np.ndarray) -> torch.Tensor | list[torch.Tensor]:
            if self.is_sparse:
                return [torch.sparse_coo_tensor(indices=torch.from_numpy(arr.indices).unsqueeze(0), values=arr.values)
                        for arr in list(embs_obj)]
            elif embs_obj.dtype == np.object_:  # multi-vector embedding tensors with varying sizes, which we treat as an object/list
                return [torch.from_numpy(arr) for arr in list(embs_obj)]
            else:  # a single tensor with all the (single- or multi-vector) embeddings
                return torch.from_numpy(embs_obj)

        p = dataset.path / (self.id + "_embeddings")
        k = self.get_embs_key()
        d = load_to_torch(
            np.load(p / "docs.npz", allow_pickle=True)[k])
        q = load_to_torch(
            np.load(p / "queries.npz", allow_pickle=True)[k])

        if self.is_multi:
            d, q = self.pad_multi_vectors(d), self.pad_multi_vectors(q)
        elif self.is_sparse:
            d, q = self.join_sparse_vectors(d, q)
        else:
            d, q = self.normalize(d), self.normalize(q)

        self.d = d
        assert q.size(0) == dataset.num_queries, \
            "the number of query embeddings doesn't match the the number of queries in the dataset"
        
        dev_q_idxs = dataset.get_queries_indices(split=DataSplit.DEV)
        test_q_idxs = dataset.get_queries_indices(split=DataSplit.TEST)
        if self.is_sparse:
            dev_q = slice_sparse_coo_tensor(q, dev_q_idxs)
            test_q = slice_sparse_coo_tensor(q, test_q_idxs)
        else:
            dev_q = q[dev_q_idxs]
            test_q = q[test_q_idxs]
        self.q_dict = {DataSplit.DEV: dev_q,
                       DataSplit.TEST: test_q}
        self._emb_cache[key] = (self.d, self.q_dict)
        return self.d, self.q_dict

    def clean_embs(self, dataset):
        key = (str(dataset.path.resolve()), self.id)
        if key in self._emb_cache:
            del self._emb_cache[key]

    def set_queries(self, new_q):
        self.q_dict = new_q

    def get_embs_key(self):
        return "arr_0"

    @staticmethod
    def normalize(t: torch.Tensor):
        return F.normalize(t.float(), dim=-1)

    @staticmethod
    def pad_multi_vectors(t: torch.Tensor):
        return torch.nn.utils.rnn.pad_sequence(t, batch_first=True, padding_value=0)

    @staticmethod
    def join_sparse_vectors(a: list[torch.Tensor], b: list[torch.Tensor]):
        def build_sparse(indices_list, values_list, rows):
            if not indices_list:
                return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                               torch.tensor([], dtype=torch.float32),
                                               size=(rows, max_cols), dtype=torch.float32)
            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list)
            return torch.sparse_coo_tensor(indices, values, size=(rows, max_cols), dtype=torch.float32)

        all_indices_a = []
        all_values_a = []
        all_indices_b = []
        all_values_b = []
        max_cols = 0

        for i, t in enumerate(a):
            t = t.coalesce()
            idx = t.indices()
            val = t.values()

            if idx.shape[0] != 1:
                raise ValueError("Expected 1D sparse tensor")

            col_idx = idx[0]
            row_idx = torch.full((col_idx.size(0),), i, dtype=torch.long)
            indices = torch.stack([row_idx, col_idx])

            all_indices_a.append(indices)
            all_values_a.append(val)

            max_cols = max(max_cols, t.shape[0])

        for i, t in enumerate(b):
            t = t.coalesce()
            idx = t.indices()
            val = t.values()

            if idx.shape[0] != 1:
                raise ValueError("Expected 1D sparse tensor")

            col_idx = idx[0]
            row_idx = torch.full((col_idx.size(0),), i, dtype=torch.long)
            indices = torch.stack([row_idx, col_idx])

            all_indices_b.append(indices)
            all_values_b.append(val)

            max_cols = max(max_cols, t.shape[0])

        A = build_sparse(all_indices_a, all_values_a, len(a))
        B = build_sparse(all_indices_b, all_values_b, len(b))
        return A, B

    def run_retrieval(self, dataset, q=None, d=None, split=DataSplit.TEST, k=50):
        device = get_device()
        if q is None:
            q = self.q_dict[split].to(device)
        if d is None:
            d = self.d.to(device)
        #计算相似度，取前 k 个文档
        #get_topk() 根据向量类型计算相似度，再为每个 query 选出分数最高的 k 个文档。
        #top_vals	检索到的文档相似度分数	[query数量, k]
         #top_idx	这些文档在文档集合中的索引	[query数量, k]
        top_vals, top_idx = get_topk(d, q, k=k, is_multi=self.is_multi)
      #创建结果字典，并取出数据集中的问题文本。 
        results = {}
        questions = dataset.benchmark['question']
        #获取当前划分对应的 query 索引，再选出这些问题。
        q_indices = dataset.get_queries_indices(split)
        questions = questions[q_indices.numpy()]
        #把检索张量整理成字典
        for q, val_row, idx_row in zip(
                questions, top_vals.cpu().tolist(), top_idx.cpu().tolist()
        ):
            results[q] = [(idx, val) for (idx, val) in zip(idx_row, val_row)]
        return results, top_idx

    def compute_scores(self, q, d):
        if self.is_multi:
            similarity = score_multi_vector(q, d, batch_size=16)
            similarity = similarity / q.size(1)
        else:
            similarity = torch.matmul(q, d.T)

        if similarity.is_sparse:
            similarity = similarity.to_dense()
        return similarity
