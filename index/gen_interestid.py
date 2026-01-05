import numpy as np
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse  # [新增]

# ================= 命令行参数解析 =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
parser.add_argument("--data_root", type=str, default="/home/xym/MQL4GRec/data", help="Data root path")
args = parser.parse_args()

# ================= 配置路径 (动态化) =================
dataset = args.dataset
data_root = args.data_root
BASE_PATH = os.path.join(data_root, dataset)

# 确保文件名格式正确，这里假设你的 embedding 文件名模式如下：
TEXT_EMB_PATH = os.path.join(BASE_PATH, f'{dataset}.emb-llama-td.npy')
IMG_EMB_PATH = os.path.join(BASE_PATH, f'{dataset}.emb-ViT-L-14.npy')
INTERACTION_PATH = os.path.join(BASE_PATH, f'{dataset}.inter.json')

OUTPUT_DIR = BASE_PATH
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'User_Interest_IDs.json')
OUTPUT_NPY = os.path.join(OUTPUT_DIR, 'User_Interest_IDs.npy')

# [重要] 建议修改 Prefix 以区别于 Item ID (<a_...>)
# 如果你使用的是3层码本，且想用 o, p, q
PREFIX = ["<o_{}>", "<p_{}>", "<q_{}>"] 
# 或者是你之前文件里的: ["<a_{}>", "<b_{}>", "<c_{}>"] (如果确定不冲突)
# NUM_EMB_LIST = [256, 256, 256] 
NUM_EMB_LIST = [256, 256, 256] 

USE_COSINE = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= PyTorch K-Means 实现 =================

def torch_kmeans(X, num_clusters, max_iter=100, use_cosine=True, seed=2024):
    """
    一个基于 PyTorch 的简易 K-Means 实现 (支持 GPU)
    """
    N, D = X.shape
    
    # 1. 初始化中心点 (随机选择)
    torch.manual_seed(seed)
    random_indices = torch.randperm(N)[:num_clusters]
    centroids = X[random_indices] # [K, D]
    
    if use_cosine:
        centroids = F.normalize(centroids, p=2, dim=1)
        X = F.normalize(X, p=2, dim=1)

    labels = None
    
    for i in range(max_iter):
        # 2. 计算距离/相似度
        if use_cosine:
            # Cosine: Dot product (因为已经 normalize 了)
            # [N, D] @ [D, K] -> [N, K]
            sims = torch.mm(X, centroids.T)
            new_labels = torch.argmax(sims, dim=1)
        else:
            # Euclidean: (x-c)^2 expanded
            # 这里用 cdist 更快: [N, K]
            dists = torch.cdist(X, centroids)
            new_labels = torch.argmin(dists, dim=1)

        # 检查收敛
        if labels is not None and torch.equal(new_labels, labels):
            break
        labels = new_labels

        # 3. 更新中心点 (Vectorized update)
        # 将 labels 转为 one-hot: [N, K]
        one_hot = F.one_hot(labels, num_clusters).float()
        
        # 聚合每个簇的向量和: [K, N] @ [N, D] -> [K, D]
        sum_vectors = torch.mm(one_hot.T, X)
        
        # 计算每个簇的数量
        counts = one_hot.sum(dim=0).unsqueeze(1) + 1e-6 # [K, 1] 防止除零
        
        new_centroids = sum_vectors / counts
        
        if use_cosine:
            new_centroids = F.normalize(new_centroids, p=2, dim=1)
            
        centroids = new_centroids

    return centroids, labels

# ================= 核心类：PyTorch 残差量化 =================

class ResidualQuantizerTorch:
    def __init__(self, num_emb_list, use_cosine=True, device='cuda'):
        self.num_emb_list = num_emb_list
        self.use_cosine = use_cosine
        self.device = device

    def train_and_encode(self, X_np):
        print(f"3. Training RQ-Means on {self.device}...")
        
        # 转为 Tensor 并移至 GPU
        residuals = torch.from_numpy(X_np).float().to(self.device)
        n_samples = residuals.shape[0]
        
        # 结果容器 (CPU numpy)
        all_codes = np.zeros((n_samples, len(self.num_emb_list)), dtype=int)
        
        for i, n_clusters in enumerate(self.num_emb_list):
            print(f"   Layer {i+1}: Clustering {n_samples} vectors into {n_clusters} codes...")
            
            # 如果是 Cosine 模式，确保输入是归一化的
            if self.use_cosine:
                residuals = F.normalize(residuals, p=2, dim=1)
            
            # 运行 K-Means
            centers, codes = torch_kmeans(
                residuals, 
                n_clusters, 
                max_iter=50, # 50次迭代通常足够
                use_cosine=self.use_cosine
            )
            
            # 保存 Code
            all_codes[:, i] = codes.cpu().numpy()
            
            # 计算残差 (Residual = Current - Center)
            # 获取对应的中心点
            reconstructed = centers[codes] # [N, D]
            
            # 更新残差
            residuals = residuals - reconstructed
            
        return all_codes

# ================= 数据加载 (保持不变) =================

def load_and_fuse_embeddings():
    print("1. Loading Embeddings...")
    try:
        text_emb = np.load(TEXT_EMB_PATH)
        img_emb = np.load(IMG_EMB_PATH)
        # Numpy 端的预处理
        text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
        img_emb = img_emb / (np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-8)
        fused_emb = np.concatenate([text_emb, img_emb], axis=1)
        return fused_emb
    except Exception as e:
        print(f"Error: {e}")
        return None

def compute_user_embeddings(fused_item_emb, json_path):
    print(f"2. Computing User Mean Embeddings...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    user_history = {}
    if isinstance(data, list):
        for entry in data:
            uid = entry.get('user_id') or entry.get('user')
            iid = entry.get('item_id') or entry.get('item_id_list')
            if uid is not None and iid is not None:
                user_history[str(uid)] = iid if isinstance(iid, list) else [iid]
    elif isinstance(data, dict):
        user_history = data

    user_embs = []
    user_ids = [] 
    num_items = fused_item_emb.shape[0]
    
    for uid, item_list in tqdm(user_history.items(), desc="Pooling Users"):
        train_items = item_list[:-2]
        valid_indices = [int(i) for i in train_items if 0 <= int(i) < num_items]
        if valid_indices:
            vectors = fused_item_emb[valid_indices]
            mean_vector = np.mean(vectors, axis=0)
            user_embs.append(mean_vector)
            user_ids.append(str(uid))
    
    # 返回 float32
    return np.array(user_embs, dtype=np.float32), user_ids

# ================= 主流程 =================
if __name__ == "__main__":
    print(f"Processing Dataset: {dataset}")
    
    fused_items = load_and_fuse_embeddings()
    if fused_items is not None:
        user_matrix, user_id_list = compute_user_embeddings(fused_items, INTERACTION_PATH)
        print(f"   User Matrix Shape: {user_matrix.shape}")

        # 使用 PyTorch 量化器
        # 注意：你需要确保 ResidualQuantizerTorch 类在脚本里定义了
        quantizer = ResidualQuantizerTorch(NUM_EMB_LIST, USE_COSINE, DEVICE)
        user_codes = quantizer.train_and_encode(user_matrix)

        print(f"[{dataset}] Saving results...")
        all_indices_dict = {}
        for idx, uid in enumerate(user_id_list):
            codes = user_codes[idx]
            # 根据 PREFIX 格式化
            formatted_code = [PREFIX[i].format(int(c)) for i, c in enumerate(codes)]
            all_indices_dict[uid] = formatted_code

        with open(OUTPUT_JSON, 'w') as f:
            json.dump(all_indices_dict, f, indent=4)
            
        # np.save(OUTPUT_NPY, user_codes) # 可选
        print(f"[{dataset}] Done! Saved to {OUTPUT_JSON}")