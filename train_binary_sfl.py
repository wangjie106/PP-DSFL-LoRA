# filename: train_binary_sfl.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # --- [新增] ---
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    matthews_corrcoef
)
from sklearn.cluster import KMeans
import logging
import time
import csv
import random
import copy
import os
from dataclasses import dataclass, field, asdict
from typing import List

# 假设 data_utils.py 在同一目录下
from data_utils import FT_Dataset

# ============================== 0. Focal Loss 定义 (新增) ==============================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): 聚焦参数，通常设为 2.0。
            alpha (list/tensor): 类别权重。例如 [0.7, 0.3] 表示给类别 0 (Normal) 更大的权重。
            reduction (str): 'mean' 或 'sum'。
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits [Batch, Num_Classes]
        # targets: labels [Batch]
        
        # 1. 计算标准 CE Loss (保留每个样本的 loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. 获取预测正确的概率 pt
        pt = torch.exp(-ce_loss)
        
        # 3. 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 4. 应用 Alpha 类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # gather: 根据 target 的索引取出对应的 alpha
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================== 1. 配置 (Configuration) ==============================
@dataclass
class ExperimentConfig:
    exp_name: str = "binary_clustered_ga_splitfed_focal" # 修改实验名
    rounds: int = 50                
    num_clients: int = 10            
    clients_per_round: int = 4       
    local_epochs: int = 1            
    batch_size: int = 16
    max_seq_length: int = 128
    
    num_classes: int = 2             
    
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    lr: float = 2e-5
    model_name: str = "./roberta-base-local" # 请确保本地有此模型
    seed: int = 42
    aggregation_method: str = "FedDW" 
    num_data_clusters: int = 3
    clustering_interval: int = 5
    
    # --- [新增] Focal Loss 配置 ---
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    # 针对 Normal 召回率低的问题，可以给 Normal (Class 0) 更大权重
    # 建议尝试: None (自动) 或 [0.65, 0.35] (手动加权 Normal)
    focal_alpha: List[float] = None 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ============================== 2. 数据集工具 ==============================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs): self.dataset, self.idxs = dataset, list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item): return self.dataset[self.idxs[item]]

# ============================== 3. 模型定义 (Split Learning) ==============================

class ClientModel(nn.Module):
    def __init__(self, embeddings, encoder_layers, attention_mask_getter):
        super().__init__()
        self.embeddings = embeddings
        self.encoder_layers = encoder_layers
        self.get_extended_attention_mask = attention_mask_getter
    def forward(self, input_ids, attention_mask, device):
        hidden_states = self.embeddings(input_ids)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        for layer in self.encoder_layers: hidden_states = layer(hidden_states, extended_attention_mask)[0]
        return hidden_states

class ServerModel(nn.Module):
    # --- [修改] 增加 config 参数 ---
    def __init__(self, encoder_layers, classifier, attention_mask_getter, num_classes: int, config=None):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.classifier = classifier
        self.get_extended_attention_mask = attention_mask_getter
        self.num_classes = num_classes
        
        # --- [修改] 初始化损失函数 ---
        if config and hasattr(config, 'use_focal_loss') and config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, hidden_states, attention_mask, device, labels=None):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1], device)
        for layer in self.encoder_layers: hidden_states = layer(hidden_states, extended_attention_mask)[0]
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            # --- [修改] 使用初始化好的 loss_fct ---
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            
        return logits, loss

# --- 通信量计算工具 ---
def get_tensor_size_mb(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor): return 0
    return tensor.numel() * tensor.element_size() / (1024 * 1024)

# ============================== 4. 训练器类 (SFLTrainer) ==============================
class SFLTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config.seed)
        
        try:
            base_model_full = RobertaForSequenceClassification.from_pretrained(
                config.model_name, 
                num_labels=config.num_classes, 
                ignore_mismatched_sizes=True
            )
        except OSError:
            logging.error(f"无法加载模型: {config.model_name}")
            raise

        peft_config_full = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=config.lora_r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout, 
            target_modules=config.target_modules, 
            modules_to_save=["classifier"], 
            bias="none"
        )
        self.net_glob_full = get_peft_model(base_model_full, peft_config_full).to(self.device)
        self.history = []
        
        self.client_resource_profiles = {
            'high': {'split_layer': 8, 'weight_factor': 1.2},
            'medium': {'split_layer': 6, 'weight_factor': 1.0},
            'low': {'split_layer': 4, 'weight_factor': 0.8}
        }
        self.client_profiles = [random.choice(list(self.client_resource_profiles.keys())) for _ in range(config.num_clients)]
        self.client_lora_weights = [None] * config.num_clients 
        self.data_cluster_labels = [-1] * config.num_clients
        
        loss_type = "Focal Loss" if config.use_focal_loss else "Cross Entropy"
        logging.info(f"Initialized Binary SFL with {loss_type}. Config: {asdict(config)}")

    def set_seed(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def aggregate(self, w_locals, client_weights=None):
        if not w_locals: return
        if client_weights is None or self.config.aggregation_method == 'FedAvg':
            client_weights = [1.0 / len(w_locals)] * len(w_locals)
        
        w_avg = copy.deepcopy(w_locals[0])
        for k in w_avg.keys():
            if w_avg[k].dtype.is_floating_point: w_avg[k] *= client_weights[0]

        for i in range(1, len(w_locals)):
            for k in w_avg.keys():
                if w_avg[k].dtype.is_floating_point: w_avg[k] += w_locals[i][k] * client_weights[i]
        self.net_glob_full.load_state_dict(w_avg)

    def train_client(self, client_idx, local_net, server_model_for_client, train_loader):
        optimizer = AdamW(list(local_net.parameters()) + list(server_model_for_client.parameters()), lr=self.config.lr)
        total_loss, num_batches, comm_stats = 0, 0, {'uplink_sfl': 0, 'downlink_sfl': 0}
        local_net.train(); server_model_for_client.train()
        
        for epoch in range(self.config.local_epochs):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                
                # Client Forward
                fx = local_net(batch['input_ids'], batch['attention_mask'], self.device)
                comm_stats['uplink_sfl'] += get_tensor_size_mb(fx)
                
                # Server Forward
                client_fx = fx.clone().detach().requires_grad_(True)
                _, loss = server_model_for_client(client_fx, batch['attention_mask'], self.device, batch['labels'])
                
                loss.backward()
                
                # Gradient Return (Downlink)
                if client_fx.grad is not None:
                    comm_stats['downlink_sfl'] += get_tensor_size_mb(client_fx.grad)
                    fx.backward(client_fx.grad)
                
                optimizer.step()
                total_loss += loss.item(); num_batches += 1
                
        return (total_loss / num_batches if num_batches > 0 else 0), comm_stats
    
    def evaluate(self, test_loader, split_layer_for_eval=6):
        self.net_glob_full.eval()
        base_model = self.net_glob_full.base_model.model.roberta
        
        client_net = ClientModel(base_model.embeddings, nn.ModuleList(base_model.encoder.layer[:split_layer_for_eval]), self.net_glob_full.get_extended_attention_mask).to(self.device).eval()
        
        # --- [修改] 传入 config ---
        server_net = ServerModel(
            nn.ModuleList(base_model.encoder.layer[split_layer_for_eval:]), 
            self.net_glob_full.classifier, 
            self.net_glob_full.get_extended_attention_mask, 
            self.config.num_classes,
            config=self.config
        ).to(self.device).eval()
        
        all_preds, all_labels, total_eval_loss = [], [], 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                fx = client_net(batch['input_ids'], batch['attention_mask'], self.device)
                logits, loss = server_net(fx, batch['attention_mask'], self.device, batch['labels'])
                if loss is not None: total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(batch['labels'].cpu().numpy())
        
        target_names = ['Normal', 'Attack']
        report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        report_str = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        logging.info(f"\n分类评估报告 (Binary - FocalLoss={self.config.use_focal_loss}):\n{report_str}")
        return {'eval_loss': total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0, 'mcc': mcc, 'report': report_dict }

    def _extract_lora_weights_vector(self, model_state_dict):
        lora_params = {k: v for k, v in model_state_dict.items() if ('lora_A' in k or 'lora_B' in k or 'classifier' in k) and 'original_module' not in k}
        if not lora_params: return None
        return torch.cat([lora_params[k].view(-1) for k in sorted(lora_params.keys())]).cpu().numpy()

    def perform_double_clustering(self):
        client_features, available_clients = [], []
        for i, weight in enumerate(self.client_lora_weights):
            if weight is not None: client_features.append(weight); available_clients.append(i)
        
        if len(available_clients) < self.config.num_data_clusters:
            self.data_cluster_labels = [0] * self.config.num_clients
            return
        
        kmeans = KMeans(n_clusters=self.config.num_data_clusters, random_state=self.config.seed, n_init='auto').fit(np.array(client_features))
        for i, client_idx in enumerate(available_clients): self.data_cluster_labels[client_idx] = kmeans.labels_[i]

    def run(self, dataset_train, dataset_test):
        dict_users_indices = np.array_split(range(len(dataset_train)), self.config.num_clients)
        train_loaders = [DataLoader(DatasetSplit(dataset_train, list(idxs)), self.config.batch_size, shuffle=True) for idxs in dict_users_indices]
        test_loader_global = DataLoader(dataset_test, self.config.batch_size, shuffle=False)
        
        logging.info(f"开始二分类 SFL 训练, 总轮数: {self.config.rounds}, 损失函数: {'FocalLoss' if self.config.use_focal_loss else 'CrossEntropy'}")
        self.data_cluster_labels = [0] * self.config.num_clients
        
        for round_num in range(1, self.config.rounds + 1):
            round_start_time = time.time()
            if round_num > 1 and (round_num % self.config.clustering_interval == 0): self.perform_double_clustering()
            
            selected_clients = sorted(random.sample(range(self.config.num_clients), self.config.clients_per_round))
            
            w_locals, local_losses = [], []
            round_comm_stats = {'total_uplink_mb': 0, 'total_downlink_mb': 0}

            for idx in selected_clients:
                split_layer = self.client_resource_profiles[self.client_profiles[idx]]['split_layer']
                local_model = copy.deepcopy(self.net_glob_full)
                roberta = local_model.base_model.model.roberta
                
                local_net = ClientModel(roberta.embeddings, nn.ModuleList(roberta.encoder.layer[:split_layer]), local_model.get_extended_attention_mask).to(self.device)
                
                # --- [修改] 传入 config ---
                server_net = ServerModel(
                    nn.ModuleList(roberta.encoder.layer[split_layer:]), 
                    local_model.classifier, 
                    local_model.get_extended_attention_mask, 
                    self.config.num_classes,
                    config=self.config
                ).to(self.device)
                
                loss, comms = self.train_client(idx, local_net, server_net, train_loaders[idx])
                
                round_comm_stats['total_uplink_mb'] += comms['uplink_sfl']
                round_comm_stats['total_downlink_mb'] += comms['downlink_sfl']
                local_losses.append(loss); w_locals.append(local_model.state_dict())
                self.client_lora_weights[idx] = self._extract_lora_weights_vector(local_model.state_dict())

            self.aggregate(w_locals)
            
            metrics = self.evaluate(test_loader_global)
            round_duration = time.time() - round_start_time
            
            self.history.append({'round': round_num, 'avg_train_loss': np.mean(local_losses), 'round_duration_sec': round_duration, **round_comm_stats, **metrics})
            logging.info(f"Round {round_num} | Acc: {metrics['report']['accuracy']:.4f} | MCC: {metrics['mcc']:.4f}")

    def save_results(self, output_dir="results_binary_sfl"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.history: return

        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for k, v in asdict(self.config).items(): writer.writerow([f'# {k}', v])
            writer.writerow([])
            
            header = ['round', 'avg_train_loss', 'eval_loss', 'round_duration_sec', 'accuracy', 'mcc', 
                      'Normal_precision', 'Normal_recall', 'Normal_f1', 
                      'Attack_precision', 'Attack_recall', 'Attack_f1']
            writer.writerow(header)

            for record in self.history:
                report = record['report']
                row = [
                    record.get('round'), record.get('avg_train_loss'), record.get('eval_loss'), record.get('round_duration_sec'),
                    report.get('accuracy'), record.get('mcc'),
                    report.get('Normal', {}).get('precision'), report.get('Normal', {}).get('recall'), report.get('Normal', {}).get('f1-score'),
                    report.get('Attack', {}).get('precision'), report.get('Attack', {}).get('recall'), report.get('Attack', {}).get('f1-score')
                ]
                writer.writerow(row)
        logging.info(f"结果已保存: {os.path.join(output_dir, filename)}")

# ============================== 5. 主函数 ==============================
def main():
    config = ExperimentConfig()
    
    # 请确保以下路径指向您生成的二分类 jsonl 文件
    train_data_path = 'processed_data_binary/train_data.jsonl'
    test_data_path = 'processed_data_binary/test_data.jsonl'

    if not all(os.path.exists(f) for f in [train_data_path, test_data_path]):
        logging.error(f"错误：找不到数据文件。请先运行数据处理脚本生成 'processed_data_binary' 目录。")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except OSError:
        logging.error(f"错误：无法从本地路径加载分词器: {config.model_name}")
        return

    logging.info("加载二分类数据集...")
    dataset_train = FT_Dataset(train_data_path, config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset(test_data_path, config.batch_size, config.max_seq_length, tokenizer)
        
    trainer = SFLTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()
    logging.info("\n二分类 SFL 训练完成!")

if __name__ == "__main__":
    main()
