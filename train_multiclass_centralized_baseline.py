import torch
import torch.nn as nn
import torch.nn.functional as F
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
import logging
import time
import csv
import os
from dataclasses import dataclass, field, asdict
from typing import List

from data_utils import FT_Dataset

# ============================== Focal Loss ==============================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================== Config ==============================
@dataclass
class ExperimentConfig:
    exp_name: str = "multiclass_centralized_baseline_focal"
    epochs: int = 20  # Replaces'rounds'
    batch_size: int = 16
    max_seq_length: int = 128
    num_classes: int = 10 
    
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    lr: float = 2e-5
    model_name: str = "./roberta-base-local"
    seed: int = 42
    
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    focal_alpha: List[float] = field(default_factory=lambda: [
        1.0, 5.0, 5.0, 1.5, 1.0, 1.5, 1.0, 1.5, 8.0, 10.0
    ])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ============================== Trainer ==============================
class CentralizedTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_seed(config.seed)
        
        try:
            base_model = RobertaForSequenceClassification.from_pretrained(
                config.model_name, 
                num_labels=config.num_classes,
                ignore_mismatched_sizes=True
            )
        except OSError:
            logging.error(f"Error: Unable to load model from '{config.model_name}'.")
            raise

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=config.lora_r, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout, 
            target_modules=config.target_modules, 
            modules_to_save=["classifier"], 
            bias="none"
        )
        self.model = get_peft_model(base_model, peft_config).to(self.device)
        
        if config.use_focal_loss:
            self.loss_fct = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
        else:
            weights = torch.tensor(config.focal_alpha) if config.focal_alpha else None
            self.loss_fct = nn.CrossEntropyLoss(weight=weights)
            
        self.history = []
        logging.info(f"Initialized Centralized Trainer. Config: {asdict(config)}")

    def set_seed(self, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def train_epoch(self, train_loader):
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        total_loss, num_batches = 0, 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            
            loss = self.loss_fct(logits.view(-1, self.config.num_classes), batch['labels'].view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels, total_eval_loss = [], [], 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits
                
                loss = self.loss_fct(logits.view(-1, self.config.num_classes), batch['labels'].view(-1))
                total_eval_loss += loss.item()
                
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        report_dict = classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0, output_dict=True)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        logging.info(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(self.config.num_classes)], zero_division=0)}")
        
        return {
            'eval_loss': total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0,
            'mcc': mcc,
            'report': report_dict 
        }

    def run(self, dataset_train, dataset_test):
        train_loader = DataLoader(dataset_train, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=self.config.batch_size, shuffle=False)
        
        logging.info(f"Start Centralized Training for {self.config.epochs} epochs...")
        
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            logging.info(f"\n--- Epoch {epoch}/{self.config.epochs} ---")
            
            avg_train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(test_loader)
            duration = time.time() - start_time
            
            self.history.append({
                'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'epoch_duration_sec': duration,
                **metrics
            })
            
            logging.info(f"Epoch {epoch} Summary: Loss: {avg_train_loss:.4f}, Acc: {metrics['report']['accuracy']:.4f}, MCC: {metrics['mcc']:.4f}, Time: {duration:.2f}s")

    def save_results(self, output_dir="results_centralized"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        if not self.history: return

        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for k, v in asdict(self.config).items(): writer.writerow([f'# {k}', v])
            writer.writerow([])
            
            header = ['epoch', 'avg_train_loss', 'eval_loss', 'epoch_duration_sec', 'accuracy', 'mcc']
            writer.writerow(header)
            
            for record in self.history:
                report = record['report']
                row = [
                    record['epoch'], record['avg_train_loss'], record['eval_loss'], record['epoch_duration_sec'],
                    report['accuracy'], record['mcc']
                ]
                writer.writerow(row)
        logging.info(f"Results saved to: {os.path.join(output_dir, filename)}")

def main():
    config = ExperimentConfig()
    
    if not all(os.path.exists(f) for f in ['processed_data/train_data.jsonl', 'processed_data/test_data.jsonl']):
        logging.error("Error: Data files not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset_train = FT_Dataset('processed_data/train_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset('processed_data/test_data.jsonl', config.batch_size, config.max_seq_length, tokenizer)
        
    trainer = CentralizedTrainer(config)
    trainer.run(dataset_train, dataset_test)
    trainer.save_results()
    logging.info("Centralized Training Completed!")

if __name__ == "__main__":
    main()
