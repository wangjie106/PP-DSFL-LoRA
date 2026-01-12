import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef
import logging
import time
import os
import csv
from dataclasses import dataclass, field
from typing import List

from data_utils import FT_Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

@dataclass
class CentralizedConfig:
    exp_name: str = "centralized_baseline_smote"
    epochs: int = 20
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

class CentralizedTrainer:
    def __init__(self, config: CentralizedConfig):
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
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        self.history = []
        logging.info(f"Using device: {self.device}")

    def set_seed(self, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, train_loader, test_loader):
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = total_loss / len(train_loader)
            metrics = self.evaluate(test_loader)
            duration = time.time() - start_time
            
            self.history.append({
                'epoch': epoch,
                'avg_train_loss': avg_train_loss,
                'eval_loss': metrics['eval_loss'],
                'duration': duration,
                **metrics
            })
            logging.info(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Acc: {metrics['report']['accuracy']:.4f} | MCC: {metrics['mcc']:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels']
                )
                if outputs.loss is not None:
                    total_eval_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return {
            'eval_loss': total_eval_loss / len(test_loader) if len(test_loader) > 0 else 0,
            'mcc': mcc,
            'report': report
        }

    def save_results(self, output_dir="results_centralized"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.config.exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_train_loss', 'eval_loss', 'accuracy', 'mcc', 'weighted_f1', 'duration'])
            for rec in self.history:
                writer.writerow([
                    rec['epoch'], rec['avg_train_loss'], rec['eval_loss'],
                    rec['report']['accuracy'], rec['mcc'], 
                    rec['report']['weighted avg']['f1-score'], rec['duration']
                ])
        logging.info(f"Results saved to {filename}")

def main():
    config = CentralizedConfig()
    train_data_path = 'processed_data_SMOTE/train_data.jsonl'
    test_data_path = 'processed_data_SMOTE/test_data.jsonl'
    
    if not all(os.path.exists(f) for f in [train_data_path, test_data_path]):
        logging.error("Data files not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset_train = FT_Dataset(train_data_path, config.batch_size, config.max_seq_length, tokenizer)
    dataset_test = FT_Dataset(test_data_path, config.batch_size, config.max_seq_length, tokenizer)
    
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)

    trainer = CentralizedTrainer(config)
    trainer.train(train_loader, test_loader)
    trainer.save_results()

if __name__ == "__main__":
    main()
