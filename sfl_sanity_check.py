import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
import csv
import random
from collections import OrderedDict, Counter
import copy
import os

from data_utils import FT_Dataset, get_tokenizer

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

MODE = 'QUICK'

if MODE == 'QUICK':
    logging.warning("Running in SFL QUICK mode")
    ROUNDS, NUM_CLIENTS, CLIENTS_PER_ROUND, LOCAL_EPOCHS, DEBUG_DATA_SIZE, SPLIT_LAYER = 10, 4, 2, 5, 2000, 4
else:
    logging.info("Running in SFL NORMAL mode")
    ROUNDS, NUM_CLIENTS, CLIENTS_PER_ROUND, LOCAL_EPOCHS, DEBUG_DATA_SIZE, SPLIT_LAYER = 50, 20, 5, 5, None, 4

MODEL_NAME = './roberta-base-local'
MODEL_NAME_ON_HUB = 'roberta-base'

LR, BATCH_SIZE, MAX_SEQ_LENGTH = 1e-4, 16, 128
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1
RESULTS_FILENAME = f"SFL_{MODE.lower()}_R{ROUNDS}_C{NUM_CLIENTS}_{time.strftime('%Y%m%d_%H%M%S')}.csv"

class ClientModelSFL(nn.Module):
    def __init__(self, full_model, split_layer):
        super().__init__()
        base_model = full_model.base_model.model if hasattr(full_model, 'base_model') else full_model
        self.embeddings = base_model.roberta.embeddings
        self.encoder_layers = base_model.roberta.encoder.layer[:split_layer]
    
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None: 
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states = self.embeddings(input_ids=input_ids)
        for layer in self.encoder_layers: 
            hidden_states = layer(hidden_states, attention_mask=extended_attention_mask)[0]
        return hidden_states, extended_attention_mask

class ServerModelSFL(nn.Module):
    def __init__(self, full_model, split_layer):
        super().__init__()
        base_model = full_model.base_model.model if hasattr(full_model, 'base_model') else full_model
        self.encoder_layers = base_model.roberta.encoder.layer[split_layer:]
        self.classifier = base_model.classifier
    
    def forward(self, hidden_states, attention_mask, labels=None, class_weights=None):
        for layer in self.encoder_layers: 
            hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return logits, loss

def create_and_split_model(split_layer, model_path):
    logging.info(f"Loading model from path '{model_path}'...")
    logging.info(f"Split layer set to: {split_layer}")
    
    full_model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=LORA_R, 
        lora_alpha=LORA_ALPHA, 
        lora_dropout=LORA_DROPOUT, 
        target_modules=["query", "key", "value"], 
        modules_to_save=["classifier"]
    )
    full_model_peft = get_peft_model(full_model, peft_config)
    
    logging.info("Full model trainable parameters after PEFT application:")
    full_model_peft.print_trainable_parameters()
    
    client_model = ClientModelSFL(full_model_peft, split_layer)
    server_model = ServerModelSFL(full_model_peft, split_layer)
    
    logging.info("Model splitting completed.")
    return client_model.to(device), server_model.to(device)

def get_trainable_state_dict(model: nn.Module) -> dict:
    return {k: v.clone() for k, v in model.state_dict().items() if v.requires_grad}

def set_trainable_state_dict(model: nn.Module, state_dict: dict):
    model.load_state_dict(state_dict, strict=False)

def FedAvg(w_list):
    if not w_list: 
        return None
    aggregated_weights = OrderedDict()
    for key in w_list[0].keys():
        aggregated_weights[key] = torch.stack([w[key] for w in w_list]).mean(dim=0)
    return aggregated_weights

def client_sfl_train(client_id, client_model, server_model, train_loader, current_lr, 
                     class_weights=None, is_first_client_in_round=False):
    client_model.train()
    server_model.train()
    
    trainable_client_params = [p for p in client_model.parameters() if p.requires_grad]
    trainable_server_params = [p for p in server_model.parameters() if p.requires_grad]
    
    if is_first_client_in_round:
        print("\n" + "="*80)
        print(f"DEBUG: Pre-training check for the first client (ID: {client_id})")
        print(f"    Number of trainable parameter groups found in client model: {len(trainable_client_params)}")
        print(f"    Number of trainable parameter groups found in server model: {len(trainable_server_params)}")
        
        client_param_count = sum(p.numel() for p in trainable_client_params)
        server_param_count = sum(p.numel() for p in trainable_server_params)
        
        print(f"    Total trainable parameters in client model: {client_param_count:,}")
        print(f"    Total trainable parameters in server model: {server_param_count:,}")

        if not trainable_client_params:
            print("    WARNING: Client optimizer has no parameters to optimize")
        else:
            print("    OK: Client optimizer has parameters to optimize.")

        if not trainable_server_params:
            print("    WARNING: Server optimizer has no parameters to optimize")
        else:
            print("    OK: Server optimizer has parameters to optimize.")
        
        if class_weights is not None:
            print(f"    Using class weights: {class_weights.cpu().numpy()}")
        
        print("="*80 + "\n")

    if not trainable_client_params and not trainable_server_params:
        logging.error(f"Client {client_id}: Fatal error - No trainable parameters found in both client and server. Skipping training.")
        return get_trainable_state_dict(client_model), 0.0

    optimizer_client = AdamW(trainable_client_params, lr=current_lr) if trainable_client_params else None
    optimizer_server = AdamW(trainable_server_params, lr=current_lr) if trainable_server_params else None

    total_loss, num_batches = 0, 0
    for epoch in range(LOCAL_EPOCHS):
        for batch in train_loader:
            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            labels = batch_on_device['labels']
            
            if optimizer_client: optimizer_client.zero_grad()
            if optimizer_server: optimizer_server.zero_grad()

            smashed_data, extended_attention_mask = client_model(
                input_ids=batch_on_device['input_ids'], 
                attention_mask=batch_on_device['attention_mask']
            )
            smashed_data_server = smashed_data.detach().requires_grad_(True)
            
            logits, loss = server_model(smashed_data_server, extended_attention_mask, labels, 
                                       class_weights=class_weights)
            
            if loss is None: 
                continue
            
            loss.backward()
            
            if smashed_data_server.grad is not None:
                smashed_data.backward(smashed_data_server.grad)
            
            if optimizer_client: optimizer_client.step()
            if optimizer_server: optimizer_server.step()
            
            total_loss += loss.item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"  Client {client_id} (LR={current_lr:.2e}) training completed, average loss: {avg_loss:.4f}")

    return get_trainable_state_dict(client_model), avg_loss

def evaluate_sfl(client_model, server_model, dataloader):
    client_model.eval()
    server_model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SFL model", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            smashed_data, extended_attention_mask = client_model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            logits, _ = server_model(smashed_data, extended_attention_mask)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_labels)
    
    logging.info(f"  Prediction distribution: Class 0={pred_counts.get(0, 0):>6}, Class 1={pred_counts.get(1, 0):>6}")
    logging.info(f"  Ground truth distribution: Class 0={label_counts[0]:>6}, Class 1={label_counts[1]:>6}")
    
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    
    correct_0 = sum((all_preds_np == 0) & (all_labels_np == 0))
    correct_1 = sum((all_preds_np == 1) & (all_labels_np == 1))
    total_0 = label_counts[0]
    total_1 = label_counts[1]
    
    acc_0 = correct_0 / total_0 * 100 if total_0 > 0 else 0
    acc_1 = correct_1 / total_1 * 100 if total_1 > 0 else 0
    
    logging.info(f"  Class 0 accuracy: {correct_0:>6}/{total_0:>6} = {acc_0:>5.2f}%")
    logging.info(f"  Class 1 accuracy: {correct_1:>6}/{total_1:>6} = {acc_1:>5.2f}%")
    
    metrics = {'accuracy': accuracy_score(all_labels, all_preds)}
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1})
    return metrics

def split_data_for_clients(train_dataset, num_clients):
    client_datasets, all_indices = [], list(range(len(train_dataset)))
    random.shuffle(all_indices)
    for i in range(num_clients):
        subset_indices = all_indices[i::num_clients]
        client_datasets.append(Subset(train_dataset, subset_indices))
    return client_datasets

def save_results_to_csv(all_results, filename, config):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['# Configuration'])
            for k, v in config.items():
                writer.writerow([f'# {k}', v])
            writer.writerow([])
            
            fieldnames = ['round', 'loss', 'accuracy', 'f1', 'precision', 'recall']
            writer.writerow(fieldnames)
            
            for result in all_results:
                writer.writerow([
                    result['round'], 
                    f"{result['loss']:.4f}", 
                    f"{result['accuracy']:.4f}", 
                    f"{result['f1']:.4f}", 
                    f"{result['precision']:.4f}", 
                    f"{result['recall']:.4f}"
                ])
        logging.info(f"SFL evaluation results successfully saved to file: {filename}")
    except Exception as e:
        logging.error(f"Failed to save SFL result file: {e}")

def print_trainable_parameters_manually(model: nn.Module):
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
        return

    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_param if all_param > 0 else 0
    logging.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {percentage:.4f}%")

def verify_local_model(model_path):
    print("\n" + "="*70)
    print("Verifying local model...")
    
    if not os.path.exists(model_path):
        logging.error(f"Local model path does not exist: {model_path}")
        logging.error("Please run download_roberta.py to download the model first")
        return False
    
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'vocab.json', 'merges.txt']
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logging.info(f"  {file:<25} ({size_mb:>8.2f} MB)")
    
    if missing_files:
        logging.error(f"Missing key files: {missing_files}")
        return False
    
    logging.info(f"Local model verification passed: {model_path}")
    print("="*70 + "\n")
    return True

def main():
    print("="*70)
    print("Split Federated Learning (SFL): Fine-tuning RoBERTa with LoRA")
    print(f"Current mode: {MODE}")
    print(f"Improvements: Enhanced training intensity + Class weights + Detailed evaluation")
    print("="*70 + "\n")
    
    if not verify_local_model(MODEL_NAME):
        logging.error("Model verification failed, program exiting.")
        return
    
    logging.info("Step 0: Checking data files...")
    
    train_file = 'processed_data/train_data.jsonl'
    test_file = 'processed_data/test_data.jsonl'
    
    data_exists = os.path.exists(train_file) and os.path.exists(test_file)
    
    if not data_exists:
        logging.warning("Processed data files not found, attempting automatic processing...")
        
        try:
            import UNSW_NB15_processed_llm as data_processor
            
            if MODE == 'QUICK':
                debug_rows = 10000
                logging.info(f"QUICK mode: Using {debug_rows} rows of data for quick testing")
            else:
                debug_rows = None
                logging.info("NORMAL mode: Using full dataset")
            
            success = data_processor.check_and_prepare_data(
                data_path='data/UNSW-NB15.csv',
                debug_rows=debug_rows,
                force_reprocess=False
            )
            
            if not success:
                logging.error("Data processing failed, program exiting.")
                return
                
        except ImportError:
            logging.error("Unable to import data processing module'UNSW_NB15_processed_llm.py'")
            return
        except Exception as e:
            logging.error(f"Error occurred during data processing: {e}")
            return
    else:
        logging.info(f"Found processed data files")
    
    training_config = {
        "Mode": MODE,
        "Framework": "Split Federated Learning",
        "Model": MODEL_NAME_ON_HUB,
        "Model_Path": MODEL_NAME,
        "Split_Layer": SPLIT_LAYER,
        "LoRA_Rank_(r)": LORA_R,
        "LoRA_Alpha_(alpha)": LORA_ALPHA,
        "Communication_Rounds": ROUNDS,
        "Clients_per_Round": CLIENTS_PER_ROUND,
        "Total_Clients": NUM_CLIENTS,
        "Local_Epochs": LOCAL_EPOCHS,
        "Learning_Rate_(LR)": LR
    }

    logging.info("Step 1: Loading data and splitting...")
    
    tokenizer = get_tokenizer(MODEL_NAME)
    
    full_train_dataset = FT_Dataset(train_file, BATCH_SIZE, MAX_SEQ_LENGTH, tokenizer)
    test_dataset = FT_Dataset(test_file, BATCH_SIZE, MAX_SEQ_LENGTH, tokenizer)
    
    logging.info("Calculating class weights...")
    all_labels = []
    for i in range(len(full_train_dataset)):
        all_labels.append(full_train_dataset[i]['labels'].item())
    
    label_counts = Counter(all_labels)
    logging.info(f"Training set label distribution: Class 0={label_counts[0]}, Class 1={label_counts[1]}")
    
    total_samples = len(all_labels)
    class_weights = torch.tensor([
        total_samples / (2 * label_counts[0]),
        total_samples / (2 * label_counts[1])
    ], dtype=torch.float32).to(device)
    
    logging.info(f"Class weights: Class 0={class_weights[0]:.3f}, Class 1={class_weights[1]:.3f}")
    logging.info("(The larger the weight, the more important the class is in loss calculation)")
    
    if DEBUG_DATA_SIZE is not None and DEBUG_DATA_SIZE < len(full_train_dataset):
        indices = torch.randperm(len(full_train_dataset))[:DEBUG_DATA_SIZE]
        train_subset = Subset(full_train_dataset, indices.tolist())
        client_datasets = split_data_for_clients(train_subset, NUM_CLIENTS)
    else:
        client_datasets = split_data_for_clients(full_train_dataset, NUM_CLIENTS)
    
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logging.info("Step 2: Initializing global model and global scheduler...")
    
    global_client_model, global_server_model = create_and_split_model(SPLIT_LAYER, MODEL_NAME)
    
    logging.info("--- (Manual Check) Client trainable parameters ---")
    print_trainable_parameters_manually(global_client_model)
    logging.info("--- (Manual Check) Server trainable parameters ---")
    print_trainable_parameters_manually(global_server_model)

    dummy_optimizer = AdamW([torch.zeros(1)], lr=LR)
    global_scheduler = get_linear_schedule_with_warmup(
        dummy_optimizer, 
        num_warmup_steps=0,
        num_training_steps=ROUNDS
    )

    logging.info("Step 3: Starting SFL training...")
    all_round_results = []
    
    for round_num in range(1, ROUNDS + 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"Communication Round {round_num}/{ROUNDS}")
        logging.info("="*70)
        
        selected_client_ids = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        logging.info(f"Participating clients this round: {selected_client_ids}")
        
        current_round_lr = global_scheduler.get_last_lr()[0]
        logging.info(f"Current learning rate: {current_round_lr:.2e}")
        
        round_server_model = copy.deepcopy(global_server_model)
        
        local_client_weights, local_losses = [], []
        
        for i, client_id in enumerate(selected_client_ids):
            local_client_model = copy.deepcopy(global_client_model)
            
            is_first = (i == 0 and round_num == 1)
            
            client_w, loss = client_sfl_train(
                client_id, 
                local_client_model, 
                round_server_model, 
                client_loaders[client_id], 
                current_round_lr,
                class_weights=class_weights,
                is_first_client_in_round=is_first
            )
            local_client_weights.append(client_w)
            local_losses.append(loss)
            
        global_client_weights = FedAvg(local_client_weights)
        if global_client_weights: 
            set_trainable_state_dict(global_client_model, global_client_weights)
        
        final_round_server_weights = get_trainable_state_dict(round_server_model)
        set_trainable_state_dict(global_server_model, final_round_server_weights)
        
        logging.info(f"Server aggregation completed, global model updated.")
        
        logging.info("Evaluating...")
        metrics = evaluate_sfl(global_client_model, global_server_model, test_loader)
        avg_round_loss = np.mean(local_losses)
        round_result = {'round': round_num, 'loss': avg_round_loss, **metrics}
        all_round_results.append(round_result)
        
        global_scheduler.step()
        
        print()
        logging.info(f"{'='*70}")
        logging.info(f"Round {round_num} Summary")
        logging.info("="*70)
        logging.info(f"  Average training loss: {avg_round_loss:.4f}")
        logging.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Test F1 Score: {metrics['f1']:.4f}")
        logging.info(f"  Test Precision: {metrics['precision']:.4f}")
        logging.info(f"  Test Recall: {metrics['recall']:.4f}")
        logging.info("="*70)
        print()

    logging.info("Step 4: Training completed, saving results...")
    save_results_to_csv(all_round_results, RESULTS_FILENAME, training_config)
    
    print("\n" + "="*70)
    logging.info("SFL Training Completed!")
    logging.info(f"Results saved to: {RESULTS_FILENAME}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
