from sklearn.metrics import mean_absolute_error as mae, r2_score, mean_absolute_percentage_error as mape, accuracy_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_regressor(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=50):
    best_val_loss = np.inf
    best_mape = None
    best_r2 = None
    best_model_weights = None
    epochs_no_improve = 0
    torch.set_num_threads(1)  
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_outputs = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch.to(DEVICE)
            y_batch.to(DEVICE)
            outputs = model(X_batch).squeeze(1)
            loss = criterion(outputs, y_batch.squeeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_labels.extend(y_batch.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

        train_mape = mape(all_labels, all_outputs)
        train_mae = mae(all_labels, all_outputs)
        train_r2 = r2_score(all_labels, all_outputs)

        # Валидация
        val_mape, val_r2, val_mae, naive_mape, naive_r2, naive_mae, val_loss = evaluate_regressor(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mape = val_mape
            best_r2 = val_r2
            best_model_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping! No improvement for {patience} epochs.")
                break
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
            f"Train MAPE: {train_mape:.4f}, Val MAPE: {val_mape:.4f},"
            f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f},"
            f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
            f"naive mape: {naive_mape:.4f}, naive R2: {naive_r2:.4f}, naive MAE: {naive_mae:.4f}")
    return best_model_weights, {"loss": best_val_loss, "MAPE": best_mape, "R2": best_r2}

def evaluate_regressor(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).squeeze(1)
            all_outputs.extend(outputs.cpu().numpy())
            loss = criterion(outputs, y_batch.squeeze(1).float())
            total_loss += loss.item()
            all_labels.extend(y_batch.cpu().numpy())
    
    mape_score = mape(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)
    mae_score = mae(all_labels, all_outputs)
    return mape_score, r2, mae_score, mape(all_labels[1:], all_labels[:-1]), r2_score(all_labels[1:], all_labels[:-1]), mae(all_labels[1:], all_labels[:-1]), total_loss/len(loader)


def train_classifier(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=50):
    best_val_loss = np.inf
    best_roc_auc = None
    best_accuracy = None
    best_model_weights = None
    epochs_no_improve = 0
    torch.set_num_threads(1)  
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(1)
            X_batch.to(DEVICE)
            y_batch.to(DEVICE)
            loss = criterion(outputs, y_batch.squeeze(1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (outputs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds)


        # Валидация
        val_acc, val_f1, roc_auc, random, val_loss = evaluate_classifier(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_roc_auc = roc_auc
            best_accuracy = val_acc
            best_model_weights = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping! No improvement for {patience} epochs.")
                break
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"val ROC-AUC: {roc_auc:.4f}, Val F1: {val_f1:.4f}, random prediction {random:.4f}")
        
    return best_model_weights, {"loss": best_val_loss, "roc_auc": best_roc_auc, "accuracy": best_accuracy}

def evaluate_classifier(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).squeeze(1)
            preds = (outputs > 0.5).int()
            all_outputs.extend(outputs.cpu().numpy())
            loss = criterion(outputs, y_batch.squeeze(1).float())
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_outputs)
    return acc, f1, roc_auc, np.mean(all_labels), total_loss/len(loader)

# Обучение