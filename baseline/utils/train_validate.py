from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import mlflow
import torch
import logging


def train(
    model, train_loader, criterion, optimizer, device, epoch, class_names=None
):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Update tqdm postfix
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    accuracy = (
        (torch.tensor(all_preds) == torch.tensor(all_labels))
        .float()
        .mean()
        .item()
    )
    f1_weighted = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    f1_macro = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # Log metrics to MLflow
    mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
    mlflow.log_metric("train_accuracy", accuracy, step=epoch + 1)
    mlflow.log_metric("train_f1_weighted", f1_weighted, step=epoch + 1)
    mlflow.log_metric("train_f1_macro", f1_macro, step=epoch + 1)

    # Optional: log per-class F1
    # if class_names:
    #     report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    #     for cls in class_names:
    #         mlflow.log_metric(f"train_f1_{cls}", report[cls]['f1-score'], step=epoch+1)

    logging.info(
        f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1(weighted): {f1_weighted:.4f}, F1(macro): {f1_macro:.4f}"
    )


def validate(model, val_loader, criterion, device, epoch, class_names=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Sum loss over all samples (not batch average)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            loop.set_postfix(loss=loss.item())

    # Compute loss per sample
    avg_loss = total_loss / total_samples

    # Compute metrics for the whole dataset
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    f1_macro = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # Log metrics
    mlflow.log_metric("val_loss", avg_loss, step=epoch + 1)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch + 1)
    mlflow.log_metric("val_f1_weighted", f1_weighted, step=epoch + 1)
    mlflow.log_metric("val_f1_macro", f1_macro, step=epoch + 1)

    # Optional per-class logging
    # if class_names:
    #     from sklearn.metrics import classification_report
    #     report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    #     for cls in class_names:
    #         mlflow.log_metric(f"val_f1_{cls}", report[cls]['f1-score'], step=epoch+1)

    logging.info(
        f"Epoch {epoch + 1} - Val Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, F1(weighted): {f1_weighted:.4f}, F1(macro): {f1_macro:.4f}"
    )

    return f1_weighted
