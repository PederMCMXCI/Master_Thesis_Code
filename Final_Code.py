import os
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchvision import datasets, transforms, models
from pytorchcv.model_provider import get_model as ptcv_get_model
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


# Setze das Gerät auf CUDA, wenn verfügbar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Update train transforms to include RandomAffine and Cutout
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),

    # Wähle zufällig eine der folgenden Farb-/Kontrasttransformationen und wende sie gelegentlich an
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2
        ),
        transforms.RandomGrayscale(p=0.3)
    ], p=1.0),

    transforms.ToTensor(),

    # Wende eine zufällige zweite Transformation in beliebiger Reihenfolge an
    transforms.RandomOrder([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))], p=0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5)
    ]),


transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

valid_test_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset-Klasse zum Laden der Daten mit Pfadangaben und Labels
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Funktion zur Sammlung der Bildpfade und Labels
def get_image_paths_and_labels(directory, label):
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(root, file) for root, _, files in os.walk(directory)
                   for file in files if file.lower().endswith(valid_extensions)]
    labels = [label] * len(image_paths)
    return image_paths, labels

# Manuelle Angabe der Pfade für Trainings-, Validierungs- und Test-Daten
linden_train_dir = "D:\\Labeln\\binomialverteilung\\output\\train\\Linden"
keine_linden_train_dir = "D:\\Labeln\\binomialverteilung\\output\\train\\Keine Linden"
linden_val_dir = "D:\\Labeln\\binomialverteilung\\output\\val\\Linden"
keine_linden_val_dir = "D:\\Labeln\\binomialverteilung\\output\\val\\Keine Linden"
linden_test_dir = "D:\\Labeln\\binomialverteilung\\output\\test\\Linden"
keine_linden_test_dir = "D:\\Labeln\\binomialverteilung\\output\\test\\Keine Linden"

# Bildpfade und Labels sammeln für die Trainings-, Validierungs- und Test-Daten
train_image_paths, train_labels = [], []
for dir_path, label in [(linden_train_dir, 0), (keine_linden_train_dir, 1)]:
    paths, lbls = get_image_paths_and_labels(dir_path, label)
    train_image_paths.extend(paths)
    train_labels.extend(lbls)

val_image_paths, val_labels = [], []
for dir_path, label in [(linden_val_dir, 0), (keine_linden_val_dir, 1)]:
    paths, lbls = get_image_paths_and_labels(dir_path, label)
    val_image_paths.extend(paths)
    val_labels.extend(lbls)

test_image_paths, test_labels = [], []
for dir_path, label in [(linden_test_dir, 0), (keine_linden_test_dir, 1)]:
    paths, lbls = get_image_paths_and_labels(dir_path, label)
    test_image_paths.extend(paths)
    test_labels.extend(lbls)

# Ausgabe der Anzahl der Bilder vor der Balancierung
print("\nAnzahl der Bilder im ursprünglichen Trainingsdatensatz:")
print(f"Linden: {train_labels.count(0)}, Keine Linden: {train_labels.count(1)}")

print("\nAnzahl der Bilder im ursprünglichen Validierungsdatensatz:")
print(f"Linden: {val_labels.count(0)}, Keine Linden: {val_labels.count(1)}")

print("\nAnzahl der Bilder im ursprünglichen Test-Datensatz:")
print(f"Linden: {test_labels.count(0)}, Keine Linden: {test_labels.count(1)}")

# Funktion zum Balancieren des Datensatzes mit 20 % mehr "Keine Linden"-Bildern
def balance_dataset(image_paths, labels, class_label, extra_ratio=1.2):
    class_indices = [i for i, label in enumerate(labels) if label == class_label]
    other_indices = [i for i in range(len(labels)) if i not in class_indices]
    target_num_class = int(len(other_indices) * extra_ratio)

    if len(class_indices) > target_num_class:
        balanced_class_indices = random.sample(class_indices, target_num_class)
    else:
        balanced_class_indices = class_indices + random.choices(class_indices, k=target_num_class - len(class_indices))

    balanced_indices = balanced_class_indices + other_indices
    return [image_paths[i] for i in balanced_indices], [labels[i] for i in balanced_indices]

# Balancieren der Trainings-, Validierungs- und Test-Datensätze
train_image_paths, train_labels = balance_dataset(train_image_paths, train_labels, class_label=1, extra_ratio=1.2)
val_image_paths, val_labels = balance_dataset(val_image_paths, val_labels, class_label=1, extra_ratio=1.2)
test_image_paths, test_labels = balance_dataset(test_image_paths, test_labels, class_label=1, extra_ratio=1.2)

# Erstellen der Datasets
train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transform)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=valid_test_transform)
test_dataset = CustomDataset(test_image_paths, test_labels, transform=valid_test_transform)

# Sicherstellen, dass die Datasets nicht leer sind
assert len(train_dataset) > 0, "Trainingsdatensatz ist leer!"
assert len(val_dataset) > 0, "Validierungsdatensatz ist leer!"
assert len(test_dataset) > 0, "Test-Datensatz ist leer!"

# DataLoader für Training, Validierung und Test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ausgabe der Anzahl der Bilder nach der Balancierung
print("\nAnzahl der Bilder im balancierten Trainingsdatensatz:")
print(f"Linden: {train_labels.count(0)}, Keine Linden: {train_labels.count(1)}")

print("\nAnzahl der Bilder im balancierten Validierungsdatensatz:")
print(f"Linden: {val_labels.count(0)}, Keine Linden: {val_labels.count(1)}")

print("\nAnzahl der Bilder im balancierten Test-Datensatz:")
print(f"Linden: {test_labels.count(0)}, Keine Linden: {test_labels.count(1)}")


# Modell-Setup
class ModifiedResNet152(nn.Module):
    def __init__(self):
        super(ModifiedResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        # Nur eine Ausgabe für binäre Klassifikation
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# Modell initialisieren
model = ModifiedResNet152().to(device)


#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001) #weight_decay=0.0001)  # weight_decay für L2-Regularisierung
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Festlegen des zentralen Ausgabeordners
default_output_dir = r"D:\Labeln\binomialverteilung\output\metriken"

# Hilfsfunktion zum Speichern mit Zeitstempel
def save_file_with_timestamp(base_filename, extension, output_dir=default_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.{extension}"
    save_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    return save_path

# Funktion zur Speicherung der Trainingsparameter in einer Excel-Datei
def speichere_training_parameter(config, output_dir=default_output_dir, base_filename="training_parameters"):
    df = pd.DataFrame([config])
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)
    df.to_excel(save_path, index=False)
    print(f"Die Trainingsparameter wurden in '{save_path}' gespeichert.")

# Zentralisiertes Dictionary für alle Trainingsparameter
config = {
    "Modellarchitektur": "ResNet152",
    "Batch-Größe": 32,
    "Epochen": 30,
    "Lernrate": 0.001,
    # "Weight Decay": 0.0001,
    "Momentum": 0.9,
    "criterion": "nn.BCEWithLogitsLoss",
    "Optimizer": "SGD",
    "Scheduler": "CosineAnnealingLR",
    "Aktivierungsfunktion": "ReLU",
    "Train Loss": None,
    "Validation Loss": None,
    "Train Accuracy": None,
    "Validation Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1 Score": None,
    "Training Time (minutes)": None
}

def speichere_test_parameter(config, output_dir=default_output_dir, base_filename="test_parameters"):
    """
    Speichert die Testparameter in einer Excel-Datei im angegebenen Ordner.
    Fügt automatisch einen Zeitstempel zur Datei hinzu.
    """
    df = pd.DataFrame([config])
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)
    df.to_excel(save_path, index=False)
    print(f"Die Testparameter wurden in '{save_path}' gespeichert.")

test_config = {
    "Modellarchitektur": "ResNet152",
    "Batch-Größe": 32,
    "Epochen": 30,
    "Lernrate": 0.001,
    "Weight Decay": None,  # Standardmäßig leer, falls nicht verwendet
    "Momentum": 0.9,
    "criterion": "nn.BCEWithLogitsLoss",
    "Optimizer": "SGD",
    "Scheduler": "CosineAnnealingLR",
    "Aktivierungsfunktion": "ReLU",
    "Test Loss": None,
    "Test Accuracy": None,
    "Precision": None,
    "Recall": None,
    "F1 Score": None,
    "Test Duration (minutes)": None,

}

# Anzeige von Beispielbildern mit Vorhersagen und tatsächlichen Labels
def display_images_with_predictions(model, dataset, class_names, num_linden=10, num_keine_linden=10):
    linden_indices = [i for i, (_, label) in enumerate(dataset) if label == 0][:num_linden]
    keine_linden_indices = [i for i, (_, label) in enumerate(dataset) if label == 1][:num_keine_linden]

    linden_loader = DataLoader(Subset(dataset, linden_indices), batch_size=num_linden, shuffle=True)
    keine_linden_loader = DataLoader(Subset(dataset, keine_linden_indices), batch_size=num_keine_linden, shuffle=True)

    model.eval()
    plt.figure(figsize=(15, 15))
    count = 0
    mean, std = np.array([0.2, 0.2, 0.2]), np.array([0.8, 0.8, 0.8])

    with torch.no_grad():
        for images, labels in linden_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for j in range(images.size(0)):
                plt.subplot(5, 4, count + 1)
                image = images[j].cpu().permute(1, 2, 0).numpy()
                image = np.clip(std * image + mean, 0, 1)
                plt.imshow(image)
                color = 'green' if preds[j] == labels[j] else 'red'
                plt.title(f'Pred: {class_names[int(preds[j])]} | True: {class_names[labels[j]]}', color=color)
                plt.axis("off")
                count += 1
                if count >= num_linden + num_keine_linden:
                    break

        for images, labels in keine_linden_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for j in range(images.size(0)):
                plt.subplot(5, 4, count + 1)
                image = images[j].cpu().permute(1, 2, 0).numpy()
                image = np.clip(std * image + mean, 0, 1)
                plt.imshow(image)
                color = 'green' if preds[j] == labels[j] else 'red'
                plt.title(f'Pred: {class_names[int(preds[j])]} | True: {class_names[labels[j]]}', color=color)
                plt.axis("off")
                count += 1
                if count >= num_linden + num_keine_linden:
                    break

    plt.tight_layout(pad=3.0)
    plt.show()


# Funktion zur Speicherung der Confusion Matrix als Bild
def save_confusion_matrix(labels, preds, class_names, output_dir=default_output_dir, base_filename="confusion_matrix"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"Die Confusion Matrix wurde in {save_path} gespeichert.")

# Funktion zur Speicherung der Confusion Matrix für das Test-Set als Bild
def save_test_confusion_matrix(labels, preds, class_names, output_dir=default_output_dir, base_filename="test_confusion_matrix"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"Die Test-Confusion-Matrix wurde in {save_path} gespeichert.")


# Funktion zur Speicherung der falsch klassifizierten Bilder in einer Excel-Datei
def save_misclassified_to_excel(misclassified_images, output_dir=default_output_dir, base_filename="misclassified_images"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)

    df = pd.DataFrame(misclassified_images)
    df.to_excel(save_path, index=False)
    print(f"Die Liste der falsch klassifizierten Bilder wurde in {save_path} gespeichert.")

# Funktion zur Speicherung der falsch klassifizierten Test-Bilder in einer Excel-Datei
def save_test_misclassified_to_excel(misclassified_images, output_dir=default_output_dir, base_filename="test_misclassified_images"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)

    df = pd.DataFrame(misclassified_images)
    df.to_excel(save_path, index=False)
    print(f"Die Liste der falsch klassifizierten Test-Bilder wurde in {save_path} gespeichert.")


# Funktion zur Speicherung der Validierungsmetriken in einer Excel-Datei
def save_validation_metrics_to_excel(metrics, output_dir=default_output_dir, base_filename="validation_metrics_by_class"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.index.name = "Klasse"
    df_metrics.reset_index(inplace=True)
    df_metrics.to_excel(save_path, index=False)
    print(f"Validierungsmetriken pro Klasse erfolgreich gespeichert als '{save_path}'.")

def save_test_metrics_to_excel(metrics, output_dir=default_output_dir, base_filename="test_metrics_by_class"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)

    # Umwandlung der Metriken in einen DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.index.name = "Klasse"
    df_metrics.reset_index(inplace=True)

    # Speichern der Metriken in eine Excel-Datei
    df_metrics.to_excel(save_path, index=False)
    print(f"Testmetriken pro Klasse erfolgreich gespeichert als '{save_path}'.")

# Funktion zur Speicherung der Trainings- und Validierungsplots
def save_training_plots(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs,
                        output_dir=default_output_dir, base_filename="training_plots"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"Die Trainings- und Validierungsplots wurden in {save_path} gespeichert.")

# Funktion zur Speicherung der Test-Plot-Daten
def save_test_plots(test_losses, test_accuracies, num_epochs, output_dir=default_output_dir, base_filename="test_plots"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"Die Test-Plots wurden in {save_path} gespeichert.")


# Gesamtzeit starten
start_train_time = time.time()

def train_and_evaluate():
    num_epochs = config["Epochen"]
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        # Training
        model.train()
        running_train_loss, running_train_corrects = 0.0, 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1} [Training]', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0).float()
            running_train_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = running_train_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc.item())
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')

        # Validierung
        model.eval()
        running_valid_loss, running_valid_corrects = 0.0, 0
        all_preds, all_labels, all_misclassified_images = [], [], []

        for inputs, labels in tqdm(valid_loader, desc=f'Epoch {epoch + 1} [Validation]', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                confidences = torch.sigmoid(outputs).squeeze()

            running_valid_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0).float()
            running_valid_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Speichern der Fehlklassifikationen
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    all_misclassified_images.append({
                        'image_path': val_image_paths[i],
                        'true_label': val_labels[i].item() if isinstance(val_labels[i], torch.Tensor) else val_labels[
                            i],
                        'predicted_label': preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i],
                        'confidence_score': confidences[i].item() if isinstance(confidences[i], torch.Tensor) else
                        confidences[i]
                    })

        epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
        epoch_valid_acc = running_valid_corrects.double() / len(valid_loader.dataset)
        valid_losses.append(epoch_valid_loss)
        valid_accuracies.append(epoch_valid_acc.item())

        epoch_precision = precision_score(all_labels, all_preds, average='binary')
        epoch_recall = recall_score(all_labels, all_preds, average='binary')
        epoch_f1_score = f1_score(all_labels, all_preds, average='binary')

        print(f'Valid Loss: {epoch_valid_loss:.4f} Acc: {epoch_valid_acc:.4f}')
        print(f'Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1_score:.4f}')

        # Scheduler Update
        scheduler.step(epoch_valid_loss)

    total_duration = time.time() - start_train_time
    print(f"Gesamtdauer des Trainings: {total_duration / 60:.2f} Minuten")

    def test_model(model, test_loader, criterion, device):
        print("\nStarte die Evaluierung des Test-Datensatzes...")
        model.eval()
        running_test_loss, running_test_corrects = 0.0, 0
        test_preds, test_labels = [], []
        all_misclassified_images = []
        misclassified_linden = []  # Liste für falsch klassifizierte Linden
        misclassified_non_linden = []  # Liste für falsch klassifizierte Nicht-Linden

        # Sicherstellen, dass die Anzahl der Bildpfade mit dem Test-Datensatz übereinstimmt
        assert len(test_image_paths) == len(
            test_loader.dataset), "Mismatch zwischen test_image_paths und Test-Dataset-Größe."

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Test Evaluation", unit="batch")):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                confidences = torch.sigmoid(outputs).squeeze()

                running_test_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0).float()
                running_test_corrects += torch.sum(preds == labels.data)

                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

                # Berechnung des Index-Offsets für den aktuellen Batch
                start_idx = batch_idx * test_loader.batch_size
                end_idx = start_idx + len(labels)

                # Speichern der Fehlklassifikationen
                for i in range(len(labels)):
                    image_path = test_image_paths[start_idx + i]  # Pfad zum aktuellen Bild
                    true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
                    predicted_label = preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i]
                    confidence_score = confidences[i].item() if isinstance(confidences[i], torch.Tensor) else \
                    confidences[i]

                    # Alle Fehlklassifikationen speichern
                    if predicted_label != true_label:
                        all_misclassified_images.append({
                            'image_path': image_path,
                            'true_label': true_label,
                            'predicted_label': predicted_label,
                            'confidence_score': confidence_score
                        })

                    # Falsch klassifizierte Linden filtern
                    if true_label == 0 and predicted_label == 1:
                        misclassified_linden.append({
                            'image_path': image_path,
                            'true_label': true_label,
                            'predicted_label': predicted_label,
                            'confidence_score': confidence_score
                        })

                    # Falsch klassifizierte Keine Linden filtern
                    if true_label == 1 and predicted_label == 0:
                        misclassified_non_linden.append({
                            'image_path': image_path,
                            'true_label': true_label,
                            'predicted_label': predicted_label,
                            'confidence_score': confidence_score
                        })

        # Fehlklassifikationen speichern
        test_misclassified_path = save_file_with_timestamp("test_misclassified_images", "xlsx", default_output_dir)
        pd.DataFrame(all_misclassified_images).to_excel(test_misclassified_path, index=False)

        linden_path = save_file_with_timestamp("misclassified_linden", "xlsx", default_output_dir)
        pd.DataFrame(misclassified_linden).to_excel(linden_path, index=False)

        non_linden_path = save_file_with_timestamp("misclassified_non_linden", "xlsx", default_output_dir)
        pd.DataFrame(misclassified_non_linden).to_excel(non_linden_path, index=False)

        print(f"Fehlklassifikationen gespeichert in:\n{test_misclassified_path}\n{linden_path}\n{non_linden_path}")

        # Testmetriken berechnen
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_test_acc = running_test_corrects.double() / len(test_loader.dataset)

        epoch_precision = precision_score(test_labels, test_preds, average='binary')
        epoch_recall = recall_score(test_labels, test_preds, average='binary')
        epoch_f1_score = f1_score(test_labels, test_preds, average='binary')

        print(f'\nTest Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}')
        print(f'Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1_score:.4f}')
        print(f'Anzahl falsch klassifizierter Linden: {len(misclassified_linden)}')
        print(f'Anzahl falsch klassifizierter Keine Linden: {len(misclassified_non_linden)}')

        # Confusion Matrix berechnen
        cm = confusion_matrix(test_labels, test_preds)
        print("\nConfusion Matrix für Test-Set:")
        print(cm)



        return epoch_test_loss, epoch_test_acc.item(), test_preds, test_labels

    test_loss, test_accuracy, test_preds, test_labels = test_model(model, test_loader, criterion, device)

    # Berechnung der Metriken für jede Klasse
    metrics = {}
    for label, class_name in zip([0, 1], ["Linden", "Keine Linden"]):
        class_preds = [1 if pred == label else 0 for pred in all_preds]
        class_labels = [1 if lbl == label else 0 for lbl in all_labels]

        precision = precision_score(class_labels, class_preds)
        recall = recall_score(class_labels, class_preds)
        f1 = f1_score(class_labels, class_preds)
        accuracy = accuracy_score(class_labels, class_preds)

        metrics[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': accuracy
        }
        print(f"\nMetriken für Klasse '{class_name}':")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Berechnung der Metriken für jede Klasse
        metrics = {}
        for label, class_name in zip([0, 1], ["Linden", "Keine Linden"]):
            # Erstellen von Listen für Vorhersagen und tatsächliche Labels der aktuellen Klasse
            class_preds = [1 if pred == label else 0 for pred in all_preds]
            class_labels = [1 if lbl == label else 0 for lbl in all_labels]

            # Berechnung der Metriken
            precision = precision_score(class_labels, class_preds)
            recall = recall_score(class_labels, class_preds)
            f1 = f1_score(class_labels, class_preds)
            accuracy = accuracy_score(class_labels, class_preds)

            # Speichern der Metriken in einem Dictionary
            metrics[class_name] = {
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Accuracy': accuracy
            }

            # Ausgabe der Metriken für die aktuelle Klasse
            print(f"\nMetriken für Klasse '{class_name}':")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

        # Speichern der finalen Metriken in config
        config["Train Loss"] = epoch_train_loss
        config["Validation Loss"] = epoch_valid_loss
        config["Train Accuracy"] = epoch_train_acc.item()
        config["Validation Accuracy"] = epoch_valid_acc.item()
        config["Precision"] = epoch_precision
        config["Recall"] = epoch_recall
        config["F1 Score"] = epoch_f1_score
        config["Training Time (minutes)"] = round(total_duration / 60, 2)
        config["Test Loss"] = test_loss
        config["Test Accuracy"] = test_accuracy

        # Speichern der Trainingsparameter und Metriken in Excel-Datei
        speichere_training_parameter(config)

    # Berechnung der Metriken für jede Klasse
    test_metrics = {}
    for label, class_name in zip([0, 1], ["Linden", "Keine Linden"]):
        # Erstellen von Listen für Vorhersagen und tatsächliche Labels der aktuellen Klasse
        class_preds = [1 if pred == label else 0 for pred in all_preds]
        class_labels = [1 if lbl == label else 0 for lbl in all_labels]

        # Berechnung der Metriken
        precision = precision_score(class_labels, class_preds)
        recall = recall_score(class_labels, class_preds)
        f1 = f1_score(class_labels, class_preds)
        accuracy = accuracy_score(class_labels, class_preds)

        # Speichern der Metriken in einem Dictionary
        test_metrics[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': accuracy
        }

        # Ausgabe der Metriken für die aktuelle Klasse
        print(f"\nMetriken für Klasse '{class_name}':")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    # Speicherung der finalen Metriken in test_config
    test_config = {
        "Modellarchitektur": "ResNet152",
        "Batch-Größe": 32,
        "Epochen": 30,
        "Lernrate": 0.001,
        "Momentum": 0.9,
        "criterion": "nn.BCEWithLogitsLoss",
        "Optimizer": "SGD",
        "Scheduler": "CosineAnnealingLR",
        "Aktivierungsfunktion": "ReLU",
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Precision (Linden)": test_metrics["Linden"]["Precision"],
        "Recall (Linden)": test_metrics["Linden"]["Recall"],
        "F1 Score (Linden)": test_metrics["Linden"]["F1-Score"],
        "Accuracy (Linden)": test_metrics["Linden"]["Accuracy"],
        "Precision (Keine Linden)": test_metrics["Keine Linden"]["Precision"],
        "Recall (Keine Linden)": test_metrics["Keine Linden"]["Recall"],
        "F1 Score (Keine Linden)": test_metrics["Keine Linden"]["F1-Score"],
        "Accuracy (Keine Linden)": test_metrics["Keine Linden"]["Accuracy"],
        "Test Duration (minutes)": round(total_duration / 60, 2)
    }

    # Speichern der Testparameter und Metriken in Excel-Datei
    speichere_test_parameter(test_config)

    # Plotten der Trainings- und Validierungsverluste und -genauigkeiten
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    # Plotten der Testverluste und Testgenauigkeiten
    epochs = range(1, num_epochs + 1)  # Stellen Sie sicher, dass num_epochs korrekt de
    test_losses = [test_loss] * num_epochs  # Test Loss für jede Epoche wiederholen
    test_accuracies = [test_accuracy] * num_epochs  # Test Accuracy für jede Epoche wi
    plt.figure(figsize=(12, 4))

    # Verlustdiagramm
    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Genauigkeitsdiagramm
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Speichern der Confusion Matrix und anderer Ergebnisse
    save_confusion_matrix(all_labels, all_preds, ["Linden", "Keine Linden"])
    save_test_confusion_matrix(test_labels, test_preds,  ["Linden", "Keine Linden"])
    save_misclassified_to_excel(all_misclassified_images)
    save_test_misclassified_to_excel(all_misclassified_images)
    save_validation_metrics_to_excel(metrics)
    save_test_metrics_to_excel(metrics)
    save_training_plots(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs)
    save_test_plots(test_losses, test_accuracies, num_epochs)
# Starte das Training und die Validierung
train_and_evaluate()




