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

# Update der Trainings-Transforms, um RandomAffine und Cutout hinzuzufügen
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),  # Zufälliger Zuschnitt auf die Zielgröße
    transforms.RandomApply([  # Wende Farb-/Kontrasttransformationen gelegentlich an
        transforms.ColorJitter(
            brightness=0.2,  # Helligkeitsänderung
            contrast=0.2,  # Kontraständerung
            saturation=0.2,  # Sättigungsänderung
            hue=0.2  # Farbtonänderung
        ),
        transforms.RandomGrayscale(p=0.3)  # Wahrscheinlichkeitsbasierte Umwandlung in Graustufen
    ], p=1.0),  # Anwenden mit einer Wahrscheinlichkeit von 100%
    transforms.ToTensor(),  # Konvertiert das Bild in ein Tensor-Format
    transforms.RandomOrder([  # Wendet Transformationen in zufälliger Reihenfolge an
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))], p=0.5),  # Gaussian Blur
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),  # Zufälliges Löschen
        transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5)  # Zufällige Affin-Transformation
    ]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisierung auf Standardwerte
])

# Validierungs- und Test-Transforms für Vorverarbeitung
valid_test_transform = transforms.Compose([
    transforms.Resize(320),  # Ändert die Größe des Bildes
    transforms.CenterCrop(299),  # Zuschneiden auf die Zielgröße
    transforms.ToTensor(),  # Konvertiert das Bild in ein Tensor-Format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisierung auf Standardwerte
])

# Custom Dataset-Klasse zum Laden von Bildern und Labels
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # Pfade zu den Bildern
        self.labels = labels  # Labels der Bilder
        self.transform = transform  # Transformationen

    def __len__(self):  # Gibt die Anzahl der Bilder zurück
        return len(self.image_paths)

    def __getitem__(self, idx):  # Ruft ein Bild und das zugehörige Label ab
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Öffnet das Bild im RGB-Format
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)  # Transformation anwenden
        return image, label

# Funktion zum Sammeln der Bildpfade und Labels
def get_image_paths_and_labels(directory, label):
    valid_extensions = (".jpg", ".jpeg", ".png")  # Gültige Bildformate
    image_paths = [os.path.join(root, file) for root, _, files in os.walk(directory)
                   for file in files if file.lower().endswith(valid_extensions)]  # Suche nach Bilddateien
    labels = [label] * len(image_paths)  # Labels für alle Bilder erstellen
    return image_paths, labels

# Manuelle Angabe der Pfade für Trainings-, Validierungs- und Test-Daten
linden_train_dir = "D:\\Labeln\\binomialverteilung\\output\\train\\Linden"  # Train-Pfad für Linden
keine_linden_train_dir = "D:\\Labeln\\binomialverteilung\\output\\train\\Keine Linden"  # Train-Pfad für Keine Linden
linden_val_dir = "D:\\Labeln\\binomialverteilung\\output\\val\\Linden"  # Val-Pfad für Linden
keine_linden_val_dir = "D:\\Labeln\\binomialverteilung\\output\\val\\Keine Linden"  # Val-Pfad für Keine Linden
linden_test_dir = "D:\\Labeln\\binomialverteilung\\output\\test\\Linden"  # Test-Pfad für Linden
keine_linden_test_dir = "D:\\Labeln\\binomialverteilung\\output\\test\\Keine Linden"  # Test-Pfad für Keine Linden

# Sammeln der Bildpfade und Labels für Training
train_image_paths, train_labels = [], []
for dir_path, label in [(linden_train_dir, 0), (keine_linden_train_dir, 1)]:  # Für jede Klasse
    paths, lbls = get_image_paths_and_labels(dir_path, label)  # Pfade und Labels sammeln
    train_image_paths.extend(paths)
    train_labels.extend(lbls)

# Sammeln der Bildpfade und Labels für Validierung
val_image_paths, val_labels = [], []
for dir_path, label in [(linden_val_dir, 0), (keine_linden_val_dir, 1)]:  # Für jede Klasse
    paths, lbls = get_image_paths_and_labels(dir_path, label)  # Pfade und Labels sammeln
    val_image_paths.extend(paths)
    val_labels.extend(lbls)

# Sammeln der Bildpfade und Labels für Test
test_image_paths, test_labels = [], []
for dir_path, label in [(linden_test_dir, 0), (keine_linden_test_dir, 1)]:  # Für jede Klasse
    paths, lbls = get_image_paths_and_labels(dir_path, label)  # Pfade und Labels sammeln
    test_image_paths.extend(paths)
    test_labels.extend(lbls)

# Ausgabe der Anzahl der Bilder vor der Balancierung
print("\nAnzahl der Bilder im ursprünglichen Trainingsdatensatz:")
print(f"Linden: {train_labels.count(0)}, Keine Linden: {train_labels.count(1)}")

print("\nAnzahl der Bilder im ursprünglichen Validierungsdatensatz:")
print(f"Linden: {val_labels.count(0)}, Keine Linden: {val_labels.count(1)}")

print("\nAnzahl der Bilder im ursprünglichen Test-Datensatz:")
print(f"Linden: {test_labels.count(0)}, Keine Linden: {test_labels.count(1)}")

# Funktion zum Balancieren des Datensatzes mit einer zusätzlichen Menge
def balance_dataset(image_paths, labels, class_label, extra_ratio=1.2):
    class_indices = [i for i, label in enumerate(labels) if label == class_label]  # Indizes der Zielklasse
    other_indices = [i for i in range(len(labels)) if i not in class_indices]  # Indizes anderer Klassen
    target_num_class = int(len(other_indices) * extra_ratio)  # Zielanzahl für Zielklasse

    if len(class_indices) > target_num_class:
        balanced_class_indices = random.sample(class_indices, target_num_class)  # Sample
    else:
        balanced_class_indices = class_indices + random.choices(class_indices, k=target_num_class - len(class_indices))  # Auffüllen

    balanced_indices = balanced_class_indices + other_indices  # Kombinieren
    return [image_paths[i] for i in balanced_indices], [labels[i] for i in balanced_indices]

# Balancieren der Datensätze
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Train DataLoader
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Validierungs-Loader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Test-Loader

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
        self.model = models.resnet152(pretrained=True)  # Geladenes ResNet152
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Anpassung für binäre Klassifikation

    def forward(self, x):
        return self.model(x)  # Vorwärtsdurchlauf

# Modell initialisieren
model = ModifiedResNet152().to(device)  # Modell auf das Gerät übertragen

# Verlustfunktion (Binary Cross Entropy Loss mit Logits)
criterion = nn.BCEWithLogitsLoss()

# Optimierer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Momentum für Stabilität

# Scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)  # Lernratenplan

# Festlegen des zentralen Ausgabeordners
default_output_dir = r"D:\Labeln\binomialverteilung\output\metriken"

# Hilfsfunktion zum Speichern mit Zeitstempel
def save_file_with_timestamp(base_filename, extension, output_dir=default_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Erzeugt einen Zeitstempel im Format YYYYMMDD_HHMMSS
    filename = f"{base_filename}_{timestamp}.{extension}"  # Kombiniert Basisnamen, Zeitstempel und Erweiterung
    save_path = os.path.join(output_dir, filename)  # Erzeugt den vollständigen Pfad
    os.makedirs(output_dir, exist_ok=True)  # Erstellt den Ordner, falls er nicht existiert
    return save_path  # Gibt den Speicherpfad zurück

# Funktion zur Speicherung der Trainingsparameter in einer Excel-Datei
def speichere_training_parameter(config, output_dir=default_output_dir, base_filename="training_parameters"):
    df = pd.DataFrame([config])  # Konvertiert das Konfigurations-Dictionary in einen DataFrame
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Generiert den Speicherpfad
    df.to_excel(save_path, index=False)  # Speichert den DataFrame in eine Excel-Datei
    print(f"Die Trainingsparameter wurden in '{save_path}' gespeichert.")  # Ausgabe des Speicherorts

# Zentralisiertes Dictionary für alle Trainingsparameter
config = {
    "Modellarchitektur": "ResNet152",  # Genutzte Modellarchitektur
    "Batch-Größe": 32,  # Batch-Größe
    "Epochen": 30,  # Anzahl der Trainings-Epochen
    "Lernrate": 0.001,  # Lernrate
    # "Weight Decay": 0.0001,  # L2-Regularisierung (auskommentiert, falls nicht genutzt)
    "Momentum": 0.9,  # Momentum-Wert
    "criterion": "nn.BCEWithLogitsLoss",  # Verlustfunktion
    "Optimizer": "SGD",  # Optimierungsverfahren
    "Scheduler": "CosineAnnealingLR",  # Scheduler-Typ
    "Aktivierungsfunktion": "ReLU",  # Aktivierungsfunktion
    "Train Loss": None,  # Trainingsverlust (wird später aktualisiert)
    "Validation Loss": None,  # Validierungsverlust
    "Train Accuracy": None,  # Trainingsgenauigkeit
    "Validation Accuracy": None,  # Validierungsgenauigkeit
    "Precision": None,  # Präzision
    "Recall": None,  # Recall
    "F1 Score": None,  # F1-Score
    "Training Time (minutes)": None  # Trainingszeit in Minuten
}

# Funktion zur Speicherung der Testparameter in einer Excel-Datei
def speichere_test_parameter(config, output_dir=default_output_dir, base_filename="test_parameters"):
    """
    Speichert die Testparameter in einer Excel-Datei im angegebenen Ordner.
    Fügt automatisch einen Zeitstempel zur Datei hinzu.
    """
    df = pd.DataFrame([config])  # Konvertiert das Konfigurations-Dictionary in einen DataFrame
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Generiert den Speicherpfad
    df.to_excel(save_path, index=False)  # Speichert den DataFrame in eine Excel-Datei
    print(f"Die Testparameter wurden in '{save_path}' gespeichert.")  # Ausgabe des Speicherorts

# Beispiel-Dictionary für Testparameter
test_config = {
    "Modellarchitektur": "ResNet152",  # Genutzte Modellarchitektur
    "Batch-Größe": 32,  # Batch-Größe
    "Epochen": 30,  # Anzahl der Trainings-Epochen
    "Lernrate": 0.001,  # Lernrate
    "Weight Decay": None,  # L2-Regularisierung (falls genutzt)
    "Momentum": 0.9,  # Momentum-Wert
    "criterion": "nn.BCEWithLogitsLoss",  # Verlustfunktion
    "Optimizer": "SGD",  # Optimierungsverfahren
    "Scheduler": "CosineAnnealingLR",  # Scheduler-Typ
    "Aktivierungsfunktion": "ReLU",  # Aktivierungsfunktion
    "Test Loss": None,  # Testverlust (wird später aktualisiert)
    "Test Accuracy": None,  # Testgenauigkeit
    "Precision": None,  # Präzision
    "Recall": None,  # Recall
    "F1 Score": None,  # F1-Score
    "Test Duration (minutes)": None  # Testzeit in Minuten
}

# Funktion zur Anzeige von Beispielbildern mit Vorhersagen und tatsächlichen Labels
def display_images_with_predictions(model, dataset, class_names, num_linden=10, num_keine_linden=10):
    linden_indices = [i for i, (_, label) in enumerate(dataset) if label == 0][:num_linden]  # Indizes für Linden
    keine_linden_indices = [i for i, (_, label) in enumerate(dataset) if label == 1][:num_keine_linden]  # Indizes für Keine Linden

    linden_loader = DataLoader(Subset(dataset, linden_indices), batch_size=num_linden, shuffle=True)  # DataLoader für Linden
    keine_linden_loader = DataLoader(Subset(dataset, keine_linden_indices), batch_size=num_keine_linden, shuffle=True)  # DataLoader für Keine Linden

    model.eval()  # Setzt das Modell in den Evaluierungsmodus
    plt.figure(figsize=(15, 15))  # Erzeugt ein Plot-Fenster
    count = 0  # Zähler für die Bilder
    mean, std = np.array([0.2, 0.2, 0.2]), np.array([0.8, 0.8, 0.8])  # Normalisierungsparameter

    with torch.no_grad():  # Deaktiviert das Gradienten-Tracking
        for images, labels in linden_loader:  # Iteriert über Linden-Bilder
            images, labels = images.to(device), labels.to(device)  # Überträgt Daten auf das Gerät
            outputs = model(images)  # Berechnet die Modellvorhersagen
            preds = torch.argmax(outputs, dim=1)  # Ermittelt die Vorhersagen
            for j in range(images.size(0)):  # Iteriert über Bilder im Batch
                plt.subplot(5, 4, count + 1)  # Erstellt ein Unterplot
                image = images[j].cpu().permute(1, 2, 0).numpy()  # Konvertiert Tensor zu Bild
                image = np.clip(std * image + mean, 0, 1)  # Denormalisiert das Bild
                plt.imshow(image)  # Zeigt das Bild
                color = 'green' if preds[j] == labels[j] else 'red'  # Farbe je nach Korrektheit der Vorhersage
                plt.title(f'Pred: {class_names[int(preds[j])]} | True: {class_names[labels[j]]}', color=color)  # Titel des Plots
                plt.axis("off")  # Entfernt die Achsen
                count += 1  # Erhöht den Zähler
                if count >= num_linden + num_keine_linden:  # Beendet die Schleife, wenn genug Bilder gezeigt wurden
                    break

        for images, labels in keine_linden_loader:  # Iteriert über Keine Linden-Bilder
            images, labels = images.to(device), labels.to(device)  # Überträgt Daten auf das Gerät
            outputs = model(images)  # Berechnet die Modellvorhersagen
            preds = torch.argmax(outputs, dim=1)  # Ermittelt die Vorhersagen
            for j in range(images.size(0)):  # Iteriert über Bilder im Batch
                plt.subplot(5, 4, count + 1)  # Erstellt ein Unterplot
                image = images[j].cpu().permute(1, 2, 0).numpy()  # Konvertiert Tensor zu Bild
                image = np.clip(std * image + mean, 0, 1)  # Denormalisiert das Bild
                plt.imshow(image)  # Zeigt das Bild
                color = 'green' if preds[j] == labels[j] else 'red'  # Farbe je nach Korrektheit der Vorhersage
                plt.title(f'Pred: {class_names[int(preds[j])]} | True: {class_names[labels[j]]}', color=color)  # Titel des Plots
                plt.axis("off")  # Entfernt die Achsen
                count += 1  # Erhöht den Zähler
                if count >= num_linden + num_keine_linden:  # Beendet die Schleife, wenn genug Bilder gezeigt wurden
                    break

    plt.tight_layout(pad=3.0)  # Optimiert die Layout-Abstände
    plt.show()  # Zeigt die Plots

# Funktion zur Speicherung der Confusion Matrix als Bild
def save_confusion_matrix(labels, preds, class_names, output_dir=default_output_dir, base_filename="confusion_matrix"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)  # Speicherpfad für die Datei erstellen

    cm = confusion_matrix(labels, preds)  # Erzeugt die Confusion Matrix
    plt.figure(figsize=(8, 6))  # Erstellt eine neue Abbildung
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)  # Heatmap
    plt.xlabel("Predicted")  # Beschriftung der x-Achse
    plt.ylabel("True")  # Beschriftung der y-Achse
    plt.title("Confusion Matrix")  # Titel der Matrix
    plt.savefig(save_path)  # Speichert die Abbildung
    plt.close()  # Schließt die Abbildung
    print(f"Die Confusion Matrix wurde in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der Confusion Matrix für das Test-Set als Bild
def save_test_confusion_matrix(labels, preds, class_names, output_dir=default_output_dir, base_filename="test_confusion_matrix"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)  # Speicherpfad für die Datei erstellen

    cm = confusion_matrix(labels, preds)  # Erzeugt die Confusion Matrix
    plt.figure(figsize=(8, 6))  # Erstellt eine neue Abbildung
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)  # Heatmap
    plt.xlabel("Predicted")  # Beschriftung der x-Achse
    plt.ylabel("True")  # Beschriftung der y-Achse
    plt.title("Test Confusion Matrix")  # Titel der Matrix
    plt.savefig(save_path)  # Speichert die Abbildung
    plt.close()  # Schließt die Abbildung
    print(f"Die Test-Confusion-Matrix wurde in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der falsch klassifizierten Bilder in einer Excel-Datei
def save_misclassified_to_excel(misclassified_images, output_dir=default_output_dir, base_filename="misclassified_images"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Speicherpfad für die Datei erstellen

    df = pd.DataFrame(misclassified_images)  # Konvertiert die Liste in einen DataFrame
    df.to_excel(save_path, index=False)  # Speichert die DataFrame-Daten in einer Excel-Datei
    print(f"Die Liste der falsch klassifizierten Bilder wurde in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der falsch klassifizierten Test-Bilder in einer Excel-Datei
def save_test_misclassified_to_excel(misclassified_images, output_dir=default_output_dir, base_filename="test_misclassified_images"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Speicherpfad für die Datei erstellen

    df = pd.DataFrame(misclassified_images)  # Konvertiert die Liste in einen DataFrame
    df.to_excel(save_path, index=False)  # Speichert die DataFrame-Daten in einer Excel-Datei
    print(f"Die Liste der falsch klassifizierten Test-Bilder wurde in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der Validierungsmetriken in einer Excel-Datei
def save_validation_metrics_to_excel(metrics, output_dir=default_output_dir, base_filename="validation_metrics_by_class"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Speicherpfad für die Datei erstellen

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')  # Konvertiert die Metriken in einen DataFrame
    df_metrics.index.name = "Klasse"  # Setzt den Indexnamen
    df_metrics.reset_index(inplace=True)  # Setzt den Index zurück
    df_metrics.to_excel(save_path, index=False)  # Speichert die Metriken in eine Excel-Datei
    print(f"Validierungsmetriken pro Klasse erfolgreich gespeichert als '{save_path}'.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der Testmetriken in einer Excel-Datei
def save_test_metrics_to_excel(metrics, output_dir=default_output_dir, base_filename="test_metrics_by_class"):
    save_path = save_file_with_timestamp(base_filename, "xlsx", output_dir)  # Speicherpfad für die Datei erstellen

    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')  # Konvertiert die Metriken in einen DataFrame
    df_metrics.index.name = "Klasse"  # Setzt den Indexnamen
    df_metrics.reset_index(inplace=True)  # Setzt den Index zurück
    df_metrics.to_excel(save_path, index=False)  # Speichert die Metriken in eine Excel-Datei
    print(f"Testmetriken pro Klasse erfolgreich gespeichert als '{save_path}'.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der Trainings- und Validierungsplots
def save_training_plots(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs,
                        output_dir=default_output_dir, base_filename="training_plots"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)  # Speicherpfad für die Datei erstellen

    epochs = range(1, num_epochs + 1)  # Erzeugt eine Liste der Epochen
    plt.figure(figsize=(12, 4))  # Erstellt eine neue Abbildung mit spezifischer Größe
    plt.subplot(1, 2, 1)  # Erstellt das erste Subplot für Verluste
    plt.plot(epochs, train_losses, label='Training Loss')  # Plottet die Trainingsverluste
    plt.plot(epochs, valid_losses, label='Validation Loss')  # Plottet die Validierungsverluste
    plt.title('Training and Validation Loss')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Loss')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    plt.subplot(1, 2, 2)  # Erstellt das zweite Subplot für Genauigkeiten
    plt.plot(epochs, train_accuracies, label='Training Accuracy')  # Plottet die Trainingsgenauigkeit
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')  # Plottet die Validierungsgenauigkeit
    plt.title('Training and Validation Accuracy')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Accuracy')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    plt.savefig(save_path)  # Speichert die Abbildung
    plt.close()  # Schließt die Abbildung
    print(f"Die Trainings- und Validierungsplots wurden in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Funktion zur Speicherung der Test-Plot-Daten
def save_test_plots(test_losses, test_accuracies, num_epochs, output_dir=default_output_dir, base_filename="test_plots"):
    save_path = save_file_with_timestamp(base_filename, "png", output_dir)  # Speicherpfad für die Datei erstellen

    epochs = range(1, num_epochs + 1)  # Erzeugt eine Liste der Epochen
    plt.figure(figsize=(12, 4))  # Erstellt eine neue Abbildung mit spezifischer Größe

    plt.subplot(1, 2, 1)  # Erstellt das erste Subplot für Verluste
    plt.plot(epochs, test_losses, label='Test Loss')  # Plottet die Testverluste
    plt.title('Test Loss')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Loss')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    plt.subplot(1, 2, 2)  # Erstellt das zweite Subplot für Genauigkeiten
    plt.plot(epochs, test_accuracies, label='Test Accuracy')  # Plottet die Testgenauigkeit
    plt.title('Test Accuracy')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Accuracy')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    plt.savefig(save_path)  # Speichert die Abbildung
    plt.close()  # Schließt die Abbildung
    print(f"Die Test-Plots wurden in {save_path} gespeichert.")  # Ausgabe des Speicherorts

# Gesamtzeit starten
start_train_time = time.time()  # Startet die Zeitmessung für die gesamte Trainingsdauer

# Trainings- und Validierungsprozess
def train_and_evaluate():
    num_epochs = config["Epochen"]  # Anzahl der Epochen aus der Konfiguration
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []  # Listen zur Speicherung von Verlust und Genauigkeit

    for epoch in range(num_epochs):  # Schleife über die Epochen
        print(f'Epoch {epoch + 1}/{num_epochs}')  # Ausgabe der aktuellen Epoche
        # Training
        model.train()  # Setzt das Modell in den Trainingsmodus
        running_train_loss, running_train_corrects = 0.0, 0  # Variablen für aggregierten Verlust und Korrektheit

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1} [Training]', unit='batch'):  # Iteration über Trainings-Batches
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Daten auf das Gerät übertragen
            optimizer.zero_grad()  # Gradienten zurücksetzen
            outputs = model(inputs)  # Vorwärtsdurchlauf
            loss = criterion(outputs, labels)  # Verlust berechnen
            loss.backward()  # Backpropagation durchführen
            optimizer.step()  # Parameter-Update mit dem Optimierer

            running_train_loss += loss.item() * inputs.size(0)  # Aggregiert den Verlust über alle Datenpunkte
            preds = (outputs > 0).float()  # Berechnet die Vorhersagen (Schwellenwert von 0.5)
            running_train_corrects += torch.sum(preds == labels.data)  # Summiert die korrekten Vorhersagen

        epoch_train_loss = running_train_loss / len(train_loader.dataset)  # Durchschnittlicher Trainingsverlust
        epoch_train_acc = running_train_corrects.double() / len(train_loader.dataset)  # Trainingsgenauigkeit
        train_losses.append(epoch_train_loss)  # Speichert den Verlust
        train_accuracies.append(epoch_train_acc.item())  # Speichert die Genauigkeit
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')  # Ausgabe von Verlust und Genauigkeit

        # Validierung
        model.eval()  # Setzt das Modell in den Evaluierungsmodus
        running_valid_loss, running_valid_corrects = 0.0, 0  # Variablen für aggregierten Verlust und Korrektheit
        all_preds, all_labels, all_misclassified_images = [], [], []  # Listen für Vorhersagen, Labels und Fehlklassifikationen

        for inputs, labels in tqdm(valid_loader, desc=f'Epoch {epoch + 1} [Validation]', unit='batch'):  # Iteration über Validierungs-Batches
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Daten auf das Gerät übertragen
            with torch.no_grad():  # Deaktiviert Gradientenberechnung
                outputs = model(inputs)  # Vorwärtsdurchlauf
                loss = criterion(outputs, labels)  # Verlust berechnen
                confidences = torch.sigmoid(outputs).squeeze()  # Konvertiert die Logits in Wahrscheinlichkeiten

            running_valid_loss += loss.item() * inputs.size(0)  # Aggregiert den Verlust über alle Datenpunkte
            preds = (outputs > 0).float()  # Berechnet die Vorhersagen (Schwellenwert von 0.5)
            running_valid_corrects += torch.sum(preds == labels.data)  # Summiert die korrekten Vorhersagen

            all_preds.extend(preds.cpu().numpy())  # Speichert alle Vorhersagen
            all_labels.extend(labels.cpu().numpy())  # Speichert alle Labels

            # Speichern der Fehlklassifikationen
            for i in range(len(labels)):  # Iteriert über jedes Bild im Batch
                if preds[i] != labels[i]:  # Überprüft, ob die Vorhersage falsch ist
                    all_misclassified_images.append({
                        'image_path': val_image_paths[i],  # Bildpfad
                        'true_label': val_labels[i].item() if isinstance(val_labels[i], torch.Tensor) else val_labels[i],  # Wahres Label
                        'predicted_label': preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i],  # Vorhergesagtes Label
                        'confidence_score': confidences[i].item() if isinstance(confidences[i], torch.Tensor) else confidences[i]  # Konfidenz
                    })

        epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)  # Durchschnittlicher Validierungsverlust
        epoch_valid_acc = running_valid_corrects.double() / len(valid_loader.dataset)  # Validierungsgenauigkeit
        valid_losses.append(epoch_valid_loss)  # Speichert den Verlust
        valid_accuracies.append(epoch_valid_acc.item())  # Speichert die Genauigkeit

        epoch_precision = precision_score(all_labels, all_preds, average='binary')  # Präzision berechnen
        epoch_recall = recall_score(all_labels, all_preds, average='binary')  # Recall berechnen
        epoch_f1_score = f1_score(all_labels, all_preds, average='binary')  # F1-Score berechnen

        print(f'Valid Loss: {epoch_valid_loss:.4f} Acc: {epoch_valid_acc:.4f}')  # Ausgabe von Verlust und Genauigkeit
        print(f'Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1_score:.4f}')  # Ausgabe der Metriken

        # Scheduler Update
        scheduler.step(epoch_valid_loss)  # Scheduler aktualisieren basierend auf dem Validierungsverlust

    total_duration = time.time() - start_train_time  # Berechnet die Gesamtdauer des Trainings
    print(f"Gesamtdauer des Trainings: {total_duration / 60:.2f} Minuten")  # Ausgabe der Gesamtzeit in Minuten

# Funktion zur Evaluierung des Modells mit dem Test-Datensatz
    def test_model(model, test_loader, criterion, device):
        print("\nStarte die Evaluierung des Test-Datensatzes...")  # Ausgabe zum Start des Testprozesses
        model.eval()  # Setzt das Modell in den Evaluierungsmodus
        running_test_loss, running_test_corrects = 0.0, 0  # Variablen zur Speicherung von Verlust und Korrektheit
        test_preds, test_labels = [], []  # Listen zur Speicherung von Vorhersagen und Labels
        all_misclassified_images = []  # Liste zur Speicherung aller Fehlklassifikationen
        misclassified_linden = []  # Liste für falsch klassifizierte Linden
        misclassified_non_linden = []  # Liste für falsch klassifizierte Nicht-Linden

        # Sicherstellen, dass die Anzahl der Bildpfade mit der Größe des Test-Datensatzes übereinstimmt
        assert len(test_image_paths) == len(test_loader.dataset), "Mismatch zwischen test_image_paths und Test-Dataset-Größe."

        with torch.no_grad():  # Deaktiviert die Gradientenberechnung für den Testprozess
            for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Test Evaluation", unit="batch")):
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)  # Daten auf das Gerät übertragen
                outputs = model(inputs)  # Vorhersagen berechnen
                loss = criterion(outputs, labels)  # Verlust berechnen
                confidences = torch.sigmoid(outputs).squeeze()  # Konfidenzwerte berechnen

                running_test_loss += loss.item() * inputs.size(0)  # Aggregiert den Verlust über alle Bilder
                preds = (outputs > 0).float()  # Schwellenwertbasierte Vorhersagen
                running_test_corrects += torch.sum(preds == labels.data)  # Zählt korrekte Vorhersagen

                test_preds.extend(preds.cpu().numpy())  # Speichert die Vorhersagen
                test_labels.extend(labels.cpu().numpy())  # Speichert die Labels

                # Berechnung des Index-Offsets für den aktuellen Batch
                start_idx = batch_idx * test_loader.batch_size  # Startindex basierend auf Batch-Größe
                end_idx = start_idx + len(labels)  # Endindex basierend auf der Anzahl der Labels

                # Speichern der Fehlklassifikationen
                for i in range(len(labels)):  # Iteriert über jedes Bild im Batch
                    image_path = test_image_paths[start_idx + i]  # Pfad zum aktuellen Bild
                    true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]  # Wahres Label
                    predicted_label = preds[i].item() if isinstance(preds[i], torch.Tensor) else preds[i]  # Vorhergesagtes Label
                    confidence_score = confidences[i].item() if isinstance(confidences[i], torch.Tensor) else confidences[i]  # Konfidenz

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
        test_misclassified_path = save_file_with_timestamp("test_misclassified_images", "xlsx", default_output_dir)  # Speicherpfad für alle Fehlklassifikationen
        pd.DataFrame(all_misclassified_images).to_excel(test_misclassified_path, index=False)  # Speichert die Fehlklassifikationen in eine Excel-Datei

        linden_path = save_file_with_timestamp("misclassified_linden", "xlsx", default_output_dir)  # Speicherpfad für falsch klassifizierte Linden
        pd.DataFrame(misclassified_linden).to_excel(linden_path, index=False)  # Speichert falsch klassifizierte Linden

        non_linden_path = save_file_with_timestamp("misclassified_non_linden", "xlsx", default_output_dir)  # Speicherpfad für falsch klassifizierte Nicht-Linden
        pd.DataFrame(misclassified_non_linden).to_excel(non_linden_path, index=False)  # Speichert falsch klassifizierte Nicht-Linden

        print(f"Fehlklassifikationen gespeichert in:\n{test_misclassified_path}\n{linden_path}\n{non_linden_path}")  # Ausgabe der Speicherorte

        # Testmetriken berechnen
        epoch_test_loss = running_test_loss / len(test_loader.dataset)  # Durchschnittlicher Testverlust
        epoch_test_acc = running_test_corrects.double() / len(test_loader.dataset)  # Testgenauigkeit

        epoch_precision = precision_score(test_labels, test_preds, average='binary')  # Präzision berechnen
        epoch_recall = recall_score(test_labels, test_preds, average='binary')  # Recall berechnen
        epoch_f1_score = f1_score(test_labels, test_preds, average='binary')  # F1-Score berechnen

        print(f'\nTest Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}')  # Ausgabe des Testverlusts und der Genauigkeit
        print(f'Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} F1: {epoch_f1_score:.4f}')  # Ausgabe der Testmetriken
        print(f'Anzahl falsch klassifizierter Linden: {len(misclassified_linden)}')  # Anzahl falsch klassifizierter Linden
        print(f'Anzahl falsch klassifizierter Keine Linden: {len(misclassified_non_linden)}')  # Anzahl falsch klassifizierter Keine Linden

        # Confusion Matrix berechnen
        cm = confusion_matrix(test_labels, test_preds)  # Berechnet die Confusion Matrix
        print("\nConfusion Matrix für Test-Set:")  # Ausgabe der Confusion Matrix
        print(cm)

        return epoch_test_loss, epoch_test_acc.item(), test_preds, test_labels  # Gibt die Testmetriken zurück

    # Aufruf der Testfunktion
    test_loss, test_accuracy, test_preds, test_labels = test_model(model, test_loader, criterion, device)  # Führt die Evaluierung des Test-Datensatzes durch

    # Berechnung der Metriken für jede Klasse
    metrics = {}  # Initialisierung eines Dictionaries zur Speicherung der Metriken
    for label, class_name in zip([0, 1], ["Linden", "Keine Linden"]):  # Iteriert über die Klassenlabels und deren Namen
        class_preds = [1 if pred == label else 0 for pred in all_preds]  # Erzeugt eine Liste binärer Vorhersagen für die Klasse
        class_labels = [1 if lbl == label else 0 for lbl in all_labels]  # Erzeugt eine Liste binärer Labels für die Klasse

        # Berechnung der Metriken
        precision = precision_score(class_labels, class_preds)  # Berechnet die Präzision
        recall = recall_score(class_labels, class_preds)  # Berechnet den Recall
        f1 = f1_score(class_labels, class_preds)  # Berechnet den F1-Score
        accuracy = accuracy_score(class_labels, class_preds)  # Berechnet die Genauigkeit

        # Speichern der Metriken in einem Dictionary
        metrics[class_name] = {
            'Precision': precision,  # Speichert die Präzision
            'Recall': recall,  # Speichert den Recall
            'F1-Score': f1,  # Speichert den F1-Score
            'Accuracy': accuracy  # Speichert die Genauigkeit
        }

        # Ausgabe der Metriken für die aktuelle Klasse
        print(f"\nMetriken für Klasse '{class_name}':")  # Gibt die Klasse aus
        print(f"Precision: {precision:.4f}")  # Gibt die Präzision aus
        print(f"Recall: {recall:.4f}")  # Gibt den Recall aus
        print(f"F1-Score: {f1:.4f}")  # Gibt den F1-Score aus
        print(f"Accuracy: {accuracy:.4f}")  # Gibt die Genauigkeit aus

        # Speichern der finalen Metriken in config
        config["Train Loss"] = epoch_train_loss  # Speichert den Trainingsverlust
        config["Validation Loss"] = epoch_valid_loss  # Speichert den Validierungsverlust
        config["Train Accuracy"] = epoch_train_acc.item()  # Speichert die Trainingsgenauigkeit
        config["Validation Accuracy"] = epoch_valid_acc.item()  # Speichert die Validierungsgenauigkeit
        config["Precision"] = epoch_precision  # Speichert die Präzision
        config["Recall"] = epoch_recall  # Speichert den Recall
        config["F1 Score"] = epoch_f1_score  # Speichert den F1-Score
        config["Training Time (minutes)"] = round(total_duration / 60, 2)  # Speichert die Trainingszeit in Minuten
        config["Test Loss"] = test_loss  # Speichert den Testverlust
        config["Test Accuracy"] = test_accuracy  # Speichert die Testgenauigkeit

        # Speichern der Trainingsparameter und Metriken in Excel-Datei
        speichere_training_parameter(config)  # Ruft die Funktion zum Speichern der Trainingsparameter auf

# Berechnung der Metriken für jede Klasse im Test-Set
    test_metrics = {}  # Initialisierung eines Dictionaries zur Speicherung der Test-Metriken
    for label, class_name in zip([0, 1], ["Linden", "Keine Linden"]):  # Iteriert über die Klassenlabels und deren Namen
        class_preds = [1 if pred == label else 0 for pred in all_preds]  # Erzeugt eine Liste binärer Vorhersagen für die Klasse
        class_labels = [1 if lbl == label else 0 for lbl in all_labels]  # Erzeugt eine Liste binärer Labels für die Klasse

        # Berechnung der Metriken
        precision = precision_score(class_labels, class_preds)  # Berechnet die Präzision
        recall = recall_score(class_labels, class_preds)  # Berechnet den Recall
        f1 = f1_score(class_labels, class_preds)  # Berechnet den F1-Score
        accuracy = accuracy_score(class_labels, class_preds)  # Berechnet die Genauigkeit

    # Speichern der Metriken in einem Dictionary
        test_metrics[class_name] = {
            'Precision': precision,  # Speichert die Präzision
            'Recall': recall,  # Speichert den Recall
            'F1-Score': f1,  # Speichert den F1-Score
            'Accuracy': accuracy  # Speichert die Genauigkeit
        }

        # Ausgabe der Metriken für die aktuelle Klasse
        print(f"\nMetriken für Klasse '{class_name}':")  # Gibt die Klasse aus
        print(f"Precision: {precision:.4f}")  # Gibt die Präzision aus
        print(f"Recall: {recall:.4f}")  # Gibt den Recall aus
        print(f"F1-Score: {f1:.4f}")  # Gibt den F1-Score aus
        print(f"Accuracy: {accuracy:.4f}")  # Gibt die Genauigkeit aus

    # Speicherung der finalen Metriken in test_config
    test_config = {
        "Modellarchitektur": "ResNet152",  # Name der Modellarchitektur
        "Batch-Größe": 32,  # Batch-Größe, die für das Training und Testen verwendet wird
        "Epochen": 30,  # Anzahl der Trainings-Epochen
        "Lernrate": 0.001,  # Lernrate für den Optimierer
        "Momentum": 0.9,  # Momentum-Wert für den SGD-Optimierer
        "criterion": "nn.BCEWithLogitsLoss",  # Verlustfunktion
        "Optimizer": "SGD",  # Verwendeter Optimierer
        "Scheduler": "CosineAnnealingLR",  # Scheduler-Typ für die Anpassung der Lernrate
        "Aktivierungsfunktion": "ReLU",  # Aktivierungsfunktion, die im Modell verwendet wird
        "Test Loss": test_loss,  # Verlust des Modells auf dem Test-Datensatz
        "Test Accuracy": test_accuracy,  # Genauigkeit des Modells auf dem Test-Datensatz
        "Precision (Linden)": test_metrics["Linden"]["Precision"],  # Präzision für die Klasse "Linden"
        "Recall (Linden)": test_metrics["Linden"]["Recall"],  # Recall für die Klasse "Linden"
        "F1 Score (Linden)": test_metrics["Linden"]["F1-Score"],  # F1-Score für die Klasse "Linden"
        "Accuracy (Linden)": test_metrics["Linden"]["Accuracy"],  # Genauigkeit für die Klasse "Linden"
        "Precision (Keine Linden)": test_metrics["Keine Linden"]["Precision"],  # Präzision für die Klasse "Keine Linden"
        "Recall (Keine Linden)": test_metrics["Keine Linden"]["Recall"],  # Recall für die Klasse "Keine Linden"
        "F1 Score (Keine Linden)": test_metrics["Keine Linden"]["F1-Score"],  # F1-Score für die Klasse "Keine Linden"
        "Accuracy (Keine Linden)": test_metrics["Keine Linden"]["Accuracy"],  # Genauigkeit für die Klasse "Keine Linden"
        "Test Duration (minutes)": round(total_duration / 60, 2)  # Dauer des Tests in Minuten, gerundet
    }

    # Speichern der Testparameter und Metriken in Excel-Datei
    speichere_test_parameter(test_config)  # Ruft die Funktion zum Speichern der Testparameter auf

    # Plotten der Trainings- und Validierungsverluste und -genauigkeiten
    epochs = range(1, num_epochs + 1)  # Erzeugt eine Liste von Epochen (1 bis num_epochs)
    plt.figure(figsize=(12, 4))  # Erstellt ein Diagrammfenster mit einer Breite von 12 und Höhe von 4

    # Plot für Training und Validation Loss
    plt.subplot(1, 2, 1)  # Erstellt das erste Subplot
    plt.plot(epochs, train_losses, label='Training Loss')  # Plottet die Trainingsverluste
    plt.plot(epochs, valid_losses, label='Validation Loss')  # Plottet die Validierungsverluste
    plt.title('Training and Validation Loss')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Loss')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    # Plot für Training und Validation Accuracy
    plt.subplot(1, 2, 2)  # Erstellt das zweite Subplot
    plt.plot(epochs, train_accuracies, label='Training Accuracy')  # Plottet die Trainingsgenauigkeit
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')  # Plottet die Validierungsgenauigkeit
    plt.title('Training and Validation Accuracy')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Accuracy')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu
    plt.show()  # Zeigt die Plots an

    # Plotten der Testverluste und Testgenauigkeiten
    epochs = range(1, num_epochs + 1)  # Erzeugt eine Liste von Epochen (1 bis num_epochs)
    test_losses = [test_loss] * num_epochs  # Wiederholt den Testverlust für jede Epoche
    test_accuracies = [test_accuracy] * num_epochs  # Wiederholt die Testgenauigkeit für jede Epoche
    plt.figure(figsize=(12, 4))  # Erstellt ein Diagrammfenster mit einer Breite von 12 und Höhe von 4

    # Plot für Test Loss
    plt.subplot(1, 2, 1)  # Erstellt das erste Subplot
    plt.plot(epochs, test_losses, label='Test Loss')  # Plottet den Testverlust
    plt.title('Test Loss')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Loss')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    # Plot für Test Accuracy
    plt.subplot(1, 2, 2)  # Erstellt das zweite Subplot
    plt.plot(epochs, test_accuracies, label='Test Accuracy')  # Plottet die Testgenauigkeit
    plt.title('Test Accuracy')  # Titel des Subplots
    plt.xlabel('Epochs')  # Beschriftung der x-Achse
    plt.ylabel('Accuracy')  # Beschriftung der y-Achse
    plt.legend()  # Fügt eine Legende hinzu

    plt.tight_layout()  # Optimiert die Abstände zwischen den Subplots
    plt.show()  # Zeigt die Plots an

    # Speichern der Confusion Matrix und anderer Ergebnisse
    save_confusion_matrix(all_labels, all_preds, ["Linden", "Keine Linden"])  # Speichert die Confusion Matrix für alle Labels
    save_test_confusion_matrix(test_labels, test_preds, ["Linden", "Keine Linden"])  # Speichert die Confusion Matrix für das Test-Set
    save_misclassified_to_excel(all_misclassified_images)  # Speichert alle Fehlklassifikationen in Excel
    save_test_misclassified_to_excel(all_misclassified_images)  # Speichert die Test-Fehlklassifikationen in Excel
    save_validation_metrics_to_excel(metrics)  # Speichert die Validierungsmetriken in Excel
    save_test_metrics_to_excel(metrics)  # Speichert die Testmetriken in Excel
    save_training_plots(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs)  # Speichert die Trainingsplots
    save_test_plots(test_losses, test_accuracies, num_epochs)  # Speichert die Testplots

# Starte das Training und die Validierung
train_and_evaluate()  # Ruft die Funktion für Training und Validierung auf
