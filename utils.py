import os
from skimage import io, color
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as DatasetPT
from PIL import Image, ImageOps
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from torch_geometric.data import Data, Dataset
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
from torch.nn.functional import max_pool2d
from skimage.segmentation import slic
import SimpleITK as sitk
import logging
from radiomics import featureextractor



###################################################
### Function to load images in bw from a folder ###
###################################################

def load_images_from_folder(folder):
    images = []
    for image_name in tqdm(os.listdir(folder)):
        image_path = os.path.join(folder, image_name)
        image = io.imread(image_path)
        if len(image.shape) == 3 and image.shape[2] > 1:
            image = color.rgb2gray(image[:, :, :3])
        images.append(image)
    return images



##################################################
### Function to find the minimum width, height ###
##################################################

def min_dimensions(images, min_width, min_height):
    for image in images:
        width, height = image.shape
        if width < min_width:
          min_width = width
          out_width = [width, height]
        if height < min_height:
          min_height = height
          out_height = [width, height]
    return out_width, out_height



########################################
### Function to plot multiple images ###
#########################################

def plot_images(images, title, labels):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(8):
        ax = axs[i // 4, i % 4]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(labels[i], fontsize=10)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    


#########################################
### Class to create a PyTorch Dataset ###
#########################################

class MRIDataset(DatasetPT):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.labels = {'yes': 1, 'no': 0}
        self.images, self.image_labels = self.load_images_and_labels()

    def load_images_and_labels(self):
        images = []
        image_labels = []
        for label in ['yes', 'no']:
            label_dir = os.path.join(self.directory, label)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                images.append(image_path)
                image_labels.append(self.labels[label])
        return images, image_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.image_labels[idx]
        return image, label
    
    

#########################################
### Function to resize and pad images ###
#########################################

def resize_and_pad(img, output_size=(224, 224)):
    # Get the color of the top-left pixel for padding
    padding_color = img.getpixel((3, 40))

    # Calculate the aspect ratio
    aspect_ratio = img.width / img.height

    # Calculate padding if needed
    if aspect_ratio > 1:  # Width is greater than height
        new_height = int(img.width / aspect_ratio)
        padding = (0, (img.width - new_height) // 2)
    else:  # Height is greater than or equal to width
        new_width = int(img.height * aspect_ratio)
        padding = ((img.height - new_width) // 2, 0)

    # Pad the image
    img = ImageOps.expand(img, padding, fill=padding_color)

    # Resize the image
    img = img.resize(output_size, Image.LANCZOS)

    return img



#######################################################
### Function to calculate mean and sd of the images ###
#######################################################

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std



##########################################################
### Function to plot multiple images from a dataloader ###
##########################################################

def show_images(dataset, num_images=4):
    yes_images = []
    no_images = []

    for image, label in dataset:
        if label == dataset.labels['yes'] and len(yes_images) < num_images:
            yes_images.append(image)
        elif label == dataset.labels['no'] and len(no_images) < num_images:
            no_images.append(image)

        if len(yes_images) == num_images and len(no_images) == num_images:
            break

    fig, axs = plt.subplots(2, num_images, figsize=(12, 6))
    for i in range(num_images):
        axs[0, i].imshow(yes_images[i].permute(1, 2, 0), cmap='gray')
        axs[0, i].set_title('Tumor')
        axs[0, i].axis('off')

        axs[1, i].imshow(no_images[i].permute(1, 2, 0), cmap='gray')
        axs[1, i].set_title('No tumor')
        axs[1, i].axis('off')

    plt.show()
    
    

#####################################################################
### Function to create patches and adjacency matrix from an image ###
#####################################################################
    
def create_patches_and_adjacency(image, patch_size, k=5):
    h, w, _ = image.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch.reshape(-1))

    patches = np.array(patches)
    patch_embeddings = patches

    similarities = cosine_similarity(patch_embeddings)

    adjacency_matrix = np.zeros_like(similarities)
    for i in range(similarities.shape[0]):
        idx = np.argsort(similarities[i])[-k-1:-1]
        adjacency_matrix[i, idx] = 1

    return patch_embeddings, adjacency_matrix



################################################
### Function to create patches from an image ###
################################################

def create_patches(image, patch_size, num_patches):
    h, w, _ = image.shape
    patches = {}
    idx = 0

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if idx >= num_patches:
                break
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[:2] == (patch_size, patch_size):
                patches[idx] = patch
                idx += 1

    return patches



################################
### Function to plot a graph ###
################################

def plot_graph(G, patches):
    pos = nx.spring_layout(G)

    plt.figure(figsize=(8, 8))

    nx.draw_networkx_edges(G, pos, alpha=0.5)

    ax = plt.gca()
    for node, (x, y) in pos.items():
        if node in patches:
            img = patches[node]
            im = OffsetImage(img, zoom=0.5)  # Adjust zoom as needed
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)

    plt.axis('off')

    plt.show()
    
    
    
###################################################
### Class to create a PyTorch Geometric Dataset ###
###################################################

class PatchedMRIDataset(Dataset):
    def __init__(self, original_dataset, patch_size, adjacency_k):
        super(PatchedMRIDataset, self).__init__()
        self.original_dataset = original_dataset
        self.patch_size = patch_size
        self.adjacency_k = adjacency_k

    def len(self):
        return len(self.original_dataset)

    def create_patches(self, image):
        image = image.squeeze(0).cpu().numpy()
        h, w = image.shape
        patches = []
        pos_encodings = []

        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                if patch.shape[:2] == (self.patch_size, self.patch_size):
                    patches.append(patch.reshape(-1))
                    pos_encodings.append([i / h, j / w])

        patches_array = np.array(patches)
        pos_encodings_array = np.array(pos_encodings)

        return torch.tensor(patches_array, dtype=torch.float), torch.tensor(pos_encodings_array, dtype=torch.float)


    def create_adjacency_matrix(self, patches):
        similarities = cosine_similarity(patches)
        adjacency_matrix = np.zeros_like(similarities)
        for i in range(similarities.shape[0]):
            idx = np.argsort(similarities[i])[-self.adjacency_k-1:-1]  # Exclude self
            adjacency_matrix[i, idx] = 1
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        return edge_index

    def get(self, idx):
        image, label = self.original_dataset[idx]
        patches, pos_encodings = self.create_patches(image)
        edge_index = self.create_adjacency_matrix(patches)
        x = torch.cat((patches, pos_encodings), dim=1)

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
        return data
    
    
    
############################################    
### Function that perform the train step ###
############################################

def train_step(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    correct = 0

    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()

        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    train_loss = total_train_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    return train_loss, train_acc



################################    
### Function to test a model ###
################################

def test(model, test_loader, criterion, device, plot=True):
    model.eval()

    total_val_loss = 0
    correct = 0
    
    if plot:
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  

            val_loss = criterion(out, data.y)
            total_val_loss += val_loss.item()
            
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            
            if plot:
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())

    val_loss = total_val_loss / len(test_loader.dataset)
    val_acc = correct / len(test_loader.dataset)
    
    if plot:
        return val_loss, val_acc, all_preds, all_labels
    
    return val_loss, val_acc



#################################    
### Function to train a model ###
#################################

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, plot=True):
    model.to(device)
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(epochs):
        train_loss, train_accuracy = train_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = test(model, val_loader, criterion, device, plot=False)
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
        
        epoch_info = f"[Epoch {epoch+1}/{epochs}]"
        bold_epoch_info = f"\033[1m{epoch_info}\033[0m"
        print(f"{bold_epoch_info:<20} Train Loss: {train_loss:8.4f} | "
              f"Train Accuracy: {train_accuracy:6.2f}% | "
              f"Val Loss: {val_loss:8.4f} | "
              f"Val Accuracy: {val_accuracy:6.2f}%")
        
    if plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    return model
        
        

##############################################################   
### Function to test a model and plot the confusion matrix ###
##############################################################
        
def test_and_plot(model, test_loader, criterion, device, flag="GNN"):
    
    if flag == "CNN":
        _, _, all_preds, all_labels = test_CNN(model, test_loader, criterion, device, plot=True)
    else:    
        _, _, all_preds, all_labels = test(model, test_loader, criterion, device, plot=True)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}\n\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    

########################################################
### Function that perform the train step for the CNN ###
########################################################
    
def train_step_CNN(model, train_loader, optimizer, criterion, device):
    model.train()
    total_train_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == labels).sum())

    train_loss = total_train_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    return train_loss, train_acc



######################################    
### Function to test the CNN model ###
######################################

def test_CNN(model, test_loader, criterion, device, plot=True):
    model.eval()
    
    total_val_loss = 0
    correct = 0
    
    if plot:
        all_preds = []
        all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)

            val_loss = criterion(out, labels)
            total_val_loss += val_loss.item()
            
            pred = out.argmax(dim=1)
            correct += int((pred == labels).sum())
            
            if plot:
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    val_loss = total_val_loss / len(test_loader.dataset)
    val_acc = correct / len(test_loader.dataset)

    if plot:
        return val_loss, val_acc, all_preds, all_labels
    
    return val_loss, val_acc



#######################################    
### Function to train the CNN model ###
#######################################

def train_CNN(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, plot=True):
    model.to(device)

    best_val_accuracy = 0.0
    best_model_state = None
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_accuracy = train_step_CNN(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = test_CNN(model, val_loader, criterion, device, plot=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()

        epoch_info = f"[Epoch {epoch+1}/{epochs}]"
        bold_epoch_info = f"\033[1m{epoch_info}\033[0m"
        print(f"{bold_epoch_info:<20} Train Loss: {train_loss:8.4f} | "
              f"Train Accuracy: {train_accuracy:6.2f}% | "
              f"Val Loss: {val_loss:8.4f} | "
              f"Val Accuracy: {val_accuracy:6.2f}%")

    if plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model



###################################################################################
### Class to create a PyTorch Geometric Dataset with patches of different sizes ###
###################################################################################

class PatchedMRIDatasetv2(Dataset):
    def __init__(self, original_dataset, adjacency_k):
        super(PatchedMRIDatasetv2, self).__init__()
        self.original_dataset = original_dataset
        self.adjacency_k = adjacency_k

    def len(self):
        return len(self.original_dataset)

    def create_patches(self, image):
        image = image.squeeze(0).cpu().numpy()
        h, w = image.shape
        patches = []
        pos_encodings = []
        patch_sizes = [7, 14, 56]

        for patch_size in patch_sizes:
            for i in range(0, h, patch_size):
                for j in range(0, w, patch_size):
                    patch = image[i:i+patch_size, j:j+patch_size]
                    if patch.shape[:2] == (patch_size, patch_size):
                        if patch_size != 7:
                            patch_tensor = torch.tensor(patch, dtype=torch.float).unsqueeze(0).unsqueeze(0)
                            patch_tensor = max_pool2d(patch_tensor, kernel_size=patch_size // 7)
                            patch = patch_tensor.squeeze().numpy()
                        patches.append(patch.reshape(-1))
                        pos_encodings.append([i / h, j / w])

        patches_array = np.array(patches)
        pos_encodings_array = np.array(pos_encodings)

        return torch.tensor(patches_array, dtype=torch.float), torch.tensor(pos_encodings_array, dtype=torch.float)


    def create_adjacency_matrix(self, patches):
        similarities = cosine_similarity(patches)
        adjacency_matrix = np.zeros_like(similarities)
        for i in range(similarities.shape[0]):
            idx = np.argsort(similarities[i])[-self.adjacency_k-1:-1]
            adjacency_matrix[i, idx] = 1
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        return edge_index

    def get(self, idx):
        image, label = self.original_dataset[idx]
        patches, pos_encodings = self.create_patches(image)
        edge_index = self.create_adjacency_matrix(patches)
        x = torch.cat((patches, pos_encodings), dim=1)

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
        return data



###########################################################################
### Class to create a PyTorch Geometric Dataset from Radiomics features ###
###########################################################################

class GenMRIDataset(Dataset):

    def __init__(self, directory, adjacency_k=3, num_superpixels=50):
        super(GenMRIDataset, self).__init__()
        self.directory = directory
        self.num_superpixels = num_superpixels
        self.labels = {'yes': 1, 'no': 0}
        self.images, self.image_labels = self.load_images_and_labels()
        self.adjacency_k = adjacency_k


    def create_superpixels(self, image_path):
        image = io.imread(image_path)
        if len(image.shape) == 3 and image.shape[2] > 1:
            image = color.rgb2gray(image[:, :, :3])
        segments = slic(image, n_segments=self.num_superpixels, compactness=10)
        mask_list = []
        for label in np.unique(segments):
            mask = np.zeros_like(image)
            mask[segments == label] = label + 1
            mask_list.append(mask)
        return mask_list


    def create_patches(self, image_path, mask_list):

        logging.getLogger('radiomics').setLevel(logging.CRITICAL)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        
        patches = []

        image_sitk = sitk.ReadImage(image_path)
        image_spacing = image_sitk.GetSpacing()

        for mask in mask_list:

            mask = np.where(mask != 0, 1, mask)
            mask_sitk = sitk.GetImageFromArray(mask)

            mask_spacing = mask_sitk.GetSpacing()
            if mask_spacing != image_spacing:
                resampler = sitk.ResampleImageFilter()
                resampler.SetOutputSpacing(image_spacing)
                resampler.SetSize(image_sitk.GetSize())
                mask_sitk = resampler.Execute(mask_sitk)
            
            if image_sitk.GetNumberOfComponentsPerPixel() > 1:
                image_sitk = sitk.VectorIndexSelectionCast(image_sitk, 0, sitk.sitkUInt8)

            try:
                result = extractor.execute(image_sitk, mask_sitk)
                feature_values = [v for k, v in result.items() if k.startswith(('o', 'w'))]
                features_array = np.array(feature_values)
                patches.append(features_array)
            except:
                pass
                

        if len(patches) == 0:
            return None

        patches_array = np.array(patches)
        
        return torch.tensor(patches_array, dtype=torch.float)


    def load_images_and_labels(self):
        images = []
        image_labels = []
        for label in ['yes', 'no']:
            label_dir = os.path.join(self.directory, label)
            for image_name in tqdm(os.listdir(label_dir)):
                image_path = os.path.join(label_dir, image_name)
                mask_list = self.create_superpixels(image_path)
                patches = self.create_patches(image_path, mask_list)
                if patches is not None:
                    images.append(patches)
                    image_labels.append(self.labels[label])
        return images, image_labels


    def create_adjacency_matrix(self, patches):
        similarities = cosine_similarity(patches)
        adjacency_matrix = np.zeros_like(similarities)
        for i in range(similarities.shape[0]):
            idx = np.argsort(similarities[i])[-self.adjacency_k-1:-1]
            adjacency_matrix[i, idx] = 1
        edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
        return edge_index


    def len(self):
        return len(self.images)


    def get(self, idx):
        image = self.images[idx]
        label = self.image_labels[idx]
        edge_index = self.create_adjacency_matrix(image)
        data = Data(x=image, edge_index=edge_index, y=torch.tensor([label]))
        return data