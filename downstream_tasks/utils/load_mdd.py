import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class BrainROIDataset(Dataset):
    def __init__(self, roi_data, labels, img_mean=None, img_std=None, is_train=True):
        """
        Brain ROI time series dataset
        
        Parameters:
            roi_data: Extracted ROI time series data
            labels: Corresponding labels
            img_mean: Mean for normalization
            img_std: Std for normalization
            is_train: Whether is training set
        """
        self.roi_data = roi_data
        self.labels = labels
        self.is_train = is_train
        
        # Normalize if mean and std are provided
        if img_mean is not None and img_std is not None:
            self.roi_data = (self.roi_data - img_mean) / (img_std + 1e-8)
    
    def __len__(self):
        return len(self.roi_data)
    
    def __getitem__(self, idx):
        """获取单个样本

        Args:
            idx: 索引

        Returns:
            tuple: (roi_data, label)
        """
        roi_data = self.roi_data[idx]
        label = self.labels[idx]
        
        # 确保数据是单通道的，形状为 [1, H, W]
        if len(roi_data.shape) == 2:
            roi_data = roi_data.unsqueeze(0)  # 添加通道维度
        elif len(roi_data.shape) == 3 and roi_data.shape[0] == 3:
            # 如果是3通道，转换为单通道（取平均值）
            roi_data = roi_data.mean(dim=0, keepdim=True)
            
        return roi_data, label

def get_dataloaders(data_dir, rows=90, cols=116, test_size=0.2, val_size=0.2, batch_size=32, random_state=42):
    """
    Load ROI time series data and return data loaders
    
    Parameters:
        data_dir: Data directory path
        rows: Number of rows to extract (default: 90, the minimum across all files)
        cols: Number of columns to extract
        test_size: Test set ratio
        val_size: Validation set ratio
        batch_size: Batch size
        random_state: Random seed
    
    Returns:
        train_loader, val_loader, test_loader: Data loaders for training, validation and test sets
    """
    # Extract data and labels
    data = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            # Extract labels from filenames
            # ROISignals_S1-1-0001.mat: -1- means MDD, -2- means normal
            if '-1-' in filename:
                label = 1  # MDD patient
            elif '-2-' in filename:
                label = 0  # Normal
            else:
                continue  # Skip files that don't match the format
            
            # Load .mat file
            file_path = os.path.join(data_dir, filename)
            try:
                mat_data = sio.loadmat(file_path)
                
                if 'ROISignals' not in mat_data:
                    print(f"Warning: 'ROISignals' field not found in {filename}")
                    continue
                    
                roi_signals = mat_data['ROISignals']
                
                # Ensure we only use the specified number of rows and columns
                if roi_signals.shape[0] < rows or roi_signals.shape[1] < cols:
                    print(f"Warning: {filename} has insufficient dimensions ({roi_signals.shape})")
                    continue
                
                # Take the first 'rows' rows and first 'cols' columns
                extracted_data = roi_signals[:rows, :cols]
                data.append(extracted_data)
                labels.append(label)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Convert to NumPy arrays
    X = np.array(data, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    print(f"Loaded {len(data)} samples with shape {rows}x{cols}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split into train, validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),  # Adjust validation set ratio
        random_state=random_state,
        stratify=y_train_val
    )
    
    # Calculate stats for normalization
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = BrainROIDataset(X_train, y_train, mean, std, is_train=True)
    val_dataset = BrainROIDataset(X_val, y_val, mean, std, is_train=False)
    test_dataset = BrainROIDataset(X_test, y_test, mean, std, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Class distribution - Training set: {np.bincount(y_train.numpy())}")
    print(f"Class distribution - Validation set: {np.bincount(y_val.numpy())}")
    print(f"Class distribution - Test set: {np.bincount(y_test.numpy())}")
    
    return train_loader, val_loader, test_loader

def load_dataset(args, is_train=True, return_eval=False):
    """
    Load MDD datasets for Brain-JEPA format
    """
    data_dir = args.data_dir if hasattr(args, 'data_dir') else "/home/chentingyu/BrainDiagnostics/demo/ROISignals_FunImgARCWF"
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 16
    rows = args.crop_size[0] if hasattr(args, 'crop_size') else 90
    cols = args.crop_size[1] if hasattr(args, 'crop_size') else 116
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        rows=rows,
        cols=cols,
        batch_size=batch_size
    )
    
    if is_train:
        if return_eval:
            return train_loader, val_loader
        return train_loader
    else:
        return test_loader 