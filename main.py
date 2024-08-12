import argparse
import torch
import numpy as np
import pandas as pd
from models.Transformer import create_transformer_model
from sklearn.model_selection import train_test_split
import os
import json
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer for Solar Generation Forecasting')
    
    # Required arguments
    parser.add_argument('--preprocessed_data', type=str, required=True, help='Path to the preprocessed data file')
    
    # Optional arguments with default values
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (default: -1, i.e., CPU)')
    parser.add_argument('--save', type=str, default='model.pt', help='Path to save the model')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model (default: 64)')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in multi-head attention (default: 4)')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers (default: 3)')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='Number of decoder layers (default: 3)')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of feedforward network (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--loss_history', type=str, default=None, help='Path to save the loss history (default: None, i.e., do not save)')

    args = parser.parse_args()
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    return args

def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.cuda else 'cpu')
    print(f"Using device: {device}")

    # Load preprocessed data
    with open(args.preprocessed_data, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    X = preprocessed_data['X']
    y = preprocessed_data['y']
    features = preprocessed_data['features']
    
    # Add window and horizon to args
    args.window = preprocessed_data['window']
    args.horizon = preprocessed_data['horizon']
    
    # Print shapes for debugging
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of features: {len(features)}")
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(-1).to(device)

    # Print shapes after conversion for debugging
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Create a Data object to pass to Transformer
    class Data:
        def __init__(self, input_dim, output_dim):
            self.input_dim = input_dim
            self.output_dim = output_dim

    data = Data(input_dim=X_train.shape[2], output_dim=1)

    # Initialize the model
    model = create_transformer_model(args, data).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_y = y_train[i:i+args.batch_size]

            optimizer.zero_grad()
            # Create target input (shifted by one step)
            tgt_input = torch.cat([torch.zeros(batch_X.size(0), 1, 1).to(device), batch_y[:, :-1]], dim=1)
            outputs = model(batch_X, tgt_input)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(X_train) / args.batch_size)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(X_val), args.batch_size):
                batch_X = X_val[i:i+args.batch_size]
                batch_y = y_val[i:i+args.batch_size]
                tgt_input = torch.cat([torch.zeros(batch_X.size(0), 1, 1).to(device), batch_y[:, :-1]], dim=1)
                outputs = model(batch_X, tgt_input)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(X_val) / args.batch_size)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the loss history if specified
    if args.loss_history:
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        os.makedirs(os.path.dirname(args.loss_history), exist_ok=True)
        with open(args.loss_history, 'w') as f:
            json.dump(history, f)
        print(f'Loss history saved to {args.loss_history}')

    # Save the model
    torch.save(model.state_dict(), args.save)
    print(f'Model saved to {args.save}')

if __name__ == "__main__":
    main()