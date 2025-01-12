import torch
import torch.nn as nn

def train_model(model, train_loader, dev_loader, epochs, lr, device, model_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    best_loss=1e8

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) 
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dev_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                dev_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.8f}, dev Loss: {dev_loss/len(dev_loader):.8f}")

        if dev_loss < best_loss:
            print("Saving the model...")
            torch.save(model.state_dict(), model_path)
            best_loss = dev_loss


def train_RevDimTSFormer(model, train_loader, dev_loader, epochs, lr, device, model_path):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    best_loss=1e8

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device) 
            X_mark_batch = X_mark_batch.to(device)
            X_dec_batch = X_dec_batch.to(device) 
            X_dec_mark_batch = X_dec_mark_batch.to(device) 
            optimizer.zero_grad()
            outputs = model(X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch, y_batch in dev_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device) 
                X_mark_batch = X_mark_batch.to(device)
                X_dec_batch = X_dec_batch.to(device) 
                X_dec_mark_batch = X_dec_mark_batch.to(device)
                outputs = model(X_batch, X_mark_batch, X_dec_batch, X_dec_mark_batch)
                loss = criterion(outputs, y_batch)
                dev_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.8f}, dev Loss: {dev_loss/len(dev_loader):.8f}")

        if dev_loss < best_loss:
            print("Saving the model...")
            torch.save(model.state_dict(), model_path)
            best_loss = dev_loss