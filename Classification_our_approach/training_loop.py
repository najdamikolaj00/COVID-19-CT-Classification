import torch

def training_loop(model, optimizer, loss_function, k_folds, train_loaders, val_loaders, num_epochs):

    for fold in range(k_folds):
        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        for epoch in range(num_epochs):

            model.train()
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass
                inputs = batch['img']
                labels = batch['label']
                outputs = model(inputs)
                
                # Compute loss
                loss = loss_function(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad() #zeroes out the gradients of all the model parameters.
                loss.backward() #computes the gradients of the model's parameters with respect to the loss
                optimizer.step() #updates the model's parameters using the computed gradients and the chosen optimization algorithm

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            #torch.no_grad ensures that no gradients are computed during this process, as we don't need them for evaluation.
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['img']
                    labels = batch['label']
                    outputs = model(inputs)
                    
                    # Compute validation loss
                    val_loss += loss_function(outputs, labels).item()
                    
                    # Compute accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f'Fold: {fold+1}/{k_folds}, Epoch: {epoch+1}/{num_epochs}, '
                f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, '
                f'Val Acc: {(100 * correct / total):.2f}%')
