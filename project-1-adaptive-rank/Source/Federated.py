def train_client(model, loader, device, epochs=1, lr=0.001):
    """
    Simulates local training on a single client.
    Returns: The updated state_dict and the number of samples processed.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_samples = 0
    
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_samples += y.size(0)
            
    return model.state_dict(), total_samples

def fedavg(client_weights, client_sample_counts):
    """
    Performs Federated Averaging (FedAvg).
    Weights the global model update by the volume of data each client contributed.
    """
    total_samples = sum(client_sample_counts)
    global_dict = {}
    
    # Iterate through all parameters in the first model
    for key in client_weights[0].keys():
        # Compute weighted sum for each parameter across all clients
        global_dict[key] = sum(
            client_weights[i][key] * (client_sample_counts[i] / total_samples)
            for i in range(len(client_weights))
        )
    return global_dict
