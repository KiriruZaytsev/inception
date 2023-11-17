import torch
import torchvision
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device):
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_logits)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model, dataloader, loss_fn, device):

    test_loss, test_acc = 0, 0

    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_logits = model(X)

            loss = loss_fn(y_logits, y)
            test_loss += loss

            y_pred = torch.argmax(y_logits, dim = 1)
            test_acc += (y_pred == y).sum().item() / len(y_logits)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc

def train_model(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model, 
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device = device)

        test_loss, test_acc = test_step(model=model, 
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device = device)


        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results