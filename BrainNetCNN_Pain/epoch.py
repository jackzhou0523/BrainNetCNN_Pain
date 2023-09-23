import torch


def train_model(model,
          criterion,
          optimizer,
          train_loader):

    train_loss = 0.0
    train_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set to training
    model.train()

    # Training loop
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device, dtype=torch.float), target.to(device=device)

        # Clear gradients
        optimizer.zero_grad()
        output = model(data)

        # target = target.unsqueeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)

        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

        train_acc += accuracy.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    train_acc = train_acc / len(train_loader.sampler)
    return train_loss, train_acc


def valid_model(model,
          criterion,
          valid_loader):
    valid_loss = 0.0
    valid_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # start validation
    with torch.no_grad():
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                data, target = data.to(device=device, dtype=torch.float), target.to(device=device)

                output = model(data)
                # target = target.unsqueeze(1)
                loss = criterion(output, target)

                valid_loss += loss.item() * data.size(0)

                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

                valid_acc += accuracy.item() * data.size(0)


            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_acc = valid_acc / len(valid_loader.sampler)

    return valid_loss, valid_acc


