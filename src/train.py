from src.data_create import get_batch


def train(model, train_data, valid_data, optimizer, criterion, scheduler, batch_size, observation_period_num):
    model.train()
    total_loss_train = 0.0
    for batch, i in enumerate(range(0, len(train_data), batch_size)):
        data, targets = get_batch(
            train_data, i, batch_size, observation_period_num)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss_train += loss.item()
    scheduler.step()
    total_loss_train = total_loss_train/len(train_data)

    model.eval()
    total_loss_valid = 0.0
    for i in range(0, len(valid_data), batch_size):
        data, targets = get_batch(
            valid_data, i, batch_size, observation_period_num)
        output = model(data)
        total_loss_valid += len(data[0]) * \
            criterion(output, targets).cpu().item()
    total_loss_valid = total_loss_valid/len(valid_data)

    return model, total_loss_valid
