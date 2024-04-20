# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月25日
"""
import torch


def test(model,device, loader, args):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            pro_data1 = data[0].to(device)
            pro_data2 = data[1].to(device)
            output,_ ,_ = model(pro_data1, pro_data2,device)

            if args.datasetname == 'multi_class':
                predicted_values = torch.softmax(output, dim = 1)
                predicted_labels = torch.argmax(predicted_values, dim = 1)
            else:
                predicted_values = torch.sigmoid(output)
                predicted_labels = torch.round(predicted_values)

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # predicted values
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # predicted labels
            total_true_labels = torch.cat((total_true_labels, pro_data1.y.view(-1, 1).cpu()), 0)  # ground truth

    return total_true_labels.numpy().flatten(), total_pred_values.numpy(), total_pred_labels.numpy().flatten() #total_pred_values.numpy().flatten()多分类不用flatten


def train(model, train_loader, device, optimizer, criterion, args):
    model.train()
    total_loss = 0
    n_batches = 0
    correct = 0
    for batch_idx, data in enumerate(train_loader):
        pro_data1 = data[0].to(device)
        pro_data2 = data[1].to(device)
        optimizer.zero_grad()
        y_pred, cl_loss, att_loss = model(pro_data1, pro_data2, device)

        if args.datasetname == 'multi_class':
            loss = criterion(y_pred, pro_data1.y.long().to(device), )
            y_pred = torch.argmax(torch.softmax(y_pred, dim = 1),dim = 1)
        else:
            loss = criterion(y_pred, pro_data1.y.view(-1, 1).float().to(device), )
            y_pred = torch.round(torch.sigmoid(y_pred).squeeze(1))
        correct += torch.eq(y_pred, pro_data1.y).data.sum()
        loss += att_loss
        loss += (cl_loss / 20).to('cpu')
        total_loss += loss.data
        loss.backward()
        optimizer.step()
        n_batches += 1
    avg_loss = total_loss / n_batches
    acc = correct.cpu().numpy() / (len(train_loader.dataset))

    print("train avg_loss is", avg_loss)
    print("train ACC = ", acc)

    return avg_loss, acc
    # return att_loss

