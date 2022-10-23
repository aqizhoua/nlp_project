# author:aqizhou
# edit time:2022/10/23 9:41

from transformers import BertTokenizer,BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from data_process import InputDataSet,read_data
import time
import torch
import numpy as np
from sklearn.metrics import precision_score,f1_score,recall_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(model,val_dataloader):
    total_val_loss = 0
    corrects = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

        logits = torch.argmax(outputs.logits,dim=1)
        ## 把每个batch预测的准确率加入到一个list中
        ## 在加入之前，preds和labels变成cpu的格式
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        corrects.append((preds == labels_ids).mean())  ## [0.8,0.7,0.9]
        ## 返回loss
        loss = outputs.loss
        ## 把每个batch的loss加入 total_val_loss
        ## 总共有len(val_dataloader)个batch
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc

def test(batch_size,EPOCHS):
    model = BertForSequenceClassification.from_pretrained("chinese-bert-wwm-ext", num_labels=5)
    model.to(device)
    print(model.device)
    val = read_data('test.csv')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')

    val_dataset = InputDataSet(val, tokenizer, 64)

    val_dataloader = DataLoader(val_dataset, batch_size,shuffle=True)

    test_loss = []
    test_acc = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.eval()  # 这里调用了eval方法
        avg_val_loss, avg_val_acc = evaluate(model, val_dataloader)
        val_time = time.time() - t0
        print('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch + 1, EPOCHS, avg_val_loss,
                                                                                       avg_val_acc))

def eval():
    #net.eval()
    test_loss = 0
    correct = 0
    total = 0
    classnum = 9
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)
#精度调整
    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)
# 打印格式方便复制
    print('recall'," ".join('%s' % id for id in recall))
    print('precision'," ".join('%s' % id for id in precision))
    print('F1'," ".join('%s' % id for id in F1))
    print('accuracy',accuracy)

if __name__ == '__main__':
    test(batch_size=64, EPOCHS=10)