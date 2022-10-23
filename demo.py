# author:aqizhou
# edit time:2022/10/23 14:26


from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from data_process import InputDataSet, read_data
import time
import torch
import numpy as np
from sklearn.metrics import precision_score,f1_score,recall_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


def evaluate(model, val_dataloader):
    total_val_loss = 0
    corrects = []
    y_true = []
    y_pred = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        logits = torch.argmax(outputs.logits, dim=1)
        ## 把每个batch预测的准确率加入到一个list中
        ## 在加入之前，preds和labels变成cpu的格式
        # if not torch.cuda.is_available():
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        y_true.extend(labels_ids)
        y_pred.extend(preds)
        print(preds)
        print(labels_ids)
        print(y_true)
        print(y_pred)



        corrects.append((preds == labels_ids).mean())  ## [0.8,0.7,0.9]
        ## 返回loss
        loss = outputs.loss
        ## 把每个batch的loss加入 total_val_loss
        ## 总共有len(val_dataloader)个batch
        total_val_loss += loss.item()


    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print('Weighted precision', precision)
    print('Weighted recall', recall)
    print('Weighted f1-score',f1)



    return avg_val_loss, avg_val_acc,precision,recall,f1


def test(batch_size, EPOCHS):
    model = BertForSequenceClassification.from_pretrained("model.plt", num_labels=5)
    model.to(device)
    print(model.device)
    val = read_data('test.csv')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')

    val_dataset = InputDataSet(val, tokenizer, 64)

    val_dataloader = DataLoader(val_dataset, batch_size)

    test_loss = []
    test_acc = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.eval()  # 这里调用了eval方法
        avg_val_loss, avg_val_acc,precision,recall,f1 = evaluate(model, val_dataloader)
        val_time = time.time() - t0
        print('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f} precision={:.5f} recall={:.5f} f1={:.5f}===='.format(epoch + 1, EPOCHS, avg_val_loss,
                                                                                    avg_val_acc,precision,recall,f1))




if __name__ == '__main__':
    test(batch_size=64, EPOCHS=10)