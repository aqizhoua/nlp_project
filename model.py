# author:aqizhou
# edit time:2022/10/23 13:14

# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : predict.py
# @Project: Douban_Bert
# @CreateTime : 2022/3/13 上午12:08:22
# @Version：V 0.1
'''
模型训练和评估
'''
import numpy as np
from torch import nn
import time
import os
import torch
import logging

from torch.autograd import Variable
from torch.optim import AdamW
from transformers import BertTokenizer,BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.utils.notebook import format_time
from data_process import InputDataSet,read_data
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(batch_size,EPOCHS):
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=5)

    # model = BertForSequenceClassification.from_pretrained("chinese-bert-wwm", num_labels=5)

    model = BertForSequenceClassification.from_pretrained("raw_cash", num_labels=5)

    model.to(device)
    train = read_data('train.csv')
    val = read_data('test.csv')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')

    train_dataset = InputDataSet(train, tokenizer, 64)
    val_dataset = InputDataSet(val, tokenizer, 64)

    train_dataloader = DataLoader(train_dataset,batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_steps = len(train_dataloader) * EPOCHS  # len(dataset)*epochs / batchsize
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_t0 = time.time()

    log = log_creater(output_dir='./cache_last/logs/')

    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")

    train_loss = []
    test_loss = []
    test_acc = []
    test_prec = []
    test_rec = []
    test_f1 = []


    for epoch in range(EPOCHS):
        total_train_loss = 0
        t0 = time.time()
        model.train()
        for step, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()

            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_time = format_time(time.time() - t0)

        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch+1,EPOCHS,avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')

        model.eval()


        avg_val_loss, avg_val_acc, precision, recall, f1 = evaluate(model, val_dataloader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f} precision={:.5f} recall={:.5f} f1={:.5f}===='.format(
                epoch + 1, EPOCHS, avg_val_loss,
                avg_val_acc, precision, recall, f1))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

        if epoch == EPOCHS-1:#保存模型



            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained('model')
            print('Model Saved!')


        #将数据保存到列表
        train_loss.append(avg_train_loss)
        test_loss.append(avg_val_loss)
        test_acc.append(avg_val_acc)
        test_prec.append(precision)
        test_rec.append(recall)
        test_f1.append(f1)




    log.info('')
    log.info('   Training Completed!')
    print('Total training took {:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))
    #简单可视化
    x1 = range(0,EPOCHS)
    y1 = train_loss
    plt.plot(x1, y1)
    plt.title("train loss of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("train loss")
    plt.savefig('./cache_last/wwmext_1.png')
    plt.close()
    # plt.show()

    x2 = range(0,EPOCHS)
    y2 = test_loss
    plt.plot(x2, y2)
    plt.title("test loss of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("test loss")
    plt.savefig('./cache_last/wwmext_2.png')
    plt.close()
    # plt.show()

    x3 = range(0,EPOCHS)
    y3 = test_acc
    plt.plot(x3, y3)
    plt.title("test acc of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("test acc")
    plt.savefig('./cache_last/wwmext_3.png')
    plt.close()
    # plt.show()

    x3 = range(0,EPOCHS)
    y3 = test_acc
    plt.plot(x3, y3)
    plt.title("test precision of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("test precision")
    plt.savefig('./cache_last/wwmext_4.png')
    plt.close()
    # plt.show()

    x3 = range(0,EPOCHS)
    y3 = test_acc
    plt.plot(x3, y3)
    plt.title("test recall of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("test recall")
    plt.savefig('./cache_last/wwmext_5.png')
    plt.close()
    # plt.show()

    x3 = range(0,EPOCHS)
    y3 = test_acc
    plt.plot(x3, y3)
    plt.title("test f1_score of chinese-bert-wwm-ext")
    plt.xlabel("epoches")
    plt.ylabel("test f1")
    plt.savefig('./cache_last/wwmext_6.png')
    plt.close()
    # plt.show()






def evaluate(model,val_dataloader):
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
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

        logits = torch.argmax(outputs.logits,dim=1)
        ## 把每个batch预测的准确率加入到一个list中
        ## 在加入之前，preds和labels变成cpu的格式
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        corrects.append((preds == labels_ids).mean())  ## [0.8,0.7,0.9]
        y_true.extend(labels_ids)
        y_pred.extend(preds)
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

    # print('Weighted precision', precision)
    # print('Weighted recall', recall)
    # print('Weighted f1-score',f1)



    return avg_val_loss, avg_val_acc,precision,recall,f1




#训练日志，复用性很高的代码
def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


if __name__ == '__main__':
    train(batch_size=64,EPOCHS=5)

