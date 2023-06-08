import torch
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from kobert_tokenizer import KoBERTTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
from torch import nn
from sklearn.metrics import f1_score
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

date = '0412' # 생성파일 명


def get_new_output(out):
    out_neu = torch.round(out[:, 0] - 0.1206).unsqueeze(1)
    out_hap = torch.round(out[:, 1] + 0.3615).unsqueeze(1)
    out_sur = torch.round(out[:, 2] + 0.2585).unsqueeze(1)
    out_ang = torch.round(out[:, 3] + 0.4741).unsqueeze(1)
    out_dis = torch.round(out[:, 4] + 0.4857).unsqueeze(1)
    out_sad = torch.round(out[:, 5] + 0.4789).unsqueeze(1)
    out_fea = torch.round(out[:, 6] + 0.4869).unsqueeze(1)
    outputs = torch.cat((out_neu, out_hap, out_sur, out_ang, out_dis, out_sad, out_fea), 1)
    return outputs

def get_accuracy(preds, labels): # 정확도 계산
    """
    Args:
        preds (logit): (batch)
        labels (label): (batch)
    Returns:
        ()
    """
    
    pred = np.round(preds)
    score = np.sum(pred == labels) / len(labels)
    return score

def get_cls_f1(output, target):
    """
    Args:
        output (logit): (batch)
        target (label): (batch)
    Returns:
        ()
    """
    output = np.round(output)
    score = f1_score(target, output, average='macro')
    
    return score

def format_time(elapsed):
    elapsed_rounded = int(np.round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def loss_plot_1(style, name, title, n_epochs):
    ax = plt.figure().gca()
    ax.plot(style)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xlim(-1, n_epochs+1)
    plt.title(title)
    plt.savefig(name)
    

def train_model(model, train_dataloader, test_dataloader, n_epochs):

    # pre-defined settings
    criterion = nn.BCEWithLogitsLoss(reduction='mean').to(device)
    optimizer = AdamW(model.parameters(), lr = 5e-3, eps = 1e-8)
    total_steps = len(train_dataloader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    ts = ts.replace(':', '-')

    file = open(f'./result/result_{date}.txt', 'w')
    train_loss = []
    
    for epoch_i in range(0, n_epochs):
        
        # ==============Training==================
        
        print("")
        print(f'======== Epoch {epoch_i+1} /{n_epochs} ========')
        print(' :: Training Process :: ')
        print("", file=file)
        print(f'======== Epoch {epoch_i+1} /{n_epochs} ========', file=file)
        print(' :: Training Process :: ', file=file)


        # 시작 시간 설정
        t0 = time.time()

        #  initialization loss
        total_loss = 0

        model.train()

        nb_train_steps = 0
            
        for step, (token_ids, segment_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
            
            # 경과 정보 표시
            if step % 500 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            token_ids = token_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.float().to(device)

            # model
            out = model(token_ids, segment_ids, attention_mask) # (batch, 7)
            acc_out = torch.sigmoid(out)
            
            loss = criterion(out, label)
            total_loss += loss.item()

            acc_out = acc_out.clone().detach().cpu().numpy() # (batch, 7)
            label_ids = label.clone().detach().cpu().numpy() # (batch, 7)

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            nb_train_steps += 1

        # Train 과정 loss plot
        avg_train_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_train_loss)
        loss_plot_1(train_loss,f'./loss_plot/train_loss_{date}.png', 'train', n_epochs)

        print("")
        print(f"  Average training loss: {avg_train_loss:.3f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        print("", file=file)
        print(f"  Average training loss: {avg_train_loss:.3f}", file=file)
        print(f"  Training epoch took: {format_time(time.time() - t0)}", file=file)
        print(f" Validation took: {format_time(time.time() - t0)}")

        # ==============Validation================

        print("")
        print(" :: Validation Process :: ")

        t0 = time.time()

        model.eval()

        # F1 변수 초기화
        eval_f1_neu = 0
        eval_f1_hap = 0
        eval_f1_sur = 0
        eval_f1_ang = 0
        eval_f1_dis = 0
        eval_f1_sad = 0
        eval_f1_fea = 0

        nb_eval_steps = 0

        for (token_ids, segment_ids, attention_mask, label) in tqdm(test_dataloader):
            # data to gpu
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            label = label.float().to(device)            

            with torch.no_grad():     
                out = model(token_ids, segment_ids, attention_mask)
                out = torch.sigmoid(out)

            loss = criterion(out, label)
            acc_out = out.clone().detach().cpu().numpy()
            label_ids = label.clone().detach().cpu().numpy()
            
            n_out = get_new_output(out)
            n_out = n_out.clone().detach().cpu().numpy()
            
            # # label 별 output & label
            out_neu, label_neu = acc_out[:,0], label_ids[:,0]
            out_hap, label_hap = acc_out[:,1], label_ids[:,1]
            out_sur, label_sur = acc_out[:,2], label_ids[:,2]
            out_ang, label_ang = acc_out[:,3], label_ids[:,3]
            out_dis, label_dis = acc_out[:,4], label_ids[:,4]
            out_sad, label_sad = acc_out[:,5], label_ids[:,5]
            out_fea, label_fea = acc_out[:,6], label_ids[:,6]
            

            # label 별 f1 score
            eval_f1_score_neu = get_cls_f1(out_neu, label_neu)
            eval_f1_score_hap = get_cls_f1(out_hap, label_hap)
            eval_f1_score_sur = get_cls_f1(out_sur, label_sur)
            eval_f1_score_ang = get_cls_f1(out_ang, label_ang)
            eval_f1_score_dis = get_cls_f1(out_dis, label_dis)
            eval_f1_score_sad = get_cls_f1(out_sad, label_sad)
            eval_f1_score_fea = get_cls_f1(out_fea, label_fea)
            
            
            eval_f1_neu += eval_f1_score_neu
            eval_f1_hap += eval_f1_score_hap
            eval_f1_sur += eval_f1_score_sur
            eval_f1_ang += eval_f1_score_ang
            eval_f1_dis += eval_f1_score_dis
            eval_f1_sad += eval_f1_score_sad
            eval_f1_fea += eval_f1_score_fea

            nb_eval_steps += 1
        
        f1_neu = eval_f1_neu/nb_eval_steps
        f1_hap = eval_f1_hap/nb_eval_steps
        f1_sur = eval_f1_sur/nb_eval_steps
        f1_ang = eval_f1_ang/nb_eval_steps
        f1_dis = eval_f1_dis/nb_eval_steps
        f1_sad = eval_f1_sad/nb_eval_steps
        f1_fea = eval_f1_fea/nb_eval_steps
        avg_f1 = (f1_neu+f1_hap+f1_sur+f1_ang+f1_dis+f1_sad+f1_fea)/7
        
        print("")
        print("---------------------------------------------------------------------")
        print(f" NEUTRAL(중립) F1: {f1_neu:.3f}, HAPPY(행복) F1: {f1_hap:.3f}, SURPRISE(놀람) F1: {f1_sur:.3f}, ANGER(분노) F1: {f1_ang:.3f}")
        print(f" DISGUST(혐오) F1: {f1_dis:.3f}, SAD(슬픔) F1: {f1_sad:.3f}, FEAR(공포) F1: {f1_fea:.3f}")
        print(f" Average F1 Score: {avg_f1:.3f}")
        
        print("")
        print(f" Validation took: {format_time(time.time() - t0)}")

        print("", file=file)
        print("---------------------------------------------------------------------", file=file)
        print(f" NEUTRAL(중립) F1: {f1_neu:.3f}, HAPPY(행복) F1: {f1_hap:.3f}, SURPRISE(놀람) F1: {f1_sur:.3f}, ANGER(분노) F1: {f1_ang:.3f}",file=file)
        print(f" DISGUST(혐오) F1: {f1_dis:.3f}, SAD(슬픔) F1: {f1_sad:.3f}, FEAR(공포) F1: {f1_fea:.3f}",file=file)
        print(f" Average F1 Score: {avg_f1:.3f}", file=file)

        torch.save(model.state_dict(),f'./saved/23{date}/23{date}_ep' + str(epoch_i) + '_model_save.pt')

    print("")
    print(" -- Training complete -- ")
    print("", file=file)
    print(" -- Training complete -- ", file=file)
