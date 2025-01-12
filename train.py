import json
import torch
import os
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, act2id, context_window):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = []
        for dialogue_id, sentences in data.items():
            context = []
            for sentence in sentences:
                speaker = sentence["speaker"]
                text = f"{speaker}: {sentence['sentence']}"  # 当前句子
                label = act2id[sentence["dialogue_act"]]
                
                # 拼接上下文
                input_text = "[SEP] ".join(context[-context_window:] + [text])
                self.samples.append((input_text, label))

                # 更新上下文
                context.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        hidden_size = self.bert.config.hidden_size
        self.linear = torch.nn.Linear(hidden_size, len(act2id))  # 根据目标类别调整输出层维度

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.linear(out.last_hidden_state[:, 0])  # 获取CLS向量
        return out
    
class PGD(object):
 
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
 
    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}
 
    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r
 
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()
 
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

# Tokenizer编码
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='longest',
        max_length=256,  # 根据具体情况调整最大长度
        return_tensors='pt'
    )

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels

def save_model(save_name):
    save_path = './checkpoints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model, os.path.join(save_path, save_name))

def test(test_loader):
    correct = 0
    total = 0

    for index, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    accuracy = correct / total
    print(f"正确数：{correct}，总数：{total}，test准确率：{accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    data_train_path = './data/IMCS-DAC_train.json'
    data_test_path = './data/IMCS-DAC_dev.json'
    model_path = '/data/luziyu/models/bert-base-chinese'
    epoch = 5
    device = "cuda:0"
    context_window = 3  # 上下文窗口大小

    # 定义act到id的映射
    act2id = {
        "Request-Symptom": 0,
        "Inform-Symptom": 1,
        "Request-Etiology": 2,
        "Inform-Etiology": 3,
        "Request-Basic_Information": 4,
        "Inform-Basic_Information": 5,
        "Request-Existing_Examination_and_Treatment": 6,
        "Inform-Existing_Examination_and_Treatment": 7,
        "Request-Drug_Recommendation": 8,
        "Inform-Drug_Recommendation": 9,
        "Request-Medical_Advice": 10,
        "Inform-Medical_Advice": 11,
        "Request-Precautions": 12,
        "Inform-Precautions": 13,
        "Diagnose": 14,
        "Other": 15
    }

    train_dataset = Dataset(data_train_path, act2id, context_window)
    test_dataset = Dataset(data_test_path, act2id, context_window)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )

    model = Model()
    model.to(device)
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    # accuracy = test(test_loader)

    for now_epoch in range(epoch):
        model.train()
        for (input_ids, attention_mask, token_type_ids, labels) in tqdm(train_loader, desc=f"Epoch {now_epoch + 1}/{epoch}"):
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(out, labels)
            loss.backward()

            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.processor
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                out_adv = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss_adv = criterion(out_adv, labels)
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数

            optimizer.step()
            optimizer.zero_grad()

        # 测试并记录准确率
        model.eval()
        accuracy = test(test_loader)
        scheduler.step(accuracy)  # 根据测试准确率调整学习率

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(f'best.pt')
            print(f"{now_epoch + 1}轮准确率最高，已存储")

    save_model('last.pt')