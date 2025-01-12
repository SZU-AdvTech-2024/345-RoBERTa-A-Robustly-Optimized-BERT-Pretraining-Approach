import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 定义 Model 类
class Model(torch.nn.Module):
    def __init__(self, model_path, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.linear(out.last_hidden_state[:, 0])  # 获取 CLS 向量
        return out

# 加载模型
def load_model(checkpoint_path, device):
    model = torch.load(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model

# 评估函数
def evaluate(input_path, output_path, model, tokenizer, act2id, device, context_window):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    total_sentences = sum(len(sentences) for sentences in data.values())  # 总句子数

    with tqdm(total=total_sentences, desc="Evaluating", unit="sentence") as pbar:
        for dialogue_id, sentences in data.items():
            results[dialogue_id] = []
            context = []

            for sentence in sentences:
                speaker = sentence["speaker"]
                text = f"{speaker}: {sentence['sentence']}"

                # 拼接上下文
                input_text = "[SEP] ".join(context[-context_window:] + [text])

                # Tokenize the input
                encoded = tokenizer.encode_plus(
                    input_text,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt',
                    padding='longest'
                )

                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                token_type_ids = encoded['token_type_ids'].to(device)

                # Get model predictions
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    predicted_label = torch.argmax(logits, dim=1).item()

                # Map back to dialogue act
                dialogue_act = [k for k, v in act2id.items() if v == predicted_label][0]

                # Add predictions to results
                sentence["dialogue_act"] = dialogue_act
                results[dialogue_id].append(sentence)

                # 更新上下文
                context.append(text)

                pbar.update(1)  # 更新进度条

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    device = 'cuda:0'  # 使用的设备
    input_path = './data/IMCS-DAC_test.json'  # 输入文件路径
    output_path = './result.json'  # 输出文件路径
    model_path = '/data/luziyu/models/bert-base-chinese'  # 预训练模型路径
    checkpoint_path = './checkpoints/best.pt'  # 模型文件路径

    # 定义 act2id
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

    # 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 加载模型
    model = load_model(checkpoint_path, device)

    # 运行评估
    evaluate(input_path, output_path, model, tokenizer, act2id, device, context_window=3)