# file: RoBERTa.py
# 这个文件现在正被创建在您的云服务器上

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

def main():
    """
    这个函数将在云服务器上执行，并自动下载和缓存模型。
    """
    model_name = 'roberta-base'
    
    print("======================================================")
    print(f"云服务器准备从 Hugging Face Hub 下载并加载 '{model_name}'")
    print("======================================================")

    try:
        # 步骤 A: 加载分词器
        # 当这行代码在服务器上首次运行时，它会:
        # 1. 连接Hugging Face
        # 2. 下载分词器文件
        # 3. 将文件保存到服务器的本地缓存目录 (例如 /home/your_username/.cache/huggingface/hub)
        print("\n>>> 正在加载分词器...")
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        print(">>> 分词器加载成功！")

        # 步骤 B: 加载模型
        # 同样，首次运行时，它会自动下载模型权重（约500MB）并缓存到服务器磁盘
        print("\n>>> 正在加载模型 (首次运行会需要几分钟时间)...")
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
        print(">>> 模型加载成功！")

        print("\n------------------------------------------------------")
        print("所有操作均在云服务器上完成。模型文件已被缓存到服务器磁盘。")
        print("------------------------------------------------------")

    except Exception as e:
        print(f"\n[错误] 操作失败: {e}")
        print("请检查您的云服务器是否有网络连接，以及磁盘空间是否足够。")

if __name__ == '__main__':
    main()
