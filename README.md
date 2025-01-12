该代码包含train和eval两个部分。

需另外下载数据集放在 ./data/IMCS-DAC_train.json 和 ./data/IMCS-DAC_dev.json 作为训练集和验证集；下载bert-base-chinese并修改model_path 指向模型位置。

context_window是可调整的滑动窗口参数，可以尝试调整比较效果。