# ExpressionRecognition
## 表情识别是指从静态照片或视频序列中选择出表情状态，从而确定对人物的情绪与心理变化。给定人脸照片完成具体的情绪识别，选手需要根据训练集数据构建情绪识别任务，并对测试集图像进行预测，识别人脸的7种情绪。详见http://challenge.xfyun.cn/topic/info?type=facial-emotion-recognition
## 对数据进行分析后发现：
### 有很多噪音
### 标签有误分类的情况
### 样本不均衡，disgust少于其它类
### 类间差别不大，比如angry，sad和disgust

## 可选方案
### focal loss或者weighted loss
### 交叉检验
### 数据增强，cutout和cutmix（拼接后要改相应的标签，对每个标签算损失，然后再乘上混合的比例）
### label smoothing缓解训练标签错误和类间差别不大的情况
### mutual learning，三种模型互相学习

## 尝试过的
### swintransformer和vit，感觉更适合适合图像精度比较高的
### 直方图均衡化没什么用

## 大佬提供的进一步思路
### ferplus做数据清理
### 使用out of distribution detection相关技术来清理数据
### 使用诸如cleanlab之类的数据清理工具进行清理
### 引入一些无监督的技巧来辅助清理数据UMAP库
