# onehot2gather bug 复现

## 运行
```
    # 执行
    python reproduce.py 

    # 清理
    python reproduce.py clean
```
## 文件说明
1. 保存包含onehot算子的网络参数
2. torch 模型转 onehot_onnx
3. 裁剪onehot_onnx, 用gather 替换 onehot
4. 新的gather onnx 转 trt
5. 对比四种结果

## 结果说明
1. torch 结果 与 onehot_onnx 结果对比，误差在e-5内
2. onehot_onnx 与 gather_onnx 对比，误差在e-5内
3. gather_onnx 与 trt 结果对比，误差如下
    Max absolute difference: 0.2328076
    Max relative difference: 83.76954
 
## 猜想
tensorrt 在进行gather算子转换时出现误差。