# cuda_learning

## 代码文件说明
### 1. arrayAdd_managed.cu
 数组相加（向量加法），使用cudaMallocManaged管理内存

### 2. sumArr.cu
 数组累计求和，并为CPU、GPU的计算分别导出C接口，可使用test.py调用

### 3. matrixOp_template.cu
 矩阵运算，包含加法与乘法；矩阵元素类型、大小由模板参数控制

 运算流程：
 1. 数据复制到GPU，并将矩阵扩展到8N * 8N（填0扩充）
 2. 使用GPU计算扩展后的矩阵

## 测试环境
1. windows10 21H2, cuda11.6；i7 6700HQ 8G\*2 GTX 960M
2. 编译指令示例：nvcc.exe -O2 -arch=native .\matrixOp_template.cu -o .\matrixOp_template.exe
3. sumArr编译为dll文件：nvcc.exe -O2 -shared -arch=native .\sumArr.cu -o .\sumArr.dll

