# CNN
姓名：张通

ID：11911611

## 代码功能
1.  高效，快速辨别人脸
    
2.  卷积层的各函数均可调整参数，如padding
    
3.  通过矩阵乘法，编译优化等各类手段优化

4.  可跨平台运行

##代码展示
https://github.com/haha-stone/cnn
##使用说明
1. 按注释修改CMakeLists.txt
![image.png](https://i.loli.net/2021/01/04/1X3RAOBag2utQ95.png)
2.  输入path
![image.png](https://i.loli.net/2021/01/04/HuYpAms6bRDiQtj.png)
 
##主要函数
```
void addzero(Matrix *matrix, int padding);
void convolution(Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias, int anschannel);
void maxpool(const Matrix *matrix1, int size, Matrix *ans);
void Relu(Matrix *matrix);
```
### convolution
1. 转化为矩阵
```
#pragma omp parallel for schedule(dynamic)
  for (si = 0; si < matrix1->getSize() + 1 - matrix2.getSize(); si += stride) {
        for (sj = 0; sj < matrix1->getSize() + 1 - matrix2.getSize(); sj += stride) {
            for (int c = 0; c < matrix1->getChannel(); c++) {
                for (int i = 0; i < matrix2.getSize(); i++) {
                    for (int j = 0; j < matrix2.getSize(); j++) {

                        data[pl++] = matrix1->getData()[c * matrix1->getSize() * matrix1->getSize() +
                                                        (si + i) * matrix1->getSize() + sj + j];

                    }
                }
            }
        }

    }

```
2. 矩阵相乘并加上bias
```
    Matrix temp = matrix2 * (*matrix1);
    ans->setData(temp.getData());
    pl = 0;
    for (sc = 0; sc < ans->getChannel(); sc++) {
        for (si = 0; si < size * size; si++) {
            temp.getData()[pl++] += bias[sc];

        }
    }
```
### relu
调换了i,j,k的次序
```
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < matrix->getSize(); k++) {
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int i = 0; i < matrix->getChannel(); i++) {
                if (matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k] < 0)
                    matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k] = 0;
            }
        }
    }
```
##优化
1. 在卷积运算中，将kernel视为一个窗口，该窗口每移动一次就会得到9个数与其相乘，将这九个数拉直为一行，并使用行主序，使得访问内存变快

2. 使用矩阵乘法的优化，包括avx,omp,分块(omp基本没起到作用，怀疑是数据量太小)

3. 各处循环i,j,k的顺序调换，尽量使得指针跳跃变小

4. 变量使用指针传输

5. 编译器优化
## 测试
![face.jpg](https://i.loli.net/2021/01/04/hn2Ui3IqsltY1wg.jpg)
优化前
![image.png](https://i.loli.net/2021/01/04/GuKNkaqDhUtWBjr.png)
优化后
![image.png](https://i.loli.net/2021/01/04/jYKeWiNaMq892T3.png)
![fc2cd12b13b923f1ad05531e481fbf0.jpg](https://i.loli.net/2021/01/04/Ym8JQscDC3vFxRA.jpg)
##遇到的问题
1. opencv的使用

一开始是在mac上使用clion,由于openmp和avx需用gcc,而mac上编译opencv用的是clang不得不转到windows平台


## 结语
这个学期在c++课程上收益良多，从最开始的一无所知，到现在可以写一些代码，见识到了很多新鲜的东西，arm开发板之类的，非常感谢于老师和学助们一学期的付出，下学期数据库再见！



