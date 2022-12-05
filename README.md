# Fast Face Detection
Tong Zhang

## Function
-  Efficient and fast face recognition
    
-  Each function of the convolution layer is parameter tunable，such as padding
    
-  Optimization by matrix multiplication, compilation optimization and various other means.

-  Arm and X86 support.

## Usage
1. Modify CMakeLists.txt according to comments


![image.png](https://i.loli.net/2021/01/04/1X3RAOBag2utQ95.png)


2. Input path


![image.png](https://i.loli.net/2021/01/04/HuYpAms6bRDiQtj.png)
 
## Main Function
```
void addzero(Matrix *matrix, int padding);
void convolution(Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias, int anschannel);
void maxpool(const Matrix *matrix1, int size, Matrix *ans);
void Relu(Matrix *matrix);
```
### Convolution Layer
1. Transfer into Matrix
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
2. Multiply the Matrix and add bias.
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
### Relu
Change the order of i,j,k
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
## Optimization
- In the convolution operation, the kernel is regarded as a window. Every time the window is moved, nine numbers are multiplied by it, and the nine numbers are straightened into one row, and the row-major order is used to make access to memory faster.
- Optimize matrix multiplication by AVX, OMP, block（cache locality） (OMP basically does not work. It may be caused by the fact that the amount of data is too small)


3. The order of loop i, j, k is reversed , so as to make the pointer jump as small as possible

4. Variables are transferred using pointers

5. Optimized by compiler.
## Evaluation
![face.jpg](https://i.loli.net/2021/01/04/hn2Ui3IqsltY1wg.jpg)
1. Before optimization


![image.png](https://i.loli.net/2021/01/04/GuKNkaqDhUtWBjr.png)
2. After optimization


![image.png](https://i.loli.net/2021/01/04/jYKeWiNaMq892T3.png)
3.ARM


![fc2cd12b13b923f1ad05531e481fbf0.jpg](https://i.loli.net/2021/01/04/Ym8JQscDC3vFxRA.jpg)





