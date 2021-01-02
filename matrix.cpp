#include "matrix.h"

inline void quickdot(const float *x, const float *y, long begin, long end, float *ans) {
#if defined(ARM)
    *ans = 0;
    for (int i = begin; i < end; i++) {
        *ans += x[i] * y[i];
    }
#elif defined(_ENABLE_AVX2)
    float inner_prod = 0.0f;
    __m256 X, Y;
    __m256 acc = _mm256_setzero_ps();
    float temp[8];

    long i;
    for (i = begin; i + 8 < end; i += 8) {
        X = _mm256_loadu_ps(x + i);
        Y = _mm256_loadu_ps(y + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));
    }
    _mm256_storeu_ps(&temp[0], acc);
    inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                 temp[6] + temp[7];
    for (; i < end; ++i) {
        inner_prod += x[i] * y[i];
    }
    *ans = inner_prod;
#endif
}

void Relu(Matrix *matrix) {
#if defined(ARM)
    for (int k = 0; k < matrix->getChannel() * matrix->getSize() * matrix->getSize(); k++) {
        if (matrix->getData()[k] < 0)matrix->getData()[k] = 0;
    }
#elif defined(_ENABLE_AVX2)
#pragma omp parallel for schedule(dynamic)
    for(int k=0;k<matrix->getSize();k++)
    {
       for(int j=0;j<matrix->getSize();j++){
           for(int i=0;i<matrix->getChannel();i++){
             if(matrix->getData()[i*matrix->getSize()*matrix->getSize()+j*matrix->getSize()+k]<0)matrix->getData()[i*matrix->getSize()*matrix->getSize()+j*matrix->getSize()+k]=0;
         }
       }
    }
#endif
}

void maxpool(const Matrix *matrix1, int size, Matrix *ans) {
#if defined(ARM)
    int pl = 0;
    for (int sj = 0; sj < matrix1->getSize(); sj += size) {
        for (int si = 0; si < matrix1->getSize(); si += size) {
            for (int c = 0; c < matrix1->getChannel(); c++) {
                ans->getData()[pl] = 0;
                for (int j = 0; j < size; j++) {
                    for (int i = 0; i < size; i++) {
                        ans->getData()[pl] = max(ans->getData()[pl],
                                                 matrix1->getData()[c * matrix1->getSize() * matrix1->getSize() +
                                                                    (si + i) * matrix1->getSize() + sj + j]);
                    }
                }
                pl++;
            }
        }
    }
#elif defined(_ENABLE_AVX2)
    int pl=0;
#pragma omp parallel for schedule(dynamic)
    for(int sj=0;sj<matrix1->getSize();sj+=size)
    {
        for(int si=0;si<matrix1->getSize();si+=size){
            for(int c=0;c<matrix1->getChannel();c++){
                ans->getData()[pl]=0;
                for(int j=0;j<size;j++){
                    for(int i=0;i<size;i++){
                        ans->getData()[pl]=max(ans->getData()[pl],matrix1->getData()[c*matrix1->getSize()*matrix1->getSize()+(si+i)*matrix1->getSize()+sj+j]);
                    }
                }
                pl++;
            }
        }
    }
#endif

}

void convolution(const Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias) {
#if defined(ARM)
    int pl = 0, si, sj, sc;
    for (sj = 0; sj < matrix1->getSize() + 1 - matrix2.getSize(); sj += stride) {
        for (si = 0; si < matrix1->getSize() + 1 - matrix2.getSize(); si += stride) {
            for (sc = 0; sc < ans->getChannel(); sc++) {

                blockdot(sc, matrix1, matrix2, si, sj, *ans, pl);
                ans->getData()[pl] += bias[sc];
                pl++;
            }

        }
    }


#elif defined(_ENABLE_AVX2)
    int  pl=0, si, sj,sc;
#pragma omp parallel for schedule(dynamic)
    for (sj = 0; sj < matrix1->getSize()+1-matrix2.getSize(); sj +=stride)
    {
    for (si = 0; si < matrix1->getSize()+1-matrix2.getSize(); si +=stride){
        for( sc=0;sc<ans->getChannel();sc++) {

            blockdot(sc,matrix1, matrix2, si, sj, *ans, pl);ans->getData()[pl]+=bias[sc];
pl++;
            }

        }}

#endif
}

void blockdot(int sc, const Matrix *matrix1, Matrix matrix2, int si, int sj, Matrix ans, int pl) {
#if defined(ARM)
    ans.getData()[pl] = 0;
    for (int j = 0; j < matrix2.getSize(); j++) {
        for (int i = 0; i < matrix2.getSize(); i++) {
            for (int c = 0; c < matrix1->getChannel(); c++) {
                ans.getData()[pl] +=
                        matrix2.getData()[sc * matrix2.getSize() * matrix2.getSize() * matrix2.getChannel() +
                                          c * matrix2.getSize() * matrix2.getSize() + i * matrix2.getSize() + j] *
                        matrix1->getData()[c * matrix1->getSize() * matrix1->getSize() + (si + i) * matrix1->getSize() +
                                           sj + j];
            }
        }
    }
#elif defined(_ENABLE_AVX2)
    for (int i = si; i < m; i++) {
        for (int j = sj; j < n; j++) {
            __m256 acc = _mm256_setzero_ps();
            float temp[8];
            float inner_prod;
            int k;
            for (k = sk; k + 8 < p; k += 8) {

                acc = _mm256_add_ps(acc,
                                    _mm256_mul_ps(_mm256_loadu_ps(matrix1->getData() + k + i * matrix1->getColumn()),
                                                  _mm256_loadu_ps(matrix2.getData() + k + j * matrix1->getColumn())));
            }
            _mm256_storeu_ps(&temp[0], acc);
            inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                         temp[6] + temp[7] + temp[8];
            for (; k < p; k++) {
                inner_prod += matrix1->getData()[k + i * matrix1->getColumn()] * matrix2.getData()[k + j * matrix1->getColumn()];
            }
            ans.getData()[j + i * matrix2.getColumn()] += inner_prod;
        }

    }
#endif
}

void Matrix::setData(float *data) {
    Matrix::data = data;
}

Matrix::Matrix() {}


Matrix::~Matrix() {
    if (cnt == 1) {
        delete[] Matrix::data;
    }
}


unsigned int Matrix::getChannel() const {
    return channel;
}

void Matrix::setChannel(unsigned int channel) {
    Matrix::channel = channel;
}

Matrix::Matrix(unsigned int size, unsigned int channel, float *data) : size(size), channel(channel), data(data) {}

unsigned int Matrix::getSize() const {
    return size;
}

void Matrix::setSize(unsigned int size) {
    Matrix::size = size;
}

