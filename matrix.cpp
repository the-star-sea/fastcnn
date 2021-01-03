#include "matrix.h"

inline void quickdot(float *x, float *y, long begin, long end, float *ans) {
#if defined(ARM)
    *ans = 0;
    for (int i = begin; i < end; i++) {
        *ans += x[i] * y[i];
    }
#elif defined(X86)
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

inline void Relu(Matrix *matrix) {
#if defined(ARM)
    for (int k = 0; k < matrix->getChannel() * matrix->getSize() * matrix->getSize(); k++) {
        if (matrix->getData()[k] < 0)matrix->getData()[k] = 0;
    }
#elif defined(X86)
#pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < matrix->getSize(); k++) {
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int i = 0; i < matrix->getChannel(); i++) {
                if (matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k] < 0)
                    matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k] = 0;
            }
        }
    }
#endif
}

inline void maxpool(const Matrix *matrix1, int size, Matrix *ans) {
#if defined(ARM)
    int pl = 0;
    for (int c = 0; c < matrix1->getChannel(); c++)
    {
        for (int si = 0; si < matrix1->getSize(); si += size) {
            for (int sj = 0; sj < matrix1->getSize(); sj += size)     {
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
#elif defined(X86)
    int pl = 0;
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < matrix1->getChannel(); c++) {
        for (int si = 0; si < matrix1->getSize(); si += size) {
            for (int sj = 0; sj < matrix1->getSize(); sj += size) {
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
#endif

}

inline void addzero(Matrix *matrix, int padding) {
#if defined(ARM)
    float *data = new float[(matrix->getSize() + padding * 2) * (matrix->getSize() + padding * 2) *
                            matrix->getChannel()];
    int pl = 0;
    for (int i = 0; i < matrix->getSize() + 2 * padding; i++) {
        data[pl++] = 0;
    }
    for (int i = 0; i < matrix->getChannel(); i++) {
        for (int j = 0; j < padding; j++) {
            data[pl++] = 0;

        }
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int k = 0; k < matrix->getSize(); k++) {
                data[pl++] = matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k];

            }

        }
        for (int j = 0; j < padding; j++) {
            data[pl++] = 0;


        }
    }
    for (int i = 0; i < matrix->getSize() + 2 * padding; i++) {
        data[pl++] = 0;
    }
    matrix->setData(data);
    matrix->setSize(matrix->getSize() + 2 * padding);
#elif defined(X86)
    float *data = new float[(matrix->getSize() + padding * 2) * (matrix->getSize() + padding * 2) *
                            matrix->getChannel()];
    int pl = 0;
    for (int i = 0; i < matrix->getSize() + 2 * padding; i++) {
        data[pl++] = 0;
    }
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < matrix->getChannel(); i++) {
        for (int j = 0; j < padding; j++) {
            data[pl++] = 0;

        }
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int k = 0; k < matrix->getSize(); k++) {
                data[pl++] = matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k];

            }

        }
        for (int j = 0; j < padding; j++) {
            data[pl++] = 0;


        }
    }
    for (int i = 0; i < matrix->getSize() + 2 * padding; i++) {
        data[pl++] = 0;
    }
    matrix->setData(data);
    matrix->setSize(matrix->getSize() + 2 * padding);
#endif
}

inline void convolution(const Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias, int anschannel) {
    ans = new Matrix((matrix1->getSize() + 1 - matrix2.getSize()) / stride, anschannel,
                     new float[anschannel * ((matrix1->getSize() + 1 - matrix2.getSize()) / stride) *
                               ((matrix1->getSize() + 1 - matrix2.getSize()) / stride)]);
#if defined(ARM)
    int  pl=0, si, sj,sc;

    for( sc=0;sc<ans->getChannel();sc++){
        for (si = 0; si < matrix1->getSize()+1-matrix2.getSize(); si +=stride)
        {
            for (sj = 0; sj < matrix1->getSize()+1-matrix2.getSize(); sj +=stride) {


                blockdot(sc,matrix1, matrix2, si, sj, *ans, pl);ans->getData()[pl]+=bias[sc];
                pl++;
            }

        }}



#elif defined(X86)
    int pl = 0, si, sj, sc;
#pragma omp parallel for schedule(dynamic)
    for (sc = 0; sc < ans->getChannel(); sc++) {
        for (si = 0; si < matrix1->getSize() + 1 - matrix2.getSize(); si += stride) {
            for (sj = 0; sj < matrix1->getSize() + 1 - matrix2.getSize(); sj += stride) {


                blockdot(sc, matrix1, matrix2, si, sj, *ans, pl);
                ans->getData()[pl] += bias[sc];
                pl++;
            }

        }
    }

#endif
}

inline void blockdot(int sc, const Matrix *matrix1, Matrix matrix2, int si, int sj, Matrix ans, int pl) {
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
#elif defined(X86)
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
                inner_prod += matrix1->getData()[k + i * matrix1->getColumn()] *
                              matrix2.getData()[k + j * matrix1->getColumn()];
            }
            ans.getData()[j + i * matrix2.getColumn()] += inner_prod;
        }

    }
#endif
}

inline void Matrix::setData(float *data) {
    Matrix::data = data;
}

inline Matrix::Matrix() {}


inline Matrix::~Matrix() {
    if (cnt == 1) {
        delete[] Matrix::data;
    }
}


inline unsigned int Matrix::getChannel() const {
    return channel;
}

inline void Matrix::setChannel(unsigned int channel) {
    Matrix::channel = channel;
}

inline Matrix::Matrix(unsigned int size, unsigned int channel, float *data) : size(size), channel(channel),
                                                                              data(data) {}

inline unsigned int Matrix::getSize() const {
    return size;
}

inline void Matrix::setSize(unsigned int size) {
    Matrix::size = size;
}

