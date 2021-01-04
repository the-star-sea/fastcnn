//
// Created by 张通 on 2021/1/2.
//

#ifndef CNN_MATRIX_H
#define CNN_MATRIX_H
#define BLOCK 256
#define X86 //Please enable it if X64 CPU
//#define ARM //Please enable it if ARM CPU
#if  defined(X86)

#include <immintrin.h>

#endif

#include "data.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#include <memory.h>

using namespace std;

class Matrix {
private:
    unsigned int size, channel, cnt = 1, column, row;
    float *data;
public:
    Matrix(unsigned int column, unsigned int row);

    unsigned int getColumn() const;

    void setColumn(unsigned int column);

    unsigned int getRow() const;

    void setRow(unsigned int row);

    float *getData() const;

    Matrix(unsigned int size, unsigned int channel, float *data);

    void setData(float *data);

    void setChannel(unsigned int channel);

    unsigned int getChannel() const;

    unsigned int getSize() const;

    void setSize(unsigned int size);

    Matrix(Matrix const &matrix);

    Matrix();

    void operator=(Matrix const &temp);

    Matrix operator*(Matrix temp) const;

    ~Matrix();
};

void blockdot(int sc, const Matrix *matrix1, Matrix matrix2, int si, int sj, Matrix ans, int pl);

void convolution(Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias, int anschannel);

void matrixmatrix(const Matrix *matrix1, Matrix matrix2, Matrix *ans);


void doblock(const Matrix *matrix1, Matrix matrix2, int si, int sj, int sk, int m, int n, int p, Matrix ans);


void maxpool(const Matrix *matrix1, int size, Matrix *ans);

void Relu(Matrix *matrix);

void quickdot(float *x, float *y, long xbegin, long length, float *ans);

void addzero(Matrix *matrix, int padding);

Matrix Matrix::operator*(Matrix matrix2) const {
    float *an = new float[getRow() * matrix2.getColumn()];
    memset(an, 0, sizeof(float) * getRow() * matrix2.getColumn());
    Matrix *ans = new Matrix(matrix2.getColumn(), getRow(), an);
    matrixmatrix(this, matrix2, ans);

    return *ans;
}

inline void quickdot(float *x, float *y, long xbegin, long length, float *ans) {
#if defined(ARM)
    *ans = 0;
    for (int i = 0; i < length; i++) {
        *ans += x[i + xbegin] * y[i];
    }
#elif defined(X86)
    float inner_prod = 0.0f;
    __m256 X, Y;
    __m256 acc = _mm256_setzero_ps();
    float temp[8];

    long i;
    for (i = 0; i + 8 < length; i += 8) {
        X = _mm256_loadu_ps(x + i + xbegin);
        Y = _mm256_loadu_ps(y + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(X, Y));
    }
    _mm256_storeu_ps(&temp[0], acc);
    inner_prod = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] +
                 temp[6] + temp[7];
    for (; i < length; ++i) {
        inner_prod += x[i + xbegin] * y[i];
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
    ans->setSize(matrix1->getSize() / 2);
    ans->setChannel(matrix1->getChannel());
    ans->setData(new float[ans->getChannel() * ans->getSize() * ans->getSize()]);
#if defined(ARM)
    int pl = 0;
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
    for (int i = 0; i < matrix->getChannel(); i++) {
        for (int j = 0; j < (matrix->getSize() + 2 * padding) * padding; j++) {
            data[pl++] = 0;
        }
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int k = 0; k < padding; k++) {
                data[pl++] = 0;

            }
            for (int k = 0; k < matrix->getSize(); k++) {
                data[pl++] = matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k];

            }
            for (int k = 0; k < padding; k++) {
                data[pl++] = 0;


            }
        }

        for (int j = 0; j < (matrix->getSize() + 2 * padding) * padding; j++) {
            data[pl++] = 0;
        }
    }

    matrix->setData(data);
    matrix->setSize(matrix->getSize() + 2 * padding);
#elif defined(X86)
    float *data = new float[(matrix->getSize() + padding * 2) * (matrix->getSize() + padding * 2) *
                            matrix->getChannel()];
    int pl = 0;
    for (int i = 0; i < matrix->getChannel(); i++) {
        for (int j = 0; j < (matrix->getSize() + 2 * padding) * padding; j++) {
            data[pl++] = 0;
        }
        for (int j = 0; j < matrix->getSize(); j++) {
            for (int k = 0; k < padding; k++) {
                data[pl++] = 0;

            }
            for (int k = 0; k < matrix->getSize(); k++) {
                data[pl++] = matrix->getData()[i * matrix->getSize() * matrix->getSize() + j * matrix->getSize() + k];

            }
            for (int k = 0; k < padding; k++) {
                data[pl++] = 0;


            }
        }

        for (int j = 0; j < (matrix->getSize() + 2 * padding) * padding; j++) {
            data[pl++] = 0;
        }
    }

    matrix->setData(data);
    matrix->setSize(matrix->getSize() + 2 * padding);


#endif
}

inline void convolution(Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias, int anschannel) {
    float s = (matrix1->getSize() + 1 - matrix2.getSize());
    int size = ceil(s / stride);
    ans->setSize(size);
    ans->setChannel(anschannel);
#if defined(ARM)
    int pl = 0, si, sj, sc;
    ans->setData(new float[anschannel * size * size]);
    for (sc = 0; sc < ans->getChannel(); sc++) {
        for (si = 0; si < s; si += stride) {
            for (sj = 0; sj < s; sj += stride) {

                blockdot(sc, matrix1, matrix2, si, sj, *ans, pl);
                ans->getData()[pl] += bias[sc];
                pl++;
            }

        }
    }


#elif defined(X86)
    int pl = 0, si, sj, sc;
    matrix2.setColumn(matrix2.getSize() * matrix2.getSize() * matrix2.getChannel());
    matrix2.setRow(anschannel);
    matrix1->setColumn(size * size);
    matrix1->setRow(matrix2.getSize() * matrix2.getSize() * matrix2.getChannel());
    float *data = new float[matrix1->getColumn() * matrix1->getRow()];
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

    matrix1->setData(data);
    Matrix temp = matrix2 * (*matrix1);
    ans->setData(temp.getData());
    pl = 0;
    for (sc = 0; sc < ans->getChannel(); sc++) {
        for (si = 0; si < size * size; si++) {
            temp.getData()[pl++] += bias[sc];

        }
    }
#endif
}

inline void blockdot(int sc, const Matrix *matrix1, Matrix matrix2, int si, int sj, Matrix ans, int pl) {
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

}

inline void Matrix::setData(float *data) {
    Matrix::data = data;
}

inline Matrix::Matrix() { cnt++; }


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
                                                                              data(data) { cnt++; }

inline unsigned int Matrix::getSize() const {
    return size;
}

inline void Matrix::setSize(unsigned int size) {
    Matrix::size = size;
}

float *Matrix::getData() const {
    return data;
}


void Matrix::operator=(const Matrix &temp) {
    this->data = temp.data;
    size = temp.size;
    column = temp.column;
    row = temp.row;
    cnt++;
}

void matrixmatrix(const Matrix *matrix1, Matrix matrix2, Matrix *ans) {
#if defined(ARM)
    for (int i = 0; i < matrix1->getRow(); i++) {
        for (int j = 0; j < matrix2.getColumn(); j++) {
            for (int k = 0; k < matrix1->getColumn(); k++) {

                ans->getData()[j + i * matrix2.getColumn()] +=
                        matrix1->getData()[k + i * matrix1->getColumn()] *
                        matrix2.getData()[k + j * matrix1->getColumn()];

            }
        }
    }
#elif defined(X86)
    int m, n, p, si, sj, sk;
#pragma omp parallel for schedule(dynamic)
    for (sj = 0; sj < matrix2.getColumn(); sj += BLOCK) {
        for (si = 0; si < matrix1->getRow(); si += BLOCK) {
            for (sk = 0; sk < matrix1->getColumn(); sk += BLOCK) {
                m = matrix1->getRow() < si + BLOCK ? matrix1->getRow() : si + BLOCK;
                n = matrix2.getColumn() < sj + BLOCK ? matrix2.getColumn() : sj + BLOCK;
                p = sk + BLOCK < matrix1->getColumn() ? sk + BLOCK : matrix1->getColumn();
                doblock(matrix1, matrix2, si, sj, sk, m, n, p, *ans);

            }

        }

    }
#endif
}

void doblock(const Matrix *matrix1, Matrix matrix2, int si, int sj, int sk, int m, int n, int p, Matrix ans) {
#if defined(X86)
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

Matrix::Matrix(const Matrix &matrix) {
    data = matrix.data;
    size = matrix.size;
    column = matrix.column;
    row = matrix.row;
    channel = matrix.channel;
    cnt++;

}

Matrix::Matrix(unsigned int column, unsigned int row) : column(column), row(row) {}

unsigned int Matrix::getColumn() const {
    return column;
}

void Matrix::setColumn(unsigned int column) {
    Matrix::column = column;
}

unsigned int Matrix::getRow() const {
    return row;
}

void Matrix::setRow(unsigned int row) {
    Matrix::row = row;
}

#endif //CNN_MATRIX_H
