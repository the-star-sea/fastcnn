//
// Created by 张通 on 2021/1/2.
//

#ifndef CNN_MATRIX_H
#define CNN_MATRIX_H
#define X86 //Please enable it if X64 CPU
//#define ARM //Please enable it if ARM CPU
#if  defined(X86)
#include <immintrin.h>
#endif
#include "data.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <memory.h>
#include "math.h"

using namespace std;

class Matrix {
private:
    unsigned int size, channel, cnt = 1;
    float *data;
public:


    float *getData() const;

    Matrix(unsigned int size, unsigned int channel, float *data);

    void setData(float *data);

    void setChannel(unsigned int channel);

    unsigned int getChannel() const;

    unsigned int getSize() const;

    void setSize(unsigned int size);


    Matrix();




    ~Matrix();
};

void convolution(const Matrix *matrix1, Matrix matrix2, Matrix *ans, int stride, float *bias,int anschannel);

void blockdot(int sc, const Matrix *matrix1, Matrix matrix2, int si, int sj, Matrix ans, int pl);

void maxpool(const Matrix *matrix1, int size, Matrix *ans);

void Relu(Matrix *matrix);

void quickdot(float *x,  float *y, long begin, long end, float *ans);
void addzero(Matrix *matrix,int padding);
#endif //CNN_MATRIX_H
