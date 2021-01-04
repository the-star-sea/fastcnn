#include "matrix.h"

using namespace std;
using namespace cv;

int main() {
    cout<<"please input the path of the imge"<<endl;
    String path;
    cin>>path;
    Mat img = imread(path);
    clock_t start, End;
    start = clock();
    int row = img.rows;
    int col = img.cols;
    int n = row * col;
    float *res = new float[3 * n];
    for (int i = 0; i < row; i++) {
        uchar *p = img.ptr<uchar>(i);
        for (int j = 0; j < col - 1; j += 4) {
            res[i * col + j] = (float) p[3 * j + 2] / (float) 255;
            res[i * col + j + 1] = (float) p[3 * j + 5] / (float) 255;
            res[i * col + j + 2] = (float) p[3 * j + 8] / (float) 255;
            res[i * col + j + 3] = (float) p[3 * j + 11] / (float) 255;
            res[n + i * col + j] = (float) p[3 * j + 1] / (float) 255;
            res[n + i * col + j + 1] = (float) p[3 * j + 4] / (float) 255;
            res[n + i * col + j + 2] = (float) p[3 * j + 7] / (float) 255;
            res[n + i * col + j + 3] = (float) p[3 * j + 10] / (float) 255;
            res[2 * n + i * col + j] = (float) p[3 * j] / (float) 255;
            res[2 * n + i * col + j + 1] = (float) p[3 * j + 3] / (float) 255;
            res[2 * n + i * col + j + 2] = (float) p[3 * j + 6] / (float) 255;
            res[2 * n + i * col + j + 3] = (float) p[3 * j + 9] / (float) 255;
        }
    }
    Matrix *conv0 = new Matrix(128, 3, res);//RGB

    Matrix *ans1 = new Matrix;
    Matrix *para1 = new Matrix(3, 3, conv0_weight);
    Matrix *para2 = new Matrix(3, 16, conv1_weight);
    Matrix *para3 = new Matrix(3, 32, conv2_weight);
    addzero(conv0, 1);

    convolution(conv0, *para1, ans1, 2, conv0_bias, 16);
    Relu(ans1);
    Matrix *conv1 = new Matrix;
    maxpool(ans1, 2, conv1);
    Matrix *ans2 = new Matrix;
    convolution(conv1, *para2, ans2, 1, conv1_bias, 32);
    Relu(ans2);
    Matrix *conv2 = new Matrix;
    maxpool(ans2, 2, conv2);
    addzero(conv2, 1);
    Matrix *ans3 = new Matrix;
    convolution(conv2, *para3, ans3, 2, conv2_bias, 32);
    Relu(ans3);
    float *an1 = new float, *an2 = new float;
    quickdot(fc0_weight, ans3->getData(), 0, 2048, an1);
    quickdot(fc0_weight, ans3->getData(), 2048, 2048, an2);
    *an1 += fc0_bias[0];
    *an2 += fc0_bias[1];
    float man = exp(*an1) / (exp(*an1) + exp(*an2));
    cout << "nonpeople probility: " << man << endl << "people probility: " << 1 - man << endl;
    End = clock();
    double endtime = (double) (End - start) / CLOCKS_PER_SEC;
    cout << "Total time:" << endtime * 1000 << "ms" << endl;
    return 0;
}


