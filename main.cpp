#include "matrix.h"
using namespace std;
using namespace cv;

int main() {

    Mat image = imread("/Users/stone/CLionProjects/SimpleCNNbyCPP/samples/face.jpg");
    Mat Channels[3];
    split(image, Channels);//BGR
    float *r=new float [128*128*3];//RGB
    for(int i=0;i<128*128;i++){
        r[i]=Channels[2].data[i]/255;
    }
    for(int i=0;i<128*128;i++){
        r[i+128*128]=Channels[1].data[i]/255;
    }
    for(int i=0;i<128*128;i++){
        r[i+128*128*2]=Channels[0].data[i]/255;
    }

    return 0;
}


