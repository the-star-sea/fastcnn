#include "matrix.h"
using namespace std;
using namespace cv;

int main() {

    Mat image = imread("/Users/stone/CLionProjects/cnn/bg.jpg");
    Mat Channels[3];
    split(image, Channels);//BGR
/*Matrix * hha=new Matrix(4,2,new float);
float* uu=new float [32];
Matrix *bias=new Matrix(2,2,new float);
hha->setData(uu);
bias->setData(uu);
float *hh=new float [32];
for(int i=0;i<32;i++){
    uu[i]=i;
    hh[i]=0;
}
Matrix * aaaa=new Matrix;

convolution(hha,*bias,aaaa,2,hh,4);
for(int i=0;i<aaaa->getChannel();i++){
    for(int j=0;j<aaaa->getSize();j++){
        for(int k=0;k<aaaa->getSize();k++){
            cout<<aaaa->getData()[i*aaaa->getSize()*aaaa->getSize()+j*aaaa->getSize()+k]<<" ";
        }cout<<endl;
    }
    cout<<endl;
}*/
    Matrix * conv0=new Matrix(128,3,new float [128*128*3]);//RGB
    for(int i=0;i<128*128;i++){
        conv0->getData()[i]=Channels[2].data[i]/255;
    }
    for(int i=0;i<128*128;i++){
        conv0->getData()[i+128*128]=Channels[1].data[i]/255;
    }
    for(int i=0;i<128*128;i++){
        conv0->getData()[i+128*128*2]=Channels[0].data[i]/255;
    }
    Matrix *ans1=new Matrix;
    Matrix *para1=new Matrix(3,3,conv0_weight);
    Matrix *para2=new Matrix(3,16,conv1_weight);
    Matrix *para3=new Matrix(3,32,conv2_weight);
    addzero(conv0,1);
convolution(conv0,*para1,ans1,2,conv0_bias,16);
Relu(ans1);
    Matrix * conv1=new Matrix(ans1->getSize()/2,ans1->getChannel(),new float [ans1->getSize()*ans1->getSize()*ans1->getChannel()/4]);
maxpool(ans1,2,conv1);
    Matrix *ans2=new Matrix;
    convolution(conv1,*para2,ans2,1,conv1_bias,32);
    Relu(ans2);
    Matrix * conv2=new Matrix(ans2->getSize()/2,ans2->getChannel(),new float [ans2->getSize()*ans2->getSize()*ans2->getChannel()/4]);
    maxpool(ans2,2,conv2);
    addzero(conv2,1);
    Matrix*ans3=new Matrix;
    convolution(conv2,*para3,ans3,2,conv2_bias,32);
    Relu(ans3);
    float *an1=new float,*an2=new float ;
    quickdot(fc0_weight,ans3->getData(),0,2048,an1);
    quickdot(fc0_weight,ans3->getData(),2048,2048,an2);
    *an1+=fc0_bias[0];*an2+=fc0_bias[1];
    float man=exp(*an1)/(exp(*an1)+exp(*an2));
    cout<<"nonpeople probility: "<<man<<endl<<"people probility: "<<1-man;
    return 0;
}


