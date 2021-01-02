#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main() {

    Mat image = imread("/Users/stone/CLionProjects/SimpleCNNbyCPP/samples/face.jpg");
    cout<<image.channels();
    return 0;
}

