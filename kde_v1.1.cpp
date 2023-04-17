#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 定义核函数
double kernel(double x) {
    return exp(-x * x / 2) / sqrt(2 * M_PI);
}

// 核估计函数
double kernelEstimate(double x, const Mat& data, double h) {
    double sum = 0.0;
    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols; j++) {
            double distance = sqrt(pow(x - data.at<uchar>(i, j), 2.0));
            sum += kernel(distance / h);
        }
    }
    return sum / (data.rows * data.cols * h);
}

// 二值化图像
Mat binarize(const Mat& img, double threshold) {
    Mat binarized;
    cv::threshold(img, binarized, threshold, 255, THRESH_BINARY);
    return binarized;
}

// 形态学操作
Mat morphology(const Mat& img, int iterations) {
    Mat result;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(img, result, MORPH_OPEN, element, Point(-1,-1), iterations);
    morphologyEx(result, result, MORPH_CLOSE, element, Point(-1,-1), iterations);
    return result;
}

// 标记静态区域
Mat labelStatic(const Mat& img, int threshold) {
    Mat labeled;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) > threshold) {
            drawContours(labeled, contours, i, Scalar(255), FILLED);
        }
    }
    return labeled;
}

int main() {
    // 读取训练图像并计算核密度估计值
    vector<Mat> trainImages;
    for (int i = 0; i < 20; i++) {
        Mat trainImage = imread("/home/zby/codetest/KDE/train/in0000" + (i<9?"0"+to_string(i+1):to_string(i+1)) + ".jpg", IMREAD_GRAYSCALE);
        if(trainImage.empty()) {
            cerr << "Failed to read image " << i << endl;
            return -1;
        }
        trainImages.push_back(trainImage);
    }
    std::cout<<"Read Complete!"<<std::endl;
    Mat trainDensity(trainImages[0].size(), CV_64F, Scalar(0));
    double h = 20.0;
    for (int i = 0; i < trainImages.size(); i++) {
        Mat density(trainImages[i].size(), CV_64F);
        for (int j = 0; j < trainImages[i].rows; j++) {
            for (int k = 0; k < trainImages[i].cols; k++) {
                density.at<double>(j, k) = kernelEstimate(trainImages[i].at<uchar>(j, k), trainImages[i], h);
                std::cout<<density.at<double>(j, k)<<std::endl;
            }
        }
    trainDensity += density;
    }
    trainDensity /= trainImages.size();
    std::cout<<"Train Complete!"<<std::endl;
    // 读取测试图像并计算核密度估计值
    Mat testImage = imread("/home/zby/codetest/KDE/test/in000273.jpg", IMREAD_COLOR);
    Mat testDensity(testImage.size(), CV_64F);
    for (int i = 0; i < testImage.rows; i++) {
        for (int j = 0; j < testImage.cols; j++) {
            testDensity.at<double>(i, j) = kernelEstimate(testImage.at<uchar>(i, j), trainDensity, h);
        }
    }
    std::cout<<"Test Complete!"<<std::endl;
    // 二值化和形态学操作
    Mat binarized = binarize(testDensity, 200);
    Mat morphologyResult = morphology(binarized, 3);

    // 标记静态区域
    Mat labeled(testImage.size(), CV_8UC1, Scalar(0));
    labeled = labelStatic(morphologyResult, 1000);
    std::cout<<"Label Complete!"<<std::endl;
    // 保存结果
    //imwrite("labeled.jpg", labeled);
imshow("labeled", labeled);
waitKey(0);
    return 0;
}