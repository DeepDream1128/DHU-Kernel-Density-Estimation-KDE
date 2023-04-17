#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 计算平均向量
Mat calcAvgVec(vector<Mat>& images, int x, int y)
{
    int n = images.size();
    Mat avgVec(1, n, CV_32F);

    for (int i = 0; i < n; i++) {
        Mat img = images[i];
        avgVec.at<float>(0, i) = img.at<uchar>(y, x);
    }

    return avgVec;
}

// 计算核密度估计向量
// 计算核密度估计向量
// 计算核密度估计向量
Mat calcKDEVec(vector<Mat>& images, int x, int y)
{
    int n = images.size();
    Mat kdeVec(1, n, CV_32F);

    for (int i = 0; i < n; i++) {
        Mat img = images[i];

        // 计算在该位置处的像素值
        float val = img.at<uchar>(y, x);

        // 计算核密度估计
        Mat hist;
        int channels[] = {0};
        int histSize[] = {256};
        float range[] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);
        float kde = hist.at<float>(val) / n;

        kdeVec.at<float>(0, i) = kde;
    }

    return kdeVec;
}


int main()
{
    // 读取所有图像
    vector<Mat> images;
    for (int i = 1; i <= 20; i++) {
        Mat img = imread("/home/zby/codetest/KDE/train/in0000" + (i<10?"0"+to_string(i):to_string(i)) + ".jpg", IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Failed to read image " << i << endl;
            return -1;
        }

        // 转换为灰度图像
        cvtColor(img, img, COLOR_BGR2GRAY);

        images.push_back(img);
    }

    // 计算平均向量
    int rows = images[0].rows;
    int cols = images[0].cols;
    Mat avgVec = calcAvgVec(images, cols / 2, rows / 2);

    // 计算静态区域的掩码
    float threshold = 0.9;
    Mat mask(rows, cols, CV_8UC1);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            Mat kdeVec = calcKDEVec(images, x, y);
            double dist = norm(kdeVec, avgVec);
            if (dist < threshold) {
                mask.at<uchar>(y, x) = 255;  // 静态区域
            }
            else {
                mask.at<uchar>(y, x) = 0;  // 动态区域
            }
        }
    }

    // 读取测试图像
    Mat testImg = imread("/home/zby/codetest/KDE/test/in000273.jpg", IMREAD_COLOR);
    if (testImg.empty()) {
        cerr << "Failed to read test image" << endl;
        return -1;
    }

    // 标出动态区域
    Mat outputImg;
    cvtColor(testImg, outputImg, COLOR_GRAY2BGR);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (mask.at<uchar>(y, x) == 0) {
                outputImg.at<Vec3b>(y, x) = Vec3b(100, 100, 0);  // 动态区域用黑色标出
            }
        }
    }

    // 保存输出图像
    imwrite("output.jpg", outputImg);
    imshow("output", outputImg);
    waitKey(0);

    return 0;
    }