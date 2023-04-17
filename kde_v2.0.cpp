#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

double kernelx(double x) {
    if (abs(x) <= 1.0) {
        return 0.5968 * (1 - pow(x, 2.0));
    } else {
        return 0.0;
    }
}

int main() {
    int num_images = 20;
    double threshold = 1000.0;

    for (int i = 0; i < num_images; i++) {
        cv::Mat img = cv::imread("/home/zby/codetest/KDE/train/in0000" + (i < 9 ? "0" + std::to_string(i + 1) : std::to_string(i + 1)) + ".jpg");
        cv::Mat grayImg;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        cv::Mat kernelImg;
        cv::Mat kernel = cv::Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double x = (i - 1) / 3.0;
                double y = (j - 1) / 3.0;
                kernel.at<double>(i, j) = kernelx(x) * kernelx(y);
            }
        }
        cv::filter2D(grayImg, kernelImg, -1, kernel);
        cv::Mat binaryImg;
        cv::threshold(kernelImg, binaryImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > threshold) {
                cv::drawContours(img, contours, i, cv::Scalar(0, 0, 255), 2);
            }
        }
        cv::imshow("img", img);
        cv::waitKey(0);
    }

    return 0;
}
