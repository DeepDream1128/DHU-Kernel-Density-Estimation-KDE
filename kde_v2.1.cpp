#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

double kernel_fun(double x) {
    if (std::abs(x) <= 1.0) {
        return 0.5968 * (1 - std::pow(x, 2.0));
    } else {
        return 0.0;
    }
}

int main() {
    // Load training images
    std::vector<cv::Mat> train_images;
    for (int i = 0; i < 20; i++) {
        std::string file_name = "/home/zby/codetest/KDE/train/in0000" + (i<9?"0"+std::to_string(i+1):std::to_string(i+1)) + ".jpg";
        cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
        train_images.push_back(image);
    }

    // Load test image
    cv::Mat test_image = cv::imread("/home/zby/codetest/KDE/test/in000273.jpg", cv::IMREAD_COLOR);

    // Convert test image to grayscale
    cv::Mat test_gray;
    cv::cvtColor(test_image, test_gray, cv::COLOR_BGR2GRAY);

    // Calculate kernel density estimate for each pixel
    cv::Mat kde_map(test_gray.size(), CV_64FC1, cv::Scalar(0.0));
    for (int i = 0; i < test_gray.rows; i++) {
        for (int j = 0; j < test_gray.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < train_images.size(); k++) {
                double diff = (double)test_gray.at<uchar>(i,j) - (double)train_images[k].at<uchar>(i,j);
                sum += kernel_fun(diff);
            }
            kde_map.at<double>(i,j) = sum / train_images.size();
        }
    }

    // Choose threshold and mark dynamic region
    double threshold = 0;
    cv::Mat result_image(test_gray.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < test_gray.rows; i++) {
        for (int j = 0; j < test_gray.cols; j++) {
            if (kde_map.at<double>(i,j) < threshold) {
                result_image.at<uchar>(i,j) = 255;
            }
        }
    }

    // Display and save result image
    cv::imshow("Result Image", result_image);
    cv::waitKey(0);
    cv::imwrite("result.jpg", result_image);

    return 0;
}
