#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

using namespace std;
using namespace cv;

double kernel_fun(double x) {
    if (abs(x) <= 1.0) {
        return 0.5968 * (1 - pow(x, 2.0));
    } else {
        return 0.0;
    }
}

int main(int argc, char** argv) {
    const int NUM_TRAIN_IMAGES = 20;
    const string TRAIN_IMAGE_PATH = "/home/zby/codetest/KDE/train/in0000";
    const string TEST_IMAGE_PATH = "/home/zby/codetest/KDE/test/in000273.jpg";
    const int KERNEL_SIZE = 21;
    const double KERNEL_SCALE = 1.0 / 255.0;
    const int THRESHOLD = 128;
    Mat kernel = Mat::zeros(Size(KERNEL_SIZE, KERNEL_SIZE), CV_64F);
    // Read training images and estimate kernel density
    Mat density_map = Mat::zeros(Size(320, 240), CV_64F);
    for (int i = 0; i < NUM_TRAIN_IMAGES; i++) {
        string filename = TRAIN_IMAGE_PATH + (i < 9 ? "0" : "") + to_string(i + 1) + ".jpg";
        Mat image = imread(filename, IMREAD_COLOR);
        cvtColor(image, image, COLOR_BGR2GRAY);
        image.convertTo(image, CV_64F, KERNEL_SCALE);
        for (int y = 0; y < KERNEL_SIZE; y++) {
            double* row_ptr = kernel.ptr<double>(y);
            for (int x = 0; x < KERNEL_SIZE; x++) {
                double dx = (double)(x - KERNEL_SIZE / 2);
                double dy = (double)(y - KERNEL_SIZE / 2);
                row_ptr[x] = kernel_fun(sqrt(dx * dx + dy * dy));
            }
        }
        filter2D(image, image, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
        density_map += image;
    }
    density_map /= NUM_TRAIN_IMAGES;

    // Threshold density map to get static region mask
    Mat static_mask = density_map > THRESHOLD;

    // Read test image and estimate kernel density
    Mat test_image = imread(TEST_IMAGE_PATH, IMREAD_COLOR);
    cvtColor(test_image, test_image, COLOR_BGR2GRAY);
    test_image.convertTo(test_image, CV_64F, KERNEL_SCALE);
    Mat test_density_map = Mat::zeros(Size(320, 240), CV_64F);
    filter2D(test_image, test_density_map, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);

    // Multiply test density map by static mask to get dynamic region mask
    Mat dynamic_mask = test_density_map <= THRESHOLD;
    bitwise_and(dynamic_mask, static_mask, dynamic_mask);

    // Draw dynamic region on test image
    Mat output_image = test_image.clone();
    output_image.setTo(Scalar(0, 0, 255), dynamic_mask);
    imshow("Dynamic Region", output_image);
    waitKey(0);

    return 0;
}
