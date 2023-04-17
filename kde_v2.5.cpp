#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
const double PI=3.14159265358979323846;
double kernel_fun(double x) {
    if (abs(x) <= 1.0) {
        return (5.0/(2.0*4.0*PI/3.0)) * (1.0 - pow(x, 2.0));
    } else {
        return 0.0;
    }
}


int main() {
    const int num_images = 20;
    const double h = 0.855; // 核密度估计中的带宽参数
    std::string train_path_prefix = "/home/zby/codetest/KDE/train/in0000";
    std::string test_image_path = "/home/zby/codetest/KDE/test/in000825.jpg";

    cv::Mat sum_density;

    for (int i = 0; i < num_images; i++) {
        std::string image_path = train_path_prefix + (i < 9 ? "0" + std::to_string(i + 1) : std::to_string(i + 1)) + ".jpg";
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat density(image.rows, image.cols, CV_64FC3, cv::Scalar(0, 0, 0));

        if (i == 0) {
            sum_density = cv::Mat::zeros(image.rows, image.cols, CV_64FC3);
        }

        for (int row = 0; row < image.rows; row++) {
            for (int col = 0; col < image.cols; col++) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
                Eigen::Vector3d kde_rgb(0, 0, 0);

                for (int r = -1; r <= 1; r++) {
                    for (int c = -1; c <= 1; c++) {
                        if (row + r >= 0 && row + r < image.rows && col + c >= 0 && col + c < image.cols) {
                            cv::Vec3b neighbor = image.at<cv::Vec3b>(row + r, col + c);
                            for (int ch = 0; ch < 3; ch++) {
                                double diff = (pixel[ch] - neighbor[ch]) / 255.0;
                                kde_rgb[ch] += kernel_fun(diff / h);
                            }
                        }
                    }
                }

                density.at<cv::Vec3d>(row, col) = cv::Vec3d(kde_rgb[0], kde_rgb[1], kde_rgb[2]);

            }
        }

        sum_density += density;
    }

    sum_density /= (num_images * h);

    // 读取测试图片并标出动态区域和静态区域
    cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_COLOR);
    cv::Mat result_image = test_image.clone();

    double threshold = 1.15; // 可以根据需要调整静态/动态区域的
    // 遍历测试图像的每个像素
    for (int row = 0; row < test_image.rows; row++) {
        for (int col = 0; col < test_image.cols; col++) {
            cv::Vec3d density_value = sum_density.at<cv::Vec3d>(row, col);
            cv::Vec3b pixel = test_image.at<cv::Vec3b>(row, col);

            // 计算当前像素的核密度估计值
            Eigen::Vector3d kde_rgb(0, 0, 0);
            for (int r = -1; r <= 1; r++) {
                for (int c = -1; c <= 1; c++) {
                    if (row + r >= 0 && row + r < test_image.rows && col + c >= 0 && col + c < test_image.cols) {
                        cv::Vec3b neighbor = test_image.at<cv::Vec3b>(row + r, col + c);
                        for (int ch = 0; ch < 3; ch++) {
                            double diff = (pixel[ch] - neighbor[ch]) / 255.0;
                            kde_rgb[ch] += kernel_fun(diff / h);
                        }
                    }
                }
            }
            // 检查是否为动态区域
            bool is_dynamic = false;
            for (int ch = 0; ch < 3; ch++) {
                std::cout<<abs(kde_rgb[ch] - density_value[ch]) / h<<std::endl;
                if (abs(kde_rgb[ch] - density_value[ch]) / h > threshold) {
                    is_dynamic = true;
                    break;
                }
            }

            // 标出动态区域（红色）和静态区域（绿色）
            if (is_dynamic) {
                result_image.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
            } else {
                result_image.at<cv::Vec3b>(row, col) = cv::Vec3b(0,0,0);
            }
        }
    }

    // 显示结果并保存图像
    cv::imshow("Result Image", result_image);
    cv::waitKey(0);
    cv::imwrite("/home/zby/codetest/KDE/result/result_image.jpg", result_image);

    return 0;
}
