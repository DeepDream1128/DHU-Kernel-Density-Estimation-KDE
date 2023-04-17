#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

Mat kernel_density_estimator(const vector<Mat> &train_images, const Mat &test_image, double h)
{
    int N = train_images.size();
    int rows = test_image.rows;
    int cols = test_image.cols;

    double coeff = 15 / (8 * M_PI * N * pow(h, 3));

    Mat result = Mat::zeros(rows, cols, CV_8UC1);

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            Vec3b test_pixel = test_image.at<Vec3b>(r, c);
            double density = 0;

            for (const auto &train_image : train_images)
            {
                Vec3b train_pixel = train_image.at<Vec3b>(r, c);
                double prod = 1;

                for (int j = 0; j < 3; j++)
                {
                    double diff = (test_pixel[j] - train_pixel[j]) / h;
                    prod *= 1 - pow(diff, 2);
                }

                density += prod;
            }

            density *= coeff;

            if (density < 0)
            {
                result.at<uchar>(r, c) = 255; // 静态区域
            }
            else
            {
                result.at<uchar>(r, c) = 0; // 动态区域
            }
        }
    }

    return result;
}

int main()
{
    vector<Mat> train_images;
    int Gaussian_Cor = 5;
    string train_image_path = "/home/zby/codetest/KDE/train/in0000";
    string test_image_path = "/home/zby/codetest/KDE/test/in001690.jpg";
    for (int i = 0; i < 20; i++)
    {
        string img_num = i < 9 ? "0" + to_string(i + 1) : to_string(i + 1);
        string img_path = train_image_path + img_num + ".jpg";
        Mat img = imread(img_path, IMREAD_COLOR);
        // 高斯滤波
        //GaussianBlur(img, img, Size(Gaussian_Cor, Gaussian_Cor), 0, 0);
        if (img.empty())
        {
            cerr << "无法读取图片：" << img_path << endl;
            return -1;
        }
        train_images.push_back(img);
    }
    Mat test_image = imread(test_image_path, IMREAD_COLOR);
    // 高斯滤波
    //GaussianBlur(test_image, test_image, Size(Gaussian_Cor, Gaussian_Cor), 0, 0);
    if (test_image.empty())
    {
        cerr << "无法读取测试图片：" << test_image_path << endl;
        return -1;
    }

    int rows = train_images[0].rows;
    int cols = train_images[0].cols;
    int d = 3; // 数据维度（颜色通道数）

    // 计算训练集中每个像素位置的颜色通道的方差
    vector<double> variances(d, 0);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            for (int j = 0; j < d; j++)
            {
                double mean = 0;
                for (const auto &train_image : train_images)
                {
                    mean += train_image.at<Vec3b>(r, c)[j];
                }
                mean /= train_images.size();
                double variance = 0;
                for (const auto &train_image : train_images)
                {
                    double diff = train_image.at<Vec3b>(r, c)[j] - mean;
                    variance += diff * diff;
                }
                variance /= train_images.size();
                variances[j] += variance;
            }
        }
    }
    // 使用 Scott's Rule 计算最优的带宽
    int n = train_images.size();
    // 计算总方差的平均值
    double avg_variance = 0;
    for (const auto &variance : variances)
    {
        avg_variance += variance;
    }
    avg_variance /= (rows * cols * d);

    // 使用 Scott's Rule 计算最优的带宽
    double h = pow(n, -1.0 / (d + 4)) * avg_variance;
    cout << "h:" << h << endl;
    // 使用估计的最优带宽 h 对完整训练集进行核密度估计，并评估测试图像
    Mat result = kernel_density_estimator(train_images, test_image, h);

    // 显示结果图片
    imshow("Result", result);
    waitKey(0);
    // 保存结果图片
    imwrite("/home/zby/codetest/KDE/result/result.jpg", result);

    return 0;
}
