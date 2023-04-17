#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include<fstream> 

#define Ep 1
#define Gaussian 0;
using namespace std;
using namespace cv;



Mat kernel_density_estimator(const vector<Mat> &train_images, const Mat &test_image, double h)
{
    int N = train_images.size();
    int rows = test_image.rows;
    int cols = test_image.cols;
    Mat density_values = Mat::zeros(rows, cols, CV_64F);
    Mat density_values_sum = Mat::zeros(rows, cols, CV_64F);
    double sum_density = 0;
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
            sum_density += density;
            density_values.at<double>(r, c) = density;
            density_values_sum.at<double>(r, c) += density;

            density=log(abs(density));
            cout << density << endl;
            if (density < -1)
            {
                result.at<uchar>(r, c) = 0; // 静态区域
            }
            else
            {
                result.at<uchar>(r, c) = 255; // 动态区域
            }
        }
    }
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            density_values.at<double>(r, c) /= sum_density;
            density_values_sum.at<double>(r, c) /= sum_density;
        }
    }

    std::ofstream out_file("density_values.txt");
    if (!out_file)
    {
        cerr << "无法创建输出文件！" << endl;
    }
    cout<<rows<<' '<<cols<<endl;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if(density_values.at<double>(r, c)<0) density_values_sum.at<double>(r, c)=abs(density_values_sum.at<double>(r, c));
            density_values_sum.at<double>(r, c)=log(density_values_sum.at<double>(r, c));
            out_file << density_values_sum.at<double>(r, c) << " ";
        }
        out_file << endl;
    }
    out_file.close();
    return result;
}
Mat kernel_density_estimator_Gaussian(const vector<Mat> &train_images, const Mat &test_image, double h)
{
    int N = train_images.size();
    int rows = test_image.rows;
    int cols = test_image.cols;
    Mat density_values = Mat::zeros(rows, cols, CV_64F);
    double sum_density = 0;
    double coeff = 1 / (pow(2 * M_PI, 3.0 / 2) * N * pow(h, 3));

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
                double squared_distance = 0;

                for (int j = 0; j < 3; j++)
                {
                    double diff = (test_pixel[j] - train_pixel[j]) / h;
                    squared_distance += pow(diff, 2);
                }

                density += exp(-squared_distance / 2);
            }

            density *= coeff;
            sum_density += density;
            density_values.at<double>(r, c) = density;

            density=log(abs(density));
           // cout << density << endl;
            if (density < -20)
            {
                result.at<uchar>(r, c) = 255; // 静态区域
            }
            else
            {
                result.at<uchar>(r, c) = 0; // 动态区域
            }
        }
    }
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            density_values.at<double>(r, c) /= sum_density;
        }
    }

    std::ofstream out_file("density_values.txt");
    if (!out_file)
    {
        cerr << "无法创建输出文件！" << endl;
    }
    cout<<rows<<' '<<cols<<endl;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            double log_density = log(abs(density_values.at<double>(r, c)));
            out_file << log_density << " ";
        }
        out_file << endl;
    }
    out_file.close();
    return result;
}


int main()
{
    vector<Mat> train_images;
    int Gaussian_Cor = 3;
    int n = 20;
    //计时开始
    double start = static_cast<double>(getTickCount());
    string train_image_path = "/home/zby/codetest/KDE/train/in0000";
    string test_image_path = "/home/zby/codetest/KDE/test/in000273.jpg";
    // string train_image_path = "/home/zby/codetest/KDE/datasets/train/self/frame_0";
    // string test_image_path = "/home/zby/codetest/KDE/datasets/test/self/frame_0396.jpg";
    for (int i = 0; i < 20; i++)
    {
         string img_num = i < 9 ? "0" + to_string(i + 1) : to_string(i + 1);
        // string img_num = to_string(i + 672);
        
        string img_path = train_image_path + img_num + ".jpg";
        Mat img = imread(img_path, IMREAD_COLOR);
        // 高斯滤波
         GaussianBlur(img, img, Size(Gaussian_Cor, Gaussian_Cor), 0, 0);
        if (img.empty())
        {
            cerr << "无法读取图片：" << img_path << endl;
            return -1;
        }
        train_images.push_back(img);
    }
    Mat test_image = imread(test_image_path, IMREAD_COLOR);
    // 高斯滤波
     GaussianBlur(test_image, test_image, Size(Gaussian_Cor, Gaussian_Cor), 0, 0);
    if (test_image.empty())
    {
        cerr << "无法读取测试图片：" << test_image_path << endl;
        return -1;
    }

    int rows = train_images[0].rows;
    int cols = train_images[0].cols;
    int d = 3; // 数据维度（颜色通道数）

    // 计算训练集中每个像素位置的颜色通道的协方差矩阵
    vector<Mat> cov_matrices(rows * cols, Mat::zeros(d, d, CV_64F));
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            vector<Mat> samples;
            for (const auto &train_image : train_images)
            {
                Mat pixel(1, d, CV_64F);
                for (int j = 0; j < d; j++)
                {
                    pixel.at<double>(0, j) = train_image.at<Vec3b>(r, c)[j];
                }
                samples.push_back(pixel);
            }

            Mat mean, cov;
            calcCovarMatrix(samples, cov, mean, COVAR_NORMAL | COVAR_ROWS);
            cov_matrices[r * cols + c] = cov;
        }
    }

    // 计算协方差矩阵的平均值
    Mat avg_cov_matrix = Mat::zeros(d, d, CV_64F);
    for (const auto &cov_matrix : cov_matrices)
    {
        avg_cov_matrix += cov_matrix;
    }
    avg_cov_matrix /= (rows * cols);

    // 使用协方差矩阵来估计带宽
    double h = pow(n, -1.0 / (d + 4)) * sqrt(trace(avg_cov_matrix)[0] / d);

    cout << "h:" << h << endl;
    // 使用估计的最优带宽 h 对完整训练集进行核密度估计，并评估测试图像
    Mat result = kernel_density_estimator(train_images, test_image, h);
    double time = ((double)getTickCount() - start) / getTickFrequency();
    cout << "time:" << time << endl;
    // 显示结果图片
    // imshow("Result", result);
    // waitKey(0);
    // 保存结果图片
    imwrite("/home/zby/codetest/KDE/build/result.jpg", result);
    //计时结束
    

    return 0;
}
