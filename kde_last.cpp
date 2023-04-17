#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

#define Ep 1
#define Ga 0;
using namespace std;
using namespace cv;

int kernel = Ep;                                                                                                           // 核函数类型
bool use_gaussian = 1;                                                                                                     // 是否使用高斯滤波
bool self_test_train = 0;                                                                                                  // 是否使用自作训练集
string train_image_path = !self_test_train ? "../datasets/train/given/in0000" : "../datasets/train/self/frame_0";           // 训练集图像路径
string test_image_path = !self_test_train ? "../datasets/test/given/in000825.jpg" : "../datasets/test/self/frame_0000.jpg"; // 测试集图像路径

// 计算 Epanechnikov 核的系数
double epanechnikov_coeff(int N, double h)
{
    return 15 / (8 * M_PI * N * pow(h, 3));
}

// 计算高斯核的系数
double gaussian_coeff(int N, double h)
{
    return 1 / (pow(2 * M_PI, 3.0 / 2.0) * N * pow(h, 3));
}

// 计算 Epanechnikov 核
double epanechnikov_kernel(const Vec3b &test_pixel, const Vec3b &train_pixel, double h)
{
    double prod = 1;
    for (int j = 0; j < 3; j++)
    {
        double diff = (test_pixel[j] - train_pixel[j]) / h;
        prod *= 1 - pow(diff, 2);
    }
    return prod;
}

// 计算高斯核
double gaussian_kernel(const Vec3b &test_pixel, const Vec3b &train_pixel, double h)
{
    double squared_distance = 0;
    for (int j = 0; j < 3; j++)
    {
        double diff = (test_pixel[j] - train_pixel[j]) / h;
        squared_distance += pow(diff, 2);
    }
    return exp(-squared_distance / 2);
}

// 核密度估计函数
Mat kernel_density_estimator(const vector<Mat> &train_images, const Mat &test_image, double h, int kernel_type)
{
    cout << "kernel_type: " << (kernel_type ? "Epanechnikov" : "Gaussian") << endl;
    kernel = kernel_type;
    int N = train_images.size();
    int rows = test_image.rows;
    int cols = test_image.cols;
    Mat density_values = Mat::zeros(rows, cols, CV_64F);                                  // 存储每个测试像素的密度值
    Mat result = Mat::zeros(rows, cols, CV_8UC1);                                         // 存储结果图像
    double sum_density = 0;                                                               // 存储所有测试像素的密度值之和
    double coeff = (kernel_type == Ep) ? epanechnikov_coeff(N, h) : gaussian_coeff(N, h); // 核的系数

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            Vec3b test_pixel = test_image.at<Vec3b>(r, c);
            double density = 0; // 初始化当前测试像素的密度值

            // 对每张训练图像计算该位置的像素的密度值，并累加到总密度值上
            for (const auto &train_image : train_images)
            {
                Vec3b train_pixel = train_image.at<Vec3b>(r, c);

                if (kernel_type == Ep)
                    density += epanechnikov_kernel(test_pixel, train_pixel, h);
                else if (!kernel_type)
                    density += gaussian_kernel(test_pixel, train_pixel, h);
            }

            density *= coeff;                          // 乘以核的系数
            sum_density += density;                    // 将当前像素的密度值加到总密度值上
            density_values.at<double>(r, c) = density; // 存储当前像素的密度值

            // 根据密度值将像素分为静态区域和动态区域
            double threshold = kernel_type ?-0.01 : 1e-7;
            if (density > threshold)
            {
                result.at<uchar>(r, c) = 0; // 静态区域
            }
            else
            {
                result.at<uchar>(r, c) = 255; // 动态区域
            }
        }
    }

    // 将每个测试像素的密度值输出到文件中
    string output_txt_path = "../result/"+test_image_path.substr(test_image_path.size() - 8, 4)+"/";
    string filename = kernel_type ? "epanechnikov_" : "gaussian_" + use_gaussian ? "gaussian_"
                                                  : "" + self_test_train         ? "self_test_train_"
                                                                                 : "";
    std::ofstream out_file(output_txt_path + filename + to_string(h) + ".txt");
    if (!out_file)
    {
        cerr << "无法创建输出文件！" << endl;
    }
    cout << rows << ' ' << cols << endl;
    if(!self_test_train){
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            // 将密度值限制在一定范围内，避免过大或过小的值影响图像绘制结果
            if (density_values.at<double>(r, c) < 0)
            {
                density_values.at<double>(r, c) = 0;
            }
            if (density_values.at<double>(r, c) > 2.5e-4)
            {
                density_values.at<double>(r, c) = 2.5e-4;
            }
            out_file << density_values.at<double>(r, c) << " ";
        }
        out_file << endl;
    }
    }
    else{
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                out_file << density_values.at<double>(r, c) << " ";
            }
            out_file << endl;
        }
    }
    out_file.close();

    return result;
}

// 使用Scott's Rule计算带宽h
double estimate_bandwidth(const vector<Mat> &train_images, int d, int n)
{
    int rows = train_images[0].rows;
    int cols = train_images[0].cols;

    // 计算协方差矩阵的平均值
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
            calcCovarMatrix(samples, cov, mean, COVAR_NORMAL | COVAR_ROWS); // 计算协方差矩阵
            cov_matrices[r * cols + c] = cov;                               // 存储协方差矩阵
        }
    }

    Mat avg_cov_matrix = Mat::zeros(d, d, CV_64F);
    for (const auto &cov_matrix : cov_matrices)
    {
        avg_cov_matrix += cov_matrix;
    }
    avg_cov_matrix /= (rows * cols);

    double h = pow(n, -1.0 / (d + 4)) * sqrt(trace(avg_cov_matrix)[0] / d);

    return h;
}

int main()
{
    freopen("log.txt","w",stdout);
    vector<Mat> train_images; // 存储训练集中的所有图像
    int Gaussian_Cor = 3;     // 高斯滤波的卷积核大小
    int n = 20;               // 训练集中的图像数量
    int d = 3;                // 数据维度（颜色通道数）
    for (int T = 0; T < 8; T++)
    {
        //清空训练集
        train_images.clear();
        double start = static_cast<double>(getTickCount()); // 计时开始
        // 取i的二进制最高位
        int tmp = T;
        kernel = tmp % 2;
        tmp /= 2;
        use_gaussian = tmp % 2;
        tmp /= 2;
        self_test_train = tmp % 2;
        tmp /= 2;
        // cout<<"kernel type: ";
        // cin>>kernel;
        // cout<<"use gaussian filter: ";
        // cin>>use_gaussian;
        // cout<<"use self test train: ";
        // cin>>self_test_train;
        train_image_path = !self_test_train ? "../datasets/train/given/in0000" : "../datasets/train/self/frame_0";           // 训练集图像路径
        test_image_path = !self_test_train ? "../datasets/test/given/in000273.jpg" : "../datasets/test/self/frame_0939.jpg"; // 测试集图像路径
        // 读取训练集中的所有图像，并进行高斯滤波
        for (int i = 0; i < 20; i++)
        {
            string img_num = !self_test_train ? (i < 9 ? "0" + to_string(i + 1) : to_string(i + 1)) : to_string(i + 672);
            string img_path = train_image_path + img_num + ".jpg";
            Mat img = imread(img_path, IMREAD_COLOR);
            if (use_gaussian)
                GaussianBlur(img, img, Size(Gaussian_Cor, Gaussian_Cor), 0, 0); // 进行高斯滤波
            if (img.empty())
            {
                cerr << "无法读取图片：" << img_path << endl;
                return -1;
            }
            train_images.push_back(img);
        }
        // 读取测试图像，并进行高斯滤波
        Mat test_image = imread(test_image_path, IMREAD_COLOR);
        if (use_gaussian)
            GaussianBlur(test_image, test_image, Size(Gaussian_Cor, Gaussian_Cor), 0, 0); // 进行高斯滤波
        if (test_image.empty())
        {
            cerr << "无法读取测试图片：" << test_image_path << endl;
            return -1;
        }

        // 计算h
        double h = estimate_bandwidth(train_images, d, n);
        //double h=40;
        cout << "h: " << h << endl;

        // 使用估计的最优带宽 h 对完整训练集进行核密度估计，并评估测试图像
        Mat result = kernel_density_estimator(train_images, test_image, h, kernel); // 进行核密度估计，1代表epanechnikov核，0代表高斯核
        // 显示结果图片
        // imshow("Result", result);
        // waitKey(0);
        
        // 保存结果图片
        string output_path;
        if (kernel)
        {
            if (use_gaussian)
                output_path = "../result/"+test_image_path.substr(test_image_path.size() - 8, 4)+"/result_ep_gf_" + test_image_path.substr(test_image_path.size() - 8, 4) + ".jpg";
            else
                output_path = "../result/"+test_image_path.substr(test_image_path.size() - 8, 4)+"/result_ep_" + test_image_path.substr(test_image_path.size() - 8, 4) + ".jpg";
        }
        else
        {
            if (use_gaussian)
                output_path = "../result/"+test_image_path.substr(test_image_path.size() - 8, 4)+"/result_gau_gf_" + test_image_path.substr(test_image_path.size() - 8, 4) + ".jpg";
            else
                output_path = "../result/"+test_image_path.substr(test_image_path.size() - 8, 4)+"/result_gau_" + test_image_path.substr(test_image_path.size() - 8, 4) + ".jpg";
        }

        imwrite(output_path, result);
        
        // 计时结束
        double time = ((double)getTickCount() - start) / getTickFrequency();
        cout << "time:" << time << 's' << endl;
        cout << "---------------------------------" << endl;
    }
    fclose(stdout);
    return 0;
}