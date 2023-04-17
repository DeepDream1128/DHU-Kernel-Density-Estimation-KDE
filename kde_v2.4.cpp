#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

const double PI=3.14159265358979323846;
double kernel_fun(double x) {
    if (abs(x) <= 1.0) {
        return (5.0/(2.0*4.0*PI/3.0)) * (1.0 - pow(x, 2.0));
    } else {
        return 0.0;
    }
}

int main() {
    // 设置参数
    int n = 20; // 图像数量
    int width = 320; // 图像宽度
    int height = 240; // 图像高度
    int d = 3; // 像素维度
    double h = 0.855; // 核带宽

    // 读取训练图像
    vector<MatrixXd> X(n, MatrixXd::Zero(width * height, d));
    for (int i = 0; i < n; i++) {
        string filename = "/home/zby/codetest/KDE/train/in0000" + (i<9?"0"+to_string(i+1):to_string(i+1)) + ".jpg";
        Mat image = imread(filename, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Can't read image file " << filename << endl;
            return -1;
        }
        cvtColor(image, image, COLOR_BGR2RGB);
        MatrixXd xi(width * height, d);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Vec3b pixel = image.at<Vec3b>(y, x);
                xi(y * width + x, 0) = pixel[0];
                xi(y * width + x, 1) = pixel[1];
                xi(y * width + x, 2) = pixel[2];
            }
        }
        X[i] = xi;
    }

    // 计算核密度估计
    MatrixXd K(n, width * height);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < width * height; j++) {
            double ksum = 0.0;
            for (int l = 0; l < n; l++) {
                VectorXd xjl = (X[l].row(j) - X[i].row(j)) / h;
                //将xit中每一个元素都带入核函数
                for (int k = 0; k < d; k++) {
                    ksum += kernel_fun(xjl(k));
                }
            }
            K(i, j) = ksum / (n * h);
        }
    }

    // 读取测试图像
    string test_filename = "/home/zby/codetest/KDE/test/in000825.jpg";
    Mat test_image = imread(test_filename, IMREAD_COLOR);
    if (test_image.empty()) {
        cerr << "Can't read image file " << test_filename << endl;
        return -1;
    }
    cvtColor(test_image,test_image,COLOR_BGR2RGB);

// 将测试图像转换为向量
MatrixXd Xtest(width * height, d);
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        Vec3b pixel = test_image.at<Vec3b>(y, x);
        Xtest(y * width + x, 0) = pixel[0];
        Xtest(y * width + x, 1) = pixel[1];
        Xtest(y * width + x, 2) = pixel[2];
    }
}

// 计算测试图像的密度估计
VectorXd Ktest(width * height);
for (int j = 0; j < width * height; j++) {
    double ksum = 0.0;
    for (int i = 0; i < n; i++) {
        VectorXd xit = (Xtest.row(j) - X[i].row(j)) / h;
        for (int k = 0; k < d; k++) {
            ksum += kernel_fun(xit(k));
        }
    }
    Ktest(j) = ksum / (n * h);
}


//输出Ktest
cout << Ktest << endl;
// 将密度估计映射到图像上
Mat result_image = Mat(height, width, CV_8UC3);
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        double k = Ktest(y * width + x);
        if (k > 0) {
            result_image.at<Vec3b>(y, x) = Vec3b(0,0,0); // 标记为动态区域
        } else {
            result_image.at<Vec3b>(y, x) = Vec3b(255,255,255); // 标记为静态区域
        }
    }
}

// 显示结果
namedWindow("Result", WINDOW_NORMAL);
imshow("Result", result_image);
// 显示测试图
namedWindow("Test", WINDOW_NORMAL);
imshow("Test", test_image);
waitKey(0);

return 0;
}
