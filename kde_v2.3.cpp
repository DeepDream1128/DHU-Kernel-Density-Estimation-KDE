#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

double kernel_fun(double x) {
    if (abs(x) <= 1.0) {
        return 0.5968 * (1 - pow(x, 2.0));
    } else {
        return 0.0;
    }
}

MatrixXf compute_kernel_density_estimate(const MatrixXf& data, const MatrixXf& grid, double h) {
    int n = data.rows();
    int m = grid.rows();
    MatrixXf result(m, 1);
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            double d = (grid.row(i) - data.row(j)).norm() / h;
            sum += kernel_fun(d);
            cout<<sum<<endl;
        }
        result(i, 0) = sum / (n * h);
    }
    return result;
}

int main() {
    // Read training images
    int num_train = 20;
    vector<Mat> train_images;
    cout<<"Begin!"<<endl;
    for (int i = 0; i < num_train; i++) {
        string filename = "/home/zby/codetest/KDE/train/in0000" + (i<9?"0"+to_string(i+1):to_string(i+1)) + ".jpg";
        Mat img = imread(filename);
        if (img.empty()) {
            cerr << "Error: cannot read image " << filename << endl;
            return -1;
        }
        cout<<"Read "<<i<<" Complete!"<<endl;
        train_images.push_back(img);
    }

    cout<<"Read1 Complete!"<<endl;
    // Convert training images to data matrix
    int height = 240;
    int width = 320;
    int num_pixels = height * width;
    MatrixXf data(num_pixels, 3);
    for (int i = 0; i < num_pixels; i++) {
        int y = i / width;
        int x = i % width;
        for (int j = 0; j < 3; j++) {
            data(i, j) = train_images[0].at<Vec3b>(y, x)[j] / 255.0;
        }
    }
    cout<<"Read2 Complete!"<<endl;
    for (int i = 1; i < num_train; i++) {
        for (int j = 0; j < num_pixels; j++) {
            int y = j / width;
            int x = j % width;
            for (int k = 0; k < 3; k++) {
                data(j, k) += train_images[i].at<Vec3b>(y, x)[k] / 255.0;
            }
        }
    }
    data /= num_train;
cout<<"Read3 Complete!"<<endl;
    // Compute kernel density estimate on a grid
    int num_test_pixels = 300 * 300;
    MatrixXf grid(num_test_pixels, 3);
    double step = 1.0 / 299;
    for (int i = 0; i < 300; i++) {
        for (int j = 0; j < 300; j++) {
            int index = i * 300 + j;
            grid(index, 0) = i * step;
            grid(index, 1) = j * step;
            grid(index, 2) = 0.0;
            cout<<grid(index, 0)<<" "<<grid(index, 1)<<" "<<grid(index, 2)<<endl;
        }
    }
    MatrixXf pdf = compute_kernel_density_estimate(data, grid, 0.855);
cout<<"Read4 Complete!"<<endl;
    // Convert density estimate to image
Mat pdf_img(300, 300, CV_8UC1);
for (int i = 0; i < num_test_pixels; i++) {
    int y = i / 300;
    int x = i % 300;
    pdf_img.at<uchar>(y, x) = pdf(i, 0) * 255;
}

// Read test image
Mat test_img = imread("/home/zby/codetest/KDE/test/in000273.jpg");
if (test_img.empty()) {
    cerr << "Error: cannot read image /home/zby/codetest/KDE/test/in000273.jpg" << endl;
    return 1;
}
cout<<"Test Read Complete!"<<endl;

// Convert test image to data matrix
MatrixXf test_data(num_pixels, 3);
for (int i = 0; i < num_pixels; i++) {
    int y = i / width;
    int x = i % width;
    for (int j = 0; j < 3; j++) {
        test_data(i, j) = test_img.at<Vec3b>(y, x)[j] / 255.0;
    }
}

cout<<"Convert Complete!"<<endl;
// Compute density estimate on test image
MatrixXf test_pdf = compute_kernel_density_estimate(data, test_data, 0.05);

// Convert density estimate to image
Mat test_pdf_img(height, width, CV_8UC1);
for (int i = 0; i < num_pixels; i++) {
    int y = i / width;
    int x = i % width;
    test_pdf_img.at<uchar>(y, x) = test_pdf(i, 0) * 255;
}

// Compute static/dynamic segmentation
Mat static_img(height, width, CV_8UC1);
double threshold = 0.15;
double min_val, max_val;
cv::Mat pdf_mat(num_test_pixels, 1, CV_32F, pdf.data());
minMaxLoc(pdf_mat, &min_val, &max_val);
threshold = max(threshold * max_val, min_val);
for (int i = 0; i < num_pixels; i++) {
    int y = i / width;
    int x = i % width;
    if (test_pdf(i, 0) >= threshold) {
        static_img.at<uchar>(y, x) = 255;
    } else {
        static_img.at<uchar>(y, x) = 0;
    }
}
cout<<"Compute Complete!"<<endl;
Mat dynamic_img;
test_img.copyTo(dynamic_img, static_img);

// Display images
imshow("Kernel Density Estimate", pdf_img);
imshow("Test Image", test_img);
imshow("Density Estimate on Test Image", test_pdf_img);
imshow("Static Regions", static_img);
imshow("Dynamic Regions", dynamic_img);
waitKey();

return 0;
}
