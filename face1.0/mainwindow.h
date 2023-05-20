#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "ui_mainwindow.h"

#include <QMainWindow>
#include <QTimer>
#include <QBuffer>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDebug>
#include <QDateTime>
#include <QVariant>

//#include <opencv2/face.hpp>
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
//#include <opencv2\imgproc\types_c.h>

#include <dlib/dnn.h>
//#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib\opencv.h>
#include <dlib/matrix.h>
#include <dlib/image_loader/jpeg_loader.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;
//namespace Ui {
//class MainWindow;
//}
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
explicit MainWindow(QWidget *parent = 0);
~MainWindow();

public:

private:
    //将矩阵对象转换为字节数组
    std::vector<uchar> matrixToByteArray(const matrix<float,0,1>& mat);
    void readFarme();
    void capture();     //抓图
    void read();        //读取数据库
    void savedatabase(QString name, std::vector<uchar> buffer, std::vector<matrix<float,0,1>> face_descriptors);
    //std::vector<matrix<float,0,1>> ExFaceFeatureInformation(cv::Mat image);
    void ExFaceFeatureInformation(cv::Mat image);

private slots:
    void on_open_clicked();
    void on_close_clicked();
    void on_capture_clicked();


private:
    Ui::MainWindow *ui;
    cv::CascadeClassifier faceCascade; //声明人脸检测对象
    cv::VideoCapture m_capture;        //声明捕获视频帧的对象
    cv::Mat frame;   //声明储存视频的单个帧对象
    QSqlDatabase db; //声明数据库连接的对象
    QSqlQuery *query; //声明数据库表指针
    QTimer *m_timer; //声明定时器对象
    QImage  image;
};

#endif // MAINWINDOW_H
