#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include "faceRecognition.h"
#include "ui_mainwindow.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>

#include <QMainWindow>
#include <QTimer>
#include <QBuffer>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDebug>
#include <QDateTime>
#include <QVariant>
#include <QSqlError>
#include <QDataStream>

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>
#include <dlib/misc_api.h>

using namespace dlib;
using namespace std;
class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
explicit MainWindow(QWidget *parent = 0);
~MainWindow();

public:
    static QSqlDatabase db;  //声明数据库连接的对象
    QSqlDatabase getDatabaseConnection();

private:
    std::vector<float> matrixTovector(const dlib::matrix<float, 0, 1>& matrix);
    QString matrixToString(const std::vector<dlib::matrix<float,0,1>>& face_descriptors);
    void displayFarme();
    void storeData(QString name, std::vector<uchar> buffer, std::vector<matrix<float,0,1>> face_descriptors);
    std::vector<matrix<float,0,1>> ExFaceFeatureInformation(cv::Mat grayImg);

private slots:
    void on_open_clicked();
    void on_close_clicked();
    void on_captureFace_clicked();
    void on_searchPeople_clicked();
    void on_identify_clicked();

private:
    Ui::MainWindow *ui;
    cv::CascadeClassifier faceCascade; //声明人脸检测对象
    cv::VideoCapture m_capture;        //声明捕获视频帧的对象
    cv::Mat frame;    //声明储存视频的单个帧对象
    QSqlQuery *query; //声明数据库表指针
    QTimer *m_timer;  //声明定时器对象
    QImage  image;
};

#endif // MAINWINDOW_H
