#ifndef FACERECOGNITION_H
#define FACERECOGNITION_H

#include "ui_facerecognition.h"
#include "opencv2/opencv.hpp"

#include <QDialog>
#include <QTimer>
#include <QBuffer>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QDebug>
#include <QDateTime>
#include <QSqlError>

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>

using namespace dlib;
using namespace std;

class faceRecognition : public QDialog
{
    Q_OBJECT

public:
    explicit faceRecognition(QWidget *parent = nullptr);
    ~faceRecognition();

private:
    void displayFarme();
    QImage matToQImage(cv::Mat cvImg);
    std::vector<matrix<float,0,1>> ExFaceFeatureInformation(cv::Mat grayImg);
    void CalculateMatchingDegree(std::vector<matrix<float,0,1>> faceFeatureInf);
    dlib::matrix<float, 0, 1> StringTomatrix(const QString& dataString);

private slots:
    void on_open_clicked();
    void on_close_clicked();
    void on_recognition_clicked();
    void on_test_clicked();

private:
    Ui::faceRecognition *ui;

    cv::CascadeClassifier faceCascade; //声明人脸检测对象
    cv::VideoCapture m_capture;        //声明捕获视频帧的对象
    cv::Mat frame;                     //声明储存视频的单个帧对象
    QImage  image;
    QTimer *m_timer;  //声明定时器对象
    QSqlDatabase db;
};

#endif // FACERECOGNITION_H
