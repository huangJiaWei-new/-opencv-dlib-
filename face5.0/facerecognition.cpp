﻿#include "faceRecognition.h"
#include "ui_facerecognition.h"
#include "mainwindow.h"

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

/*---------------------------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------------------------*/
faceRecognition::faceRecognition(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::faceRecognition),
    m_timer(new QTimer)
{
    ui->setupUi(this);
    connect(m_timer,&QTimer::timeout,this,&faceRecognition::displayFarme);

    db = QSqlDatabase::database("qt_sql_default_connection"); //连接到img1.db数据库
}

faceRecognition::~faceRecognition()
{
    delete ui;
}


/*--------------@brief：点击"open"按钮，打开摄像头-------------*/
void faceRecognition::on_open_clicked()
{
    m_capture.open(0);
    m_timer->start(50);
    /*加载人脸检测文件*/
    faceCascade.load("D:/QT_Training/facedetection/opencv/opencv-build/install/etc/haarcascades/haarcascade_frontalface_alt.xml");
    if(!faceCascade.load("D:/QT_Training/facedetection/opencv/opencv-build/install/etc/haarcascades/haarcascade_frontalface_alt.xml"))
    {
        qDebug() << "Failed to load face cascade!";
        return;
    }
}


/*--------------@brief：点击"close"按钮，关闭笔记本摄像头--------------*/
void faceRecognition::on_close_clicked()
{
    m_timer->stop();
    m_capture.release();
    ui->video->clear();
}


/*--------------@brief：点击"在线识别"按钮，进行人脸识别-------------*/
void faceRecognition::on_recognition_clicked()
{
    std::vector<cv::Rect> faces; //"faces"存储检测到的人脸矩形区域
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //将图像转换为灰度图像
    cv::equalizeHist(gray, gray); //生成灰度图像直方图，这行代码影响了ExFaceFeatureInformation函数的调用

    //将检测到的人脸存储在"faces"中
    faceCascade.detectMultiScale(gray, faces, 1.1, 2,
                                 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
    if(faces.empty())
    {
        qDebug() <<"No face detected!";
        return;
    }
    else
    {
        qDebug() <<"face success detected!";
    }

    //只处理第一张人脸
    cv::Rect faceRect = faces[0];       //faceRect存储检测到的第一个人脸的矩形框
    cv::Mat  faceROI  = gray(faceRect); //faceROI从灰度图像gray中提取出的人脸区域
    std::vector<uchar> faceImgbuffer;
    cv::imencode(".jpg", faceROI, faceImgbuffer); //图像以JPEG格式进行编码，并将编码后的图像数据存储在"faceImgbuffer"中
    std::vector<matrix<float,0,1>> data = ExFaceFeatureInformation(faceROI);
    CalculateMatchingDegree(data);
}


/*--------------@brief：点击"测试"按钮,打印数据库的数据--------------*/
void faceRecognition::on_test_clicked()
{
    std::vector<matrix<float, 0, 1>> faceLibrary;  //"faceLibrary"存储人脸特征向量
    std::vector<QString> nameLibrary;              //存储人脸对应的姓名

    //从名为face3的表中提取人员信息
    QSqlQuery query(db);
    query.prepare("SELECT name, colorImage, faceFeaIn FROM face3");

    if(!query.exec()) //执行查询语句
    {
        qDebug() << "Failed to read img1.db data";
        return;
    }

    for(int i = 0; query.next(); i++) //获取数据库人员数据的所有记录
    {
        QString name         = query.value("name").toString();
        QByteArray buffer    = query.value("colorImage").toByteArray();
        QString featureInfSt = query.value("faceFeaIn").toString();

        nameLibrary.push_back(name); //处理人脸图像
        dlib::matrix<float,0,1> faceDescriptor = StringTomatrix(featureInfSt); //将数据库存储的人脸特征信息进行格式转换
        faceLibrary.push_back(faceDescriptor);
    }

    for(const QString& name : nameLibrary) //打印数据库人员名字
    {
        qDebug() << "Name: " << name;
    }

    for(const auto& descriptor : faceLibrary) //打印数据库人脸特征向量
    {
        // 在这里对每个人脸特征向量进行处理
        for(long i = 0; i < descriptor.size(); i++)
        {
            qDebug() << descriptor(i);
        }
    }
}


/*--------------@brief：在video控件中显示摄像头画面--------------*/
void faceRecognition::displayFarme()
{
    m_capture.read(frame);      //从摄像头捕获一帧图像，存储在frame变量中
    image = matToQImage(frame); //opencv图像转换为QImage类型
    ui->video->setPixmap(QPixmap::fromImage(image));
}


/*-------------@brief：Opencv图像转QImage图像，方便在控件中显示-------------*/
QImage faceRecognition::matToQImage(cv::Mat cvImg)
{
    QImage qImg;
    if(cvImg.channels() == 3) //3 channels color image
    {
        cv::cvtColor(cvImg, cvImg, cv::COLOR_BGR2RGB);
        qImg = QImage((const unsigned char*)(cvImg.data),
        cvImg.cols, cvImg.rows,
        cvImg.cols*cvImg.channels(),
        QImage::Format_RGB888);
    }
    else if(cvImg.channels() == 1) //grayscale image
    {
        qImg = QImage((const unsigned char*)(cvImg.data),
        cvImg.cols, cvImg.rows,
        cvImg.cols*cvImg.channels(),
        QImage::Format_Indexed8);
    }
    else
    {
        qImg = QImage((const unsigned char*)(cvImg.data),
        cvImg.cols, cvImg.rows,
        cvImg.cols*cvImg.channels(),
        QImage::Format_RGB888);
    }
    return qImg;
}


/*--------------@brief：提取人脸特征信息，存储在"face_descriptors"中--------------*/
std::vector<matrix<float,0,1>> faceRecognition::ExFaceFeatureInformation(cv::Mat grayImg)
{
    dlib::array2d<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<uchar>(grayImg)); //将灰度图像加载到img中

    //定义人脸检测对象
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    //加载人脸检测模型
    dlib::shape_predictor sp;
    dlib::deserialize("D:/QT_Training/facedetection/face3/shape_predictor_5_face_landmarks.dat")>>sp;

    //加载人脸特征提取模型
    anet_type net; //anet_type是人脸识别的神经网络模型类型
    deserialize("D:/QT_Training/facedetection/face3/dlib_face_recognition_resnet_model_v1.dat")>>net;

    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; //声明人脸容器

    for(auto face : detector(img))   //循环遍历每个检测到的人脸区域
    {
        auto shape = sp(img, face);  //对人脸"face"进行关键点定位
        matrix<rgb_pixel> face_chip; //存储提取的人脸图像
        //将当前照片提取的人脸图像放入"face_chip"中
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(face_chip); //将提取的人脸图像存储到"faces"容器中
    }

    //将faces中的每个人脸图像转换为128D向量,也就是提取人脸特征信息，存储在"face_descriptors"容器中
    //matrix<float,0,1>代表一个矩阵
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);

    for(const auto& descriptor : face_descriptors) //打印数据库的值
    {
        for(long i = 0; i < descriptor.size(); i++)
        {
            qDebug() << descriptor(i);
        }
    }
    return face_descriptors;
}


/*--------------@brief：计算人脸特征信息匹配度-------------*/
void faceRecognition::CalculateMatchingDegree(std::vector<matrix<float,0,1>> faceFeatureInf)
{
    std::vector<QString> nameLibrary;              //存储人脸对应的姓名的容器
    std::vector<QImage>  faceImageLibrary;         //存储人脸图像的容器
    std::vector<matrix<float, 0, 1>> faceFeatureInfLibrary;  //"faceLibrary"存储人脸特征向量的容器
    float minDistance = 0.6;    //用于循环比较，找出最匹配的人脸，'0.6'是官方训练网络使用的决策阈值
    size_t   minIndex    = -1;  //用于记录最小距离对应的索引


    //从名为face3的表中提取人员信息
    QSqlQuery query(db);
    query.prepare("SELECT name, colorImage, faceFeaIn FROM face3");

    if(!query.exec())
    {
        qDebug() << "Failed to read img1.db data";
        return;
    }

    for(int i = 0; query.next(); i++) //获取数据库人员数据的所有记录
    {       
        QString name             = query.value("name").toString();
        QByteArray colorImage    = query.value("colorImage").toByteArray();
        QString featureInfSt     = query.value("faceFeaIn").toString();

        //处理人名
        nameLibrary.push_back(name);

        //处理图像数据
        QImage image;
        image.loadFromData(colorImage, ".jpg"); //这里需要指定正确的图像格式和编码
        faceImageLibrary.push_back(image);

        //处理人脸特征信息
        dlib::matrix<float,0,1> faceDescriptor = StringTomatrix(featureInfSt); //将数据库存储的人脸特征信息进行格式转换
        faceFeatureInfLibrary.push_back(faceDescriptor);
    }

    //遍历待识别的人脸特征向量
    for(const auto& face_descriptor : faceFeatureInf)
    {
        for(size_t i = 0; i < faceFeatureInfLibrary.size(); ++i) //在人脸库中进行匹配
        {     
            float distance = length(face_descriptor - faceFeatureInfLibrary[i]); //'length'函数计算欧氏距离
            qDebug() <<"distance" <<distance;
            if(distance < minDistance) //找出最匹配的人脸
            {
                minDistance = distance;
                minIndex = i;
            }
        }
    }

    //判断距离是否小于阈值（用于判断是否匹配）
    if(minIndex != -1)
    {
        //显示匹配人员姓名
        QString recognizedName;
        qDebug()<<"识别成功";
        recognizedName = nameLibrary[minIndex];
        ui->name->setText(recognizedName);
        //显示显示匹配人员图像
        QImage grayImage = faceImageLibrary[minIndex];
        QImage scaledImage = grayImage.scaled(ui->face->size().width(), ui->face->size().height(),
                                          Qt::KeepAspectRatio); //让图片适应label控件大小
        ui->face->setPixmap(QPixmap::fromImage(scaledImage));
    }
}


/*--------------@brief：将QString转换为dlib::matrix<float,0,1>--------------*/
dlib::matrix<float, 0, 1> faceRecognition::StringTomatrix(const QString& dataString)
{
    QStringList dataStringList = dataString.split(';'); //将"QString"拆分为单独的矩阵字符串
    dlib::matrix<float,0,1> faceDescriptor;

    for(const QString& matrixString : dataStringList)
    {
        QStringList matrixData = matrixString.split(','); //将矩阵字符串拆分为单个值
        std::vector<float> matrixValues;

        for(const QString& valueString : matrixData)
        {
            float value = valueString.toFloat(); //将值字符串转换为浮点数
            matrixValues.push_back(value); //将值添加到矩阵值向量
        }

        faceDescriptor = dlib::mat(matrixValues);
    }

    return faceDescriptor;
}






































