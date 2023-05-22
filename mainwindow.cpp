#include "mainwindow.h"
#include "ui_mainwindow.h"

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

/*-------------@brief：Opencv图像转QImage图像，方便在控件显示-------------*/
QImage matToQImage(cv::Mat cvImg)
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

MainWindow::MainWindow(QWidget *parent) :
QMainWindow(parent),
ui(new Ui::MainWindow),
m_timer(new QTimer)
{
    ui->setupUi(this);
    connect(m_timer,&QTimer::timeout,this,&MainWindow::readFarme);

    //打开SQLite数据库连接
    db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName("img1");
    db.open();
    query = new QSqlQuery;
    query->exec("create table if not exists face3("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL, "
                "image BLOB NOT NULL), "
                "matrix BLOB NOT NULL)");
    if(!db.open())
    {
        qDebug() << "Failed to open database";
        return;
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

/*--------------@brief：在label_video控件中显示摄像头画面--------------*/
void MainWindow::readFarme()
{
    m_capture.read(frame);      //从摄像头捕获一帧图像，存储在frame变量中
    image = matToQImage(frame); //opencv图像转换为QImage类型
    ui->label_video->setPixmap(QPixmap::fromImage(image));
}

/*--------------@brief：打开笔记本摄像头--------------*/
void MainWindow::on_open_clicked()
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

/*--------------@brief：关闭笔记本摄像头--------------*/
void MainWindow::on_close_clicked()
{
    m_timer->stop();
    m_capture.release();
    ui->label_video->clear();
}

/*--------------@brief：检测人脸并进行拍照并存入数据库--------------*/
void MainWindow::on_capture_clicked()
{      
    std::vector<cv::Rect> faces; //声明存储检测到的人脸矩形区域变量
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); //将图像转换为灰度图像
    cv::equalizeHist(gray, gray);                  //生成灰度图像直方图，这行代码影响了ExFaceFeatureInformation函数的调用
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
    std::vector<uchar> buffer;
    cv::imencode(".jpg", faceROI, buffer); //图像以JPEG格式进行编码，并将编码后的图像数据存储在buffer向量中
    std::vector<matrix<float,0,1>> data = ExFaceFeatureInformation(faceROI);
    savedatabase(ui->lineEdit_name->text(), buffer, data);

}

/*--------------@brief：从数据库中读取图像并显示--------------*/
void MainWindow::read()
{
    //从数据库中读取图像
    query->prepare("SELECT image FROM face3 WHERE id=:id");
    query->bindValue(":id", 1); //这里的1指的是图像的ID
    if (!query->exec() || !query->next())
    {
        qDebug() << "Failed to read image data";
        return;
    }
    //结果中提取图像数据，并将其存储在QByteArray对象byteArray中
    QByteArray imageData = query->value("image").toByteArray();
    QImage image;
    image.loadFromData(imageData, ".jpg"); //这里需要指定正确的图像格式和编码
    ui->label->setPixmap(QPixmap::fromImage(image));//显示图像
}

/*--------------@brief：点击save按钮存储人脸特征信息--------------*/
/*void MainWindow::on_save_clicked()
{
    ExFaceFeatureInformation();
}*/


/*--------------@brief：将人脸信息放入数据库--------------*/
void MainWindow::savedatabase(QString name, std::vector<uchar> buffer, std::vector<matrix<float,0,1>> face_descriptors)
{

    QByteArray byteArray(reinterpret_cast<const char*>(buffer.data()),
                         buffer.size());     //存储在buffer向量中的图像数据转换为QByteArray类型的数据,方便存储到数据库
    //std::vector<QByteArray> featureDataList; //存储特征数据的容器

    //存储在face_descriptors容器中的人脸特征信息转换为QByteArray类型的数据，方便存储到数据库
    /*for(const auto& feature : face_descriptors)
    {
        QByteArray featureData(reinterpret_cast<const char*>(feature.begin()),
                               feature.size() * sizeof(float));
        featureDataList.push_back(featureData);
    }*/

    QSqlQuery query; //创建数据库表
    //数据存储在名为"face3"的表中
    query.prepare("INSERT INTO face3 (name, image, matrix) VALUES (:name, :image, :matrix)");
    query.bindValue(":name", name);
    query.bindValue(":image", byteArray);
    //query.bindValue(":image", QVariant(face_descriptors));
    /*for(const auto& feature : featureDataList) //将人脸特征信息存入数据库
    {
        query.bindValue(":matrix", QVariant(feature));
    }*/
    for (const auto& feature : face_descriptors)
    {
        //将feature转换为适当的格式例如 JSON、二进制数据等）
        //这里以二进制数据形式进行示例
        QByteArray featureData(reinterpret_cast<const char*>(feature.begin()),
                               feature.size() * sizeof(float));

        // 将特征数据插入数据库的 "image" 列中
        //QSqlQuery insertQuery;
        query.bindValue(":matrix", QVariant(featureData));
        //insertQuery.exec();
    }

    if(!query.exec())
    {
        qDebug() << "Failed to insert face to database:"<<query.lastError().text();
    }
    else
    {
        qDebug() << "Face inserted successfully!";
    }
}


/*--------------@brief：将矩阵对象转换为字节数组--------------*/
std::vector<uchar> MainWindow::matrixToByteArray(const matrix<float,0,1>& mat)
{
    const float* data = mat.begin();
    size_t dataSize = mat.size() * sizeof(float);
    std::vector<uchar> byteArray(dataSize);
    uchar* byteArrayData = byteArray.data();
    memcpy(byteArrayData, reinterpret_cast<const uchar*>(data), dataSize);
    return byteArray;
}


/*--------------@brief：检测人脸特征--------------*/
std::vector<matrix<float,0,1>> MainWindow::ExFaceFeatureInformation(cv::Mat image)
{
    dlib::array2d<dlib::bgr_pixel> img;
    dlib::assign_image(img, dlib::cv_image<uchar>(image)); //将灰度图像加载到img中

    //定义人脸检测对象
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    //加载人脸检测模型
    dlib::shape_predictor sp;
    dlib::deserialize("D:/QT_Training/facedetection/face3/shape_predictor_5_face_landmarks.dat")>>sp;

    //加载人脸检测模型
    anet_type net; //anet_type是人脸识别的神经网络模型类型
    deserialize("D:/QT_Training/facedetection/face3/dlib_face_recognition_resnet_model_v1.dat")>>net;

    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; //声明人脸容器

    for(auto face : detector(img))
    {
        auto shape = sp(img, face);  //对人脸"face"进行关键点定位
        matrix<rgb_pixel> face_chip; //存储提取的人脸图像
        //提取人脸图像
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

















