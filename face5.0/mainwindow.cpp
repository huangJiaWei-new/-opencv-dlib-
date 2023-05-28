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
QSqlDatabase MainWindow::db = QSqlDatabase::addDatabase("QSQLITE"); //传入'faceRecognition.h'使用


/*-------------@brief：Opencv图像转QImage图像，方便在控件中显示-------------*/
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
    connect(m_timer,&QTimer::timeout,this,&MainWindow::displayFarme);

    db.setDatabaseName("img1.db"); //创建数据库文件
    db.open();

    query = new QSqlQuery;
    //创建名为face3的一张表，具有'name'、'colorImage'、'faceFeaIn'三个列,为姓名、人脸图像、人脸特征信息
    query->exec("CREATE TABLE IF NOT EXISTS face3 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, colorImage BLOB, faceFeaIn TEXT)");

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


/*--------------@brief：获取数据库连接'db',传入'faceRecognition'使用------------*/
QSqlDatabase MainWindow::getDatabaseConnection()
{
    return db;
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


/*--------------@brief：检测人脸进行拍照，存入数据库--------------*/
void MainWindow::on_captureFace_clicked()
{
    std::vector<cv::Rect> faces; //'faces'存储检测到的人脸矩形区域
    cv::Mat grayImage;
    cv::cvtColor(frame, grayImage, cv::COLOR_BGR2GRAY); //将图像转换为灰度图像
    cv::equalizeHist(grayImage, grayImage); //生成灰度图像直方图，这行代码影响了'ExFaceFeatureInformation'函数的调用
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 2,
                                 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30)); //将检测到的人脸存储在'faces'中
    if(faces.empty())
    {
        qDebug() <<"No face detected!";
        return;
    }
    else
    {
        qDebug() <<"face success detected!";
    }

    //只处理第一张人脸,灰度图像用于计算人脸特征，速度更快
    cv::Rect faceRect = faces[0];       //faceRect存储检测到的第一个人脸的矩形框
    cv::Mat  faceROI  = grayImage(faceRect); //faceROI从灰度图像gray中提取出的人脸区域

    //彩色人脸图像存于数据库，在识别时进行显示
    cv::Mat colorFaceROI = frame(faceRect); //彩色人脸图像
    cv::cvtColor(colorFaceROI, colorFaceROI, cv::COLOR_BGR2RGB); //'BGR'转为'RGB'
    std::vector<uchar> faceImgbuffer;
    cv::imencode(".jpg", colorFaceROI, faceImgbuffer); //图像以JPEG格式进行编码，并将编码后的图像数据存储在"faceImgbuffer"中

    std::vector<matrix<float,0,1>> data = ExFaceFeatureInformation(faceROI); //提取第一张人脸图像的特征信息
    storeData(ui->lineEdit_name->text(), faceImgbuffer, data); //将姓名，人脸图像，人脸特征信息存入数据库
}


/*--------------@brief：从数据库中读取图像，在控件中显示--------------*/
void MainWindow::on_searchPeople_clicked()
{
    //对数据库中的人进行查询
    query->prepare("SELECT colorImage, faceFeaIn FROM face3 WHERE name = ?");
    query->addBindValue(ui->p_name->text());
    if (!query->exec() || !query->next())
    {
        qDebug() << "Failed to read img1.db data";
        return;
    }

    //结果中提取图像数据,并在label控件中显示
    QByteArray imageData = query->value("colorImage").toByteArray();
    QImage image;
    image.loadFromData(imageData, ".jpg"); //这里需要指定正确的图像格式和编码

    QImage scaledImage = image.scaled(ui->label->size().width(), ui->label->size().height(),
                                      Qt::KeepAspectRatio); //让图片适应label控件大小
    ui->label->setPixmap(QPixmap::fromImage(scaledImage));  //显示人脸图像
    ui->label->setScaledContents(true);

    //从数据库中提取人脸特征信息，并打印
    QString featureInfSt = query->value("faceFeaIn").toString();
    qDebug()<< "该人员的人脸特征信息为:"<<featureInfSt;

}


/*--------------@brief：点击打开人脸识别界面按钮，打开'faceRcognition.ui'窗口------------*/
void MainWindow::on_identify_clicked()
{
    faceRecognition *m_faceRecognition = new faceRecognition;
    m_faceRecognition->setModal(true);
    m_faceRecognition->exec();
}


/*--------------@brief：将matrix<float, 0, 1>转换为vector<float>-------------*/
std::vector<float> MainWindow::matrixTovector(const dlib::matrix<float, 0, 1>& matrix)
{
    std::vector<float> vector(matrix.size());
    std::memcpy(vector.data(), matrix.begin(), matrix.size() * sizeof(float));
    return vector;
}


/*--------------@brief：将vector<dlib::matrix<float,0,1>>转换为QString--------------*/
QString MainWindow::matrixToString(const std::vector<dlib::matrix<float,0,1>>& face_descriptors)
{
    QStringList dataStrings; //存储转换后的矩阵数据字符串
    for(const auto& matrix : face_descriptors)
    {
        QStringList matrixData;
        for(float value : matrix)
            matrixData << QString::number(value); //将当前矩阵数据转为字符串赋值给"matrixData"列表

        QString matrixString = matrixData.join(","); //将字符串元素连接起来，用逗号隔离
        dataStrings << matrixString; //每个矩阵的字符串形式"matrixString"添加到"dataStrings"列表中
    }
    QString dataString = dataStrings.join(";");

    return dataString;
}


/*--------------@brief：在label_video控件中显示摄像头画面--------------*/
void MainWindow::displayFarme()
{
    m_capture.read(frame);      //从摄像头捕获一帧图像，存储在frame变量中
    image = matToQImage(frame); //opencv图像转换为QImage类型
    ui->label_video->setPixmap(QPixmap::fromImage(image));
}


/*--------------@brief：点击save按钮存储人脸特征信息--------------*/
/*void MainWindow::on_save_clicked()
{
    ExFaceFeatureInformation();
}*/


/*--------------@brief：将姓名，人脸图像，人脸特征信息存入数据库--------------*/
void MainWindow::storeData(QString name, std::vector<uchar> buffer, std::vector<matrix<float,0,1>> face_descriptors)
{
    //将std::vector<matrix<float,0,1>>转换为QString类型，方便存入数据库
    QString featureInfSt = matrixToString(face_descriptors);
    qDebug() << "dataString:" << featureInfSt;

    query->prepare("INSERT INTO face3 (name, colorImage, faceFeaIn) VALUES (?, ?, ?)");
    query->addBindValue(name);
    //存储在buffer向量中的图像数据转换为QByteArray类型的数据,方便存储到数据库
    query->addBindValue(QByteArray(reinterpret_cast<const char*>(buffer.data()), buffer.size()));
    query->addBindValue(featureInfSt);

    if(!query->exec())
    {
        qDebug() << "Failed to insert face to database:"<<query->lastError().text();
    }
    else
    {
        qDebug() << "Face inserted successfully!";
    }
}


/*--------------@brief：提取人脸特征信息，存储在"face_descriptors"中--------------*/
std::vector<matrix<float,0,1>> MainWindow::ExFaceFeatureInformation(cv::Mat grayImg)
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
                                                      //可能不用声明容器，每次都是进行识别一张脸

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
    //可能不用声明容器，每次都是进行识别一张脸
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















