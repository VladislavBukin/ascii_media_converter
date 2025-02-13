// main.cpp
#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QTextEdit>
#include <QLineEdit>
#include <QTabWidget>
#include <QFormLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QProgressBar>
#include <QTimer>
#include <QPalette>
#include <QStyleFactory>
#include <QProcess>
#include <QDir>
#include <QDateTime>
#include <QCloseEvent>
#include <QMediaPlayer>
#include <QUrl>
#include <QFile>
#include <QTextStream>
#include <QVector>
#include <QFont>
#include <QComboBox>

// OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Класс PreprocessingThread – поток для предобработки видео (и GIF),
// в котором каждый кадр преобразуется в HTML-строку с ASCII-артом.
#include <QThread>
class PreprocessingThread : public QThread {
    Q_OBJECT
public:
    PreprocessingThread(const QString &videoPath, int desiredWidth, const QString &asciiChars, QObject* parent = nullptr)
        : QThread(parent),
          m_videoPath(videoPath),
          m_desiredWidth(desiredWidth),
          m_asciiChars(asciiChars),
          m_runFlag(true)
    {}

    void stop() { m_runFlag = false; }

signals:
    void finished(const QVector<QString>& frames, double fps);
    void progress(int processed, int total);

protected:
    void run() override {
        QVector<QString> asciiFrames;
        cv::VideoCapture cap(m_videoPath.toStdString());
        if (!cap.isOpened()) {
            emit finished(asciiFrames, 0.0);
            return;
        }
        double realFps = cap.get(cv::CAP_PROP_FPS);
        if (realFps <= 0)
            realFps = 24.0;

        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        if (totalFrames < 1)
            totalFrames = 0;

        int processedCount = 0;
        int asciiLen = m_asciiChars.length();

        cv::Mat frame;
        while (m_runFlag) {
            if (!cap.read(frame) || frame.empty())
                break;

            int h = frame.rows;
            int w = frame.cols;
            double aspectRatio = static_cast<double>(h) / w;
            int newH = static_cast<int>(m_desiredWidth * aspectRatio * 0.55);
            if (newH < 1)
                newH = 1;

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(m_desiredWidth, newH));
            QVector<QString> lines;
            // Проходим по каждому пикселю
            for (int row = 0; row < newH; ++row) {
                QString rowStr;
                for (int col = 0; col < m_desiredWidth; ++col) {
                    cv::Vec3b pixel = resized.at<cv::Vec3b>(row, col);
                    int b = pixel[0];
                    int g = pixel[1];
                    int r = pixel[2];
                    double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    int idx = static_cast<int>(gray / 255 * (asciiLen - 1));
                    if (idx < 0) idx = 0;
                    if (idx >= asciiLen) idx = asciiLen - 1;
                    QChar ch = m_asciiChars.at(idx);
                    rowStr += QString("<span style=\"color: rgb(%1,%2,%3)\">%4</span>")
                              .arg(r).arg(g).arg(b).arg(ch);
                }
                lines.append(rowStr);
            }
            QString htmlFrame = lines.join("<br>");
            asciiFrames.append(htmlFrame);

            processedCount++;
            if (totalFrames > 0)
                emit progress(processedCount, totalFrames);
        }
        cap.release();
        emit finished(asciiFrames, realFps);
    }

private:
    QString m_videoPath;
    int m_desiredWidth;
    QString m_asciiChars;
    bool m_runFlag;
};

//
// Класс AsciiArtApp – главное окно приложения.
// Реализованы три вкладки: "Изображение в ASCII", "Видео в ASCII" и "GIF в ASCII".
// а для GIF воспроизведение зациклено.
// Добавлены пресеты набора символов – QComboBox для каждой вкладки.
class AsciiArtApp : public QMainWindow {
    Q_OBJECT
public:
    AsciiArtApp(QWidget* parent = nullptr)
        : QMainWindow(parent),
          m_preprocThread(nullptr),
          m_videoFps(24.0),
          m_videoLength(0),
          m_videoStartTime(0),
          m_currentFrameIndex(0),
          m_gifPreprocThread(nullptr),
          m_gifFps(24.0),
          m_gifLength(0),
          m_gifStartTime(0),
          m_currentGifFrameIndex(0)
    {
        setWindowTitle("Генератор ASCII-арта");
        resize(700, 700);
        QApplication::setStyle(QStyleFactory::create("Fusion"));

        QPalette darkPalette;
        darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::WindowText, Qt::white);
        darkPalette.setColor(QPalette::Base, QColor(15, 15, 15));
        darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::Text, Qt::white);
        darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
        darkPalette.setColor(QPalette::ButtonText, Qt::white);
        QApplication::setPalette(darkPalette);

        m_monospaceFont = QFont("Courier New", 10);

        m_tabWidget = new QTabWidget(this);
        setCentralWidget(m_tabWidget);

        // Вкладка для изображения
        m_imageTab = new QWidget;
        // Вкладка для видео
        m_videoTab = new QWidget;
        // Вкладка для GIF
        m_gifTab = new QWidget;
        m_tabWidget->addTab(m_imageTab, "Изображение в ASCII");
        m_tabWidget->addTab(m_videoTab, "Видео в ASCII");
        m_tabWidget->addTab(m_gifTab, "GIF в ASCII");

        initImageTab();
        initVideoTab();
        initGifTab();

        // Инициализация медиаплеера для аудио (видео)
        m_player = new QMediaPlayer(this);

        m_playTimer = new QTimer(this);
        m_playTimer->setInterval(15);
        connect(m_playTimer, &QTimer::timeout, this, &AsciiArtApp::showNextFrame);

        // Таймер для воспроизведения GIF (зациклено)
        m_gifPlayTimer = new QTimer(this);
        m_gifPlayTimer->setInterval(15);
        connect(m_gifPlayTimer, &QTimer::timeout, this, &AsciiArtApp::showNextGifFrame);
    }

    ~AsciiArtApp() {
        // Очистка для видео
        if (m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->quit();
            m_preprocThread->wait();
            delete m_preprocThread;
        }
        // Очистка для GIF
        if (m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->quit();
            m_gifPreprocThread->wait();
            delete m_gifPreprocThread;
        }
    }

protected:
    void closeEvent(QCloseEvent* event) override {
        m_playTimer->stop();
        m_gifPlayTimer->stop();
        if (m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->quit();
            m_preprocThread->wait();
        }
        if (m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->quit();
            m_gifPreprocThread->wait();
        }
        if (m_player)
            m_player->stop();
        QMainWindow::closeEvent(event);
    }

private slots:
    // --- Слоты для работы с изображением ---
    void openImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите изображение",
                                                        "", "Изображения (*.png *.jpg *.jpeg *.bmp *.gif)");
        if (!fileName.isEmpty()) {
            m_currentImagePath = fileName;
            convertImageToAscii();
        }
    }

    void convertImageToAscii() {
        if (m_currentImagePath.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Сначала выберите изображение.");
            return;
        }
        cv::Mat img = cv::imread(m_currentImagePath.toStdString());
        if (img.empty()) {
            QMessageBox::warning(this, "Ошибка", "Не удалось открыть изображение.");
            return;
        }

        int dw = m_imgSpinWidth->value();
        QString asciiChars = m_imgCharsetEdit->text();
        if (asciiChars.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }

        int h_ = img.rows;
        int w_ = img.cols;
        double aspect = static_cast<double>(h_) / w_;
        int newH = static_cast<int>(dw * aspect * 0.55);
        if (newH < 1)
            newH = 1;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(dw, newH));
        QVector<QString> lines;
        int length = asciiChars.length();
        for (int row = 0; row < newH; ++row) {
            QString rowStr;
            for (int col = 0; col < dw; ++col) {
                cv::Vec3b pixel = resized.at<cv::Vec3b>(row, col);
                int b = pixel[0];
                int g = pixel[1];
                int r = pixel[2];
                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                int idx = static_cast<int>(gray / 255 * (length - 1));
                if (idx < 0) idx = 0;
                if (idx >= length) idx = length - 1;
                QChar ch = asciiChars.at(idx);
                rowStr += QString("<span style=\"color: rgb(%1,%2,%3)\">%4</span>")
                          .arg(r).arg(g).arg(b).arg(ch);
            }
            lines.append(rowStr);
        }
        QString html = lines.join("<br>");
        m_imgAsciiDisplay->setHtml(html);
    }

    void saveHtmlImage() {
        QString htmlCode = m_imgAsciiDisplay->toHtml();
        if (htmlCode.trimmed().isEmpty()) {
            QMessageBox::information(this, "Пусто", "Нет ASCII-арта.");
            return;
        }
        QString fileName = QFileDialog::getSaveFileName(this, "Сохранить HTML", "", "HTML файлы (*.html)");
        if (!fileName.isEmpty()) {
            QFile file(fileName);
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&file);
                out << htmlCode;
                file.close();
                QMessageBox::information(this, "Успех", "Сохранено:\n" + fileName);
            } else {
                QMessageBox::critical(this, "Ошибка записи", "Не удалось записать файл.");
            }
        }
    }

    // --- Слоты для работы с видео ---
    void openVideo() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите видео",
                                                        "", "Видео файлы (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)");
        if (!fileName.isEmpty()) {
            m_currentVideoPath = fileName;
            QMessageBox::information(this, "Выбрано видео", m_currentVideoPath);
        }
    }

    void startPreprocessing() {
        if (m_currentVideoPath.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Сначала выберите видеофайл.");
            return;
        }
        int w = m_videoSpinWidth->value();
        QString chars = m_videoCharsetEdit->text();
        if (chars.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }

        m_videoAsciiDisplay->clear();
        m_btnPreprocPlay->setEnabled(false);
        m_btnStop->setEnabled(true);
        m_progressVideo->setValue(0);

        m_preprocThread = new PreprocessingThread(m_currentVideoPath, w, chars, this);
        connect(m_preprocThread, &PreprocessingThread::finished, this, &AsciiArtApp::onPreprocessingFinished);
        connect(m_preprocThread, &PreprocessingThread::progress, this, &AsciiArtApp::onPreprocessingProgress);
        m_preprocThread->start();
    }

    void onPreprocessingFinished(const QVector<QString>& frames, double fps) {
        m_asciiFrames = frames;
        m_videoFps = fps;
        m_videoLength = frames.size();

        if (m_preprocThread) {
            m_preprocThread->quit();
            m_preprocThread->wait();
            m_preprocThread->deleteLater();
            m_preprocThread = nullptr;
        }

        if (m_asciiFrames.isEmpty() || m_videoLength == 0) {
            QMessageBox::warning(this, "Ошибка", "Не удалось получить кадры из видео.");
            m_btnPreprocPlay->setEnabled(true);
            m_btnStop->setEnabled(false);
            return;
        }
        m_progressVideo->setValue(100);
        extractAudio(m_currentVideoPath);

        m_videoStartTime = QDateTime::currentMSecsSinceEpoch();
        m_currentFrameIndex = 0;
        m_playTimer->start();
    }

    void onPreprocessingProgress(int processed, int total) {
        if (total > 0) {
            int percentage = static_cast<int>(static_cast<double>(processed) / total * 100);
            m_progressVideo->setValue(percentage);
        } else {
            m_progressVideo->setValue(0);
        }
    }

    void extractAudio(const QString &path) {
        // Извлекаем аудио с помощью ffmpeg
        QString tmpDir = QDir::tempPath();
        QString outWav = tmpDir + "/temp_audio.wav";
        QString program = "ffmpeg";
        QStringList arguments;
        arguments << "-y" << "-i" << path << "-vn" << "-acodec" << "pcm_s16le" << outWav;

        QProcess process;
        process.start(program, arguments);
        if (!process.waitForFinished(10000)) {
            QMessageBox::warning(this, "Ошибка ffmpeg", "Timeout при извлечении звука.");
            return;
        }
        if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0) {
            QString err = process.readAllStandardError();
            QMessageBox::warning(this, "Ошибка ffmpeg", "Не удалось извлечь/преобразовать звук:\n" + err);
            return;
        }
        // Воспроизведение аудио с помощью QMediaPlayer
        m_player->setSource(QUrl::fromLocalFile(outWav));
        m_player->play();
    }

    void showNextFrame() {
        qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
        qint64 elapsed = currentTime - m_videoStartTime;
        int frameIndex = static_cast<int>(elapsed / 1000.0 * m_videoFps);
        if (frameIndex >= m_videoLength) {
            stopVideo();
            return;
        }
        if (frameIndex != m_currentFrameIndex) {
            m_videoAsciiDisplay->setHtml(m_asciiFrames[frameIndex]);
            m_currentFrameIndex = frameIndex;
        }
    }

    void stopVideo() {
        m_playTimer->stop();
        if (m_player)
            m_player->stop();
        m_btnPreprocPlay->setEnabled(true);
        m_btnStop->setEnabled(false);
        if (m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->quit();
            m_preprocThread->wait();
            m_preprocThread->deleteLater();
            m_preprocThread = nullptr;
        }
    }


    // --- Слоты для работы с GIF ---
    void openGif() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите GIF",
                                                        "", "GIF файлы (*.gif)");
        if (!fileName.isEmpty()) {
            m_currentGifPath = fileName;
            QMessageBox::information(this, "Выбран GIF", m_currentGifPath);
        }
    }

    void startPreprocessingGif() {
        if (m_currentGifPath.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Сначала выберите GIF-файл.");
            return;
        }
        int w = m_gifSpinWidth->value();
        QString chars = m_gifCharsetEdit->text();
        if (chars.isEmpty()) {
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }

        m_gifAsciiDisplay->clear();
        m_btnPreprocGif->setEnabled(false);
        m_btnStopGif->setEnabled(true);
        m_progressGif->setValue(0);

        m_gifPreprocThread = new PreprocessingThread(m_currentGifPath, w, chars, this);
        connect(m_gifPreprocThread, &PreprocessingThread::finished, this, &AsciiArtApp::onGifPreprocessingFinished);
        connect(m_gifPreprocThread, &PreprocessingThread::progress, this, &AsciiArtApp::onGifPreprocessingProgress);
        m_gifPreprocThread->start();
    }

    void onGifPreprocessingFinished(const QVector<QString>& frames, double fps) {
        m_gifAsciiFrames = frames;
        m_gifFps = fps;
        m_gifLength = frames.size();

        if (m_gifPreprocThread) {
            m_gifPreprocThread->quit();
            m_gifPreprocThread->wait();
            m_gifPreprocThread->deleteLater();
            m_gifPreprocThread = nullptr;
        }

        if (m_gifAsciiFrames.isEmpty() || m_gifLength == 0) {
            QMessageBox::warning(this, "Ошибка", "Не удалось получить кадры из GIF.");
            m_btnPreprocGif->setEnabled(true);
            m_btnStopGif->setEnabled(false);
            return;
        }
        m_progressGif->setValue(100);

        m_gifStartTime = QDateTime::currentMSecsSinceEpoch();
        m_currentGifFrameIndex = 0;
        m_gifPlayTimer->start();
    }

    void onGifPreprocessingProgress(int processed, int total) {
        if (total > 0) {
            int percentage = static_cast<int>(static_cast<double>(processed) / total * 100);
            m_progressGif->setValue(percentage);
        } else {
            m_progressGif->setValue(0);
        }
    }

    void showNextGifFrame() {
        if (m_gifLength == 0)
            return;
        qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
        qint64 elapsed = currentTime - m_gifStartTime;
        // Зацикленное воспроизведение: используем оператор % для циклического перехода
        int frameIndex = static_cast<int>(elapsed / 1000.0 * m_gifFps) % m_gifLength;
        if (frameIndex != m_currentGifFrameIndex) {
            m_gifAsciiDisplay->setHtml(m_gifAsciiFrames[frameIndex]);
            m_currentGifFrameIndex = frameIndex;
        }
    }

    void stopGif() {
        m_gifPlayTimer->stop();
        m_btnPreprocGif->setEnabled(true);
        m_btnStopGif->setEnabled(false);
        if (m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->quit();
            m_gifPreprocThread->wait();
            m_gifPreprocThread->deleteLater();
            m_gifPreprocThread = nullptr;
        }
    }

    // Слот для сохранения исходного GIF

private:
    // --- Инициализация вкладки "Изображение" ---
    void initImageTab() {
        QVBoxLayout* layout = new QVBoxLayout;
        m_imageTab->setLayout(layout);

        QHBoxLayout* topLayout = new QHBoxLayout;
        m_btnOpenImg = new QPushButton("Открыть изображение");
        connect(m_btnOpenImg, &QPushButton::clicked, this, &AsciiArtApp::openImage);
        topLayout->addWidget(m_btnOpenImg);

        QGroupBox* widthGroup = new QGroupBox("Ширина");
        QFormLayout* widthForm = new QFormLayout;
        m_imgSpinWidth = new QSpinBox;
        m_imgSpinWidth->setRange(10, 800);
        m_imgSpinWidth->setValue(80);
        widthForm->addRow("Символов:", m_imgSpinWidth);
        widthGroup->setLayout(widthForm);
        topLayout->addWidget(widthGroup);

        // Группа для набора символов и пресетов
        QGroupBox* charsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout* charsetLayout = new QVBoxLayout;
        m_imgCharsetEdit = new QLineEdit(".,:;i1tfLCG08@");
        charsetLayout->addWidget(m_imgCharsetEdit);

        m_imgPresetCombo = new QComboBox;
        m_imgPresetCombo->addItem("Default:  .,:;i1tfLCG08@", ".,:;i1tfLCG08@");
        m_imgPresetCombo->addItem("Preset 1:  .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
                                   " .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_imgPresetCombo->addItem("Preset 2:  .:-=+*#%@", ".:-=+*#%@");
        m_imgPresetCombo->addItem("Preset 3: @%#*+=-:. ", "@%#*+=-:. ");
        charsetLayout->addWidget(m_imgPresetCombo);
        connect(m_imgPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int index){
                    QString preset = m_imgPresetCombo->itemData(index).toString();
                    m_imgCharsetEdit->setText(preset);
                });
        charsetGroup->setLayout(charsetLayout);
        topLayout->addWidget(charsetGroup);
		
		m_btnConvertImg = new QPushButton("Конвертировать");
		connect(m_btnConvertImg, &QPushButton::clicked, this, &AsciiArtApp::convertImageToAscii);
		topLayout->addWidget(m_btnConvertImg);

        layout->addLayout(topLayout);

        m_imgAsciiDisplay = new QTextEdit;
        m_imgAsciiDisplay->setReadOnly(true);
        m_imgAsciiDisplay->setFont(m_monospaceFont);
        layout->addWidget(m_imgAsciiDisplay);

        m_btnSaveImg = new QPushButton("Сохранить HTML");
        connect(m_btnSaveImg, &QPushButton::clicked, this, &AsciiArtApp::saveHtmlImage);
        layout->addWidget(m_btnSaveImg);
    }

    // --- Инициализация вкладки "Видео" ---
    void initVideoTab() {
        QVBoxLayout* layout = new QVBoxLayout;
        m_videoTab->setLayout(layout);

        QHBoxLayout* controlsLayout = new QHBoxLayout;
        m_btnOpenVideo = new QPushButton("Открыть видео");
        connect(m_btnOpenVideo, &QPushButton::clicked, this, &AsciiArtApp::openVideo);
        controlsLayout->addWidget(m_btnOpenVideo);

        QGroupBox* videoWidthGroup = new QGroupBox("Ширина");
        QFormLayout* videoWidthForm = new QFormLayout;
        m_videoSpinWidth = new QSpinBox;
        m_videoSpinWidth->setRange(10, 400);
        m_videoSpinWidth->setValue(78);
        videoWidthForm->addRow("Символов:", m_videoSpinWidth);
        videoWidthGroup->setLayout(videoWidthForm);
        controlsLayout->addWidget(videoWidthGroup);

        // Группа для набора символов и пресетов для видео
        QGroupBox* videoCharsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout* videoCharsetLayout = new QVBoxLayout;
        m_videoCharsetEdit = new QLineEdit("@%#*+=-:. ");
        videoCharsetLayout->addWidget(m_videoCharsetEdit);

        m_videoPresetCombo = new QComboBox;
        m_videoPresetCombo->addItem("Default: @%#*+=-:. ", "@%#*+=-:. ");
        m_videoPresetCombo->addItem("Preset 1:  .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
                                   " .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_videoPresetCombo->addItem("Preset 2:  .:-=+*#%@", ".:-=+*#%@");
        m_videoPresetCombo->addItem("Preset 3:  .,:;i1tfLCG08@", ".,:;i1tfLCG08@");
        videoCharsetLayout->addWidget(m_videoPresetCombo);
        connect(m_videoPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int index){
                    QString preset = m_videoPresetCombo->itemData(index).toString();
                    m_videoCharsetEdit->setText(preset);
                });
        videoCharsetGroup->setLayout(videoCharsetLayout);
        controlsLayout->addWidget(videoCharsetGroup);

        m_btnPreprocPlay = new QPushButton("Воспроизвести");
        connect(m_btnPreprocPlay, &QPushButton::clicked, this, &AsciiArtApp::startPreprocessing);
        controlsLayout->addWidget(m_btnPreprocPlay);

        m_btnStop = new QPushButton("Остановить");
        connect(m_btnStop, &QPushButton::clicked, this, &AsciiArtApp::stopVideo);
        m_btnStop->setEnabled(false);
        controlsLayout->addWidget(m_btnStop);


        layout->addLayout(controlsLayout);

        m_progressVideo = new QProgressBar;
        m_progressVideo->setValue(0);
        layout->addWidget(m_progressVideo);

        m_videoAsciiDisplay = new QTextEdit;
        m_videoAsciiDisplay->setReadOnly(true);
        m_videoAsciiDisplay->setFont(m_monospaceFont);
        layout->addWidget(m_videoAsciiDisplay);
    }

    // --- Инициализация вкладки "GIF" ---
    void initGifTab() {
        QVBoxLayout* layout = new QVBoxLayout;
        m_gifTab->setLayout(layout);

        QHBoxLayout* controlsLayout = new QHBoxLayout;
        m_btnOpenGif = new QPushButton("Открыть GIF");
        connect(m_btnOpenGif, &QPushButton::clicked, this, &AsciiArtApp::openGif);
        controlsLayout->addWidget(m_btnOpenGif);

        QGroupBox* gifWidthGroup = new QGroupBox("Ширина");
        QFormLayout* gifWidthForm = new QFormLayout;
        m_gifSpinWidth = new QSpinBox;
        m_gifSpinWidth->setRange(10, 400);
        m_gifSpinWidth->setValue(78);
        gifWidthForm->addRow("Символов:", m_gifSpinWidth);
        gifWidthGroup->setLayout(gifWidthForm);
        controlsLayout->addWidget(gifWidthGroup);

        // Группа для набора символов и пресетов для GIF
        QGroupBox* gifCharsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout* gifCharsetLayout = new QVBoxLayout;
        m_gifCharsetEdit = new QLineEdit("@%#*+=-:. ");
        gifCharsetLayout->addWidget(m_gifCharsetEdit);

        m_gifPresetCombo = new QComboBox;
        m_gifPresetCombo->addItem("Default: @%#*+=-:. ", "@%#*+=-:. ");
        m_gifPresetCombo->addItem("Preset 1:  .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
                                   " .'`^\\\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_gifPresetCombo->addItem("Preset 2:  .:-=+*#%@", ".:-=+*#%@");
        m_gifPresetCombo->addItem("Preset 3:  .,:;i1tfLCG08@", ".,:;i1tfLCG08@");
        gifCharsetLayout->addWidget(m_gifPresetCombo);
        connect(m_gifPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int index){
                    QString preset = m_gifPresetCombo->itemData(index).toString();
                    m_gifCharsetEdit->setText(preset);
                });
        gifCharsetGroup->setLayout(gifCharsetLayout);
        controlsLayout->addWidget(gifCharsetGroup);

        m_btnPreprocGif = new QPushButton("Конвертировать");
        connect(m_btnPreprocGif, &QPushButton::clicked, this, &AsciiArtApp::startPreprocessingGif);
        controlsLayout->addWidget(m_btnPreprocGif);

        m_btnStopGif = new QPushButton("Остановить");
        connect(m_btnStopGif, &QPushButton::clicked, this, &AsciiArtApp::stopGif);
        m_btnStopGif->setEnabled(false);
        controlsLayout->addWidget(m_btnStopGif);

        layout->addLayout(controlsLayout);

        m_progressGif = new QProgressBar;
        m_progressGif->setValue(0);
        layout->addWidget(m_progressGif);

        m_gifAsciiDisplay = new QTextEdit;
        m_gifAsciiDisplay->setReadOnly(true);
        m_gifAsciiDisplay->setFont(m_monospaceFont);
        layout->addWidget(m_gifAsciiDisplay);
    }

    // --- Виджеты для вкладки "Изображение" ---
    QWidget* m_imageTab;
    QPushButton* m_btnOpenImg;
    QSpinBox* m_imgSpinWidth;
    QLineEdit* m_imgCharsetEdit;
    QComboBox* m_imgPresetCombo;
    QPushButton* m_btnConvertImg;
    QTextEdit* m_imgAsciiDisplay;
    QPushButton* m_btnSaveImg;
    QString m_currentImagePath;

    // --- Виджеты для вкладки "Видео" ---
    QWidget* m_videoTab;
    QPushButton* m_btnOpenVideo;
    QSpinBox* m_videoSpinWidth;
    QLineEdit* m_videoCharsetEdit;
    QComboBox* m_videoPresetCombo;
    QPushButton* m_btnPreprocPlay;
    QPushButton* m_btnStop;
    QPushButton* m_btnSaveVideo;
    QProgressBar* m_progressVideo;
    QTextEdit* m_videoAsciiDisplay;
    QString m_currentVideoPath;

    // --- Виджеты для вкладки "GIF" ---
    QWidget* m_gifTab;
    QPushButton* m_btnOpenGif;
    QSpinBox* m_gifSpinWidth;
    QLineEdit* m_gifCharsetEdit;
    QComboBox* m_gifPresetCombo;
    QPushButton* m_btnPreprocGif;
    QPushButton* m_btnStopGif;
    QPushButton* m_btnSaveGif;
    QProgressBar* m_progressGif;
    QTextEdit* m_gifAsciiDisplay;
    QString m_currentGifPath;

    QTabWidget* m_tabWidget;
    QFont m_monospaceFont;

    // --- Переменные для обработки видео ---
    PreprocessingThread* m_preprocThread;
    QVector<QString> m_asciiFrames;
    double m_videoFps;
    int m_videoLength;
    qint64 m_videoStartTime;
    int m_currentFrameIndex;
    QTimer* m_playTimer;
    QMediaPlayer* m_player;

    // --- Переменные для обработки GIF ---
    PreprocessingThread* m_gifPreprocThread;
    QVector<QString> m_gifAsciiFrames;
    double m_gifFps;
    int m_gifLength;
    qint64 m_gifStartTime;
    int m_currentGifFrameIndex;
    QTimer* m_gifPlayTimer;
};

#include "main.moc"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    AsciiArtApp window;
    window.show();
    return app.exec();
}
