// main.cpp

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QWidget>
#include <QPushButton>
#include <QFileDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QTextEdit>
#include <QLineEdit>
#include <QFormLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QProgressBar>
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QComboBox>
#include <QTimer>
#include <QDateTime>
#include <QUrl>
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QThread>
#include <QPainter>
#include <QTextDocument>
#include <QFontMetrics>
#include <QProcess>
#include <QTemporaryFile>
#include <QDebug>
#include <QPalette>
#include <QFont>
#include <QDir>
#include <QFile>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Класс для предобработки видео или GIF в ASCII-арт (аналог PreprocessingThread в Python)
class PreprocessingThread : public QThread {
    Q_OBJECT
public:
    PreprocessingThread(const QString &videoPath, int desiredWidth, const QString &asciiChars, bool blackWhite, QObject *parent = nullptr)
        : QThread(parent), m_videoPath(videoPath), m_desiredWidth(desiredWidth), m_asciiChars(asciiChars),
          m_blackWhite(blackWhite), m_runFlag(true) {}

    void stop() { m_runFlag = false; }

signals:
    void finished(const std::vector<QString>& frames, double fps);
    void progress(int processed, int total);

protected:
    void run() override {
        cv::VideoCapture cap(m_videoPath.toStdString());
        if (!cap.isOpened()) {
            emit finished({}, 0.0);
            return;
        }
        double realFps = cap.get(cv::CAP_PROP_FPS);
        if (realFps <= 0) realFps = 24.0;
        int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        if (totalFrames < 1)
            totalFrames = 0;

        std::vector<QString> asciiFrames;
        int asciiLen = m_asciiChars.length();
        int processedCount = 0;
        cv::Mat frame;
        while (m_runFlag && cap.read(frame)) {
            int h = frame.rows;
            int w = frame.cols;
            double aspectRatio = static_cast<double>(h) / w;
            int newH = std::max(1, static_cast<int>(m_desiredWidth * aspectRatio * 0.55));
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(m_desiredWidth, newH));
            QStringList lines;
            for (int row = 0; row < newH; ++row) {
                QString rowStr;
                for (int col = 0; col < m_desiredWidth; ++col) {
                    cv::Vec3b pixel = resized.at<cv::Vec3b>(row, col);
                    int b = pixel[0], g = pixel[1], r = pixel[2];
                    double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    int idx = static_cast<int>(gray / 255 * (asciiLen - 1));
                    idx = std::max(0, std::min(idx, asciiLen - 1));
                    QChar ch = m_asciiChars[idx];
                    if (m_blackWhite)
                        rowStr.append(ch);
                    else
                        rowStr.append(QString("<span style=\"color: rgb(%1,%2,%3)\">%4</span>")
                                      .arg(r).arg(g).arg(b).arg(ch));
                }
                lines.append(rowStr);
            }
            QString frameText = m_blackWhite ? lines.join("\n") : lines.join("<br>");
            asciiFrames.push_back(frameText);
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
    bool m_blackWhite;
    bool m_runFlag;
};

// Главное окно приложения
class AsciiArtApp : public QMainWindow {
    Q_OBJECT
public:
    AsciiArtApp(QWidget *parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("Генератор ASCII-арта");
        resize(800, 750);
        QApplication::setStyle("Fusion");
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

        m_imageTab = new QWidget;
        m_videoTab = new QWidget;
        m_gifTab = new QWidget;

        m_tabWidget->addTab(m_imageTab, "Изображение в ASCII");
        m_tabWidget->addTab(m_videoTab, "Видео в ASCII");
        m_tabWidget->addTab(m_gifTab, "GIF в ASCII");

        // Кнопка закрытия в углу
        QPushButton *btnClose = new QPushButton("Закрыть");
        btnClose->setStyleSheet("background-color: red; color: white; padding: 5px;");
        connect(btnClose, &QPushButton::clicked, this, &QWidget::close);
        m_tabWidget->setCornerWidget(btnClose, Qt::TopRightCorner);

        initImageTab();
        initVideoTab();
        initGifTab();

        m_player = new QMediaPlayer(this);
        m_audioOutput = new QAudioOutput(this);
        m_player->setAudioOutput(m_audioOutput);

        m_playTimer = new QTimer(this);
        m_playTimer->setInterval(15);
        connect(m_playTimer, &QTimer::timeout, this, &AsciiArtApp::showNextFrame);

        m_gifPlayTimer = new QTimer(this);
        m_gifPlayTimer->setInterval(15);
        connect(m_gifPlayTimer, &QTimer::timeout, this, &AsciiArtApp::showNextGifFrame);

        m_preprocThread = nullptr;
        m_videoFps = 24.0;
        m_currentFrameIndex = 0;

        m_gifPreprocThread = nullptr;
        m_gifFps = 24.0;
        m_currentGifFrameIndex = 0;

        m_imgBlackWhite = false;
        m_videoBlackWhite = false;
        m_gifBlackWhite = false;
    }

    ~AsciiArtApp() {
        if(m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->wait();
            delete m_preprocThread;
        }
        if(m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->wait();
            delete m_gifPreprocThread;
        }
        if(m_player)
            m_player->stop();
    }

protected:
    void closeEvent(QCloseEvent *event) override {
        m_playTimer->stop();
        m_gifPlayTimer->stop();
        if(m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->wait();
        }
        if(m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->wait();
        }
        if(m_player)
            m_player->stop();
        QMainWindow::closeEvent(event);
    }

private slots:
    // Слоты для обработки вкладки "Изображение в ASCII"
    void updateImgCharset(int index) {
        QString preset = m_imgPresetCombo->itemData(index).toString();
        m_imgCharsetEdit->setText(preset);
    }

    void openImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp *.gif)");
        if(!fileName.isEmpty()){
            m_currentImagePath = fileName;
            convertImageToAscii();
        }
    }

    void convertImageToAscii() {
        if(m_currentImagePath.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Сначала выберите изображение.");
            return;
        }
        cv::Mat img = cv::imread(m_currentImagePath.toStdString());
        if(img.empty()){
            QMessageBox::warning(this, "Ошибка", "Не удалось открыть изображение.");
            return;
        }
        int dw = m_imgSpinWidth->value();
        QString asciiChars = m_imgCharsetEdit->text();
        if(asciiChars.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }
        int h = img.rows, w = img.cols;
        double aspect = static_cast<double>(h) / w;
        int newH = std::max(1, static_cast<int>(dw * aspect * 0.55));
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(dw, newH));
        QStringList lines;
        int length = asciiChars.length();
        m_progressImage->setValue(0);
        for (int row = 0; row < newH; ++row) {
            QString rowStr;
            for (int col = 0; col < dw; ++col) {
                cv::Vec3b pixel = resized.at<cv::Vec3b>(row, col);
                int b = pixel[0], g = pixel[1], r = pixel[2];
                double gray = 0.299*r + 0.587*g + 0.114*b;
                int idx = static_cast<int>(gray / 255 * (length - 1));
                idx = std::max(0, std::min(idx, length - 1));
                QChar ch = asciiChars[idx];
                if(m_imgBlackWhite)
                    rowStr.append(ch);
                else
                    rowStr.append(QString("<span style=\"color: rgb(%1,%2,%3)\">%4</span>")
                                  .arg(r).arg(g).arg(b).arg(ch));
            }
            lines.append(rowStr);
            int progress = (row + 1) * 100 / newH;
            m_progressImage->setValue(progress);
            qApp->processEvents();
        }
        if(m_imgBlackWhite) {
            m_imgAsciiDisplay->setStyleSheet("background-color: black; color: white;");
            m_imgAsciiDisplay->setPlainText(lines.join("\n"));
        } else {
            m_imgAsciiDisplay->setStyleSheet("background-color: black;");
            m_imgAsciiDisplay->setHtml(lines.join("<br>"));
        }
    }

    void saveHtmlImage() {
        QString htmlCode = m_imgAsciiDisplay->toHtml();
        if(htmlCode.trimmed().isEmpty()){
            QMessageBox::information(this, "Пусто", "Нет ASCII-арта.");
            return;
        }
        QString fileName = QFileDialog::getSaveFileName(this, "Сохранить HTML", "", "HTML файлы (*.html)");
        if(!fileName.isEmpty()){
            QFile file(fileName);
            if(file.open(QIODevice::WriteOnly | QIODevice::Text)){
                QTextStream out(&file);
                out << htmlCode;
                file.close();
                QMessageBox::information(this, "Успех", QString("Сохранено:\n%1").arg(fileName));
            }
        }
    }

    void saveImageAsPicture() {
        QString asciiText = m_imgAsciiDisplay->toPlainText();
        if(asciiText.trimmed().isEmpty()){
            QMessageBox::information(this, "Пусто", "Нет ASCII-арта для сохранения.");
            return;
        }
        QFont font = m_imgAsciiDisplay->font();
        QFontMetrics fm(font);
        QStringList lines = asciiText.split("\n");
        int lineCount = lines.size();
        int maxWidth = 0;
        for(const QString &line : lines)
            maxWidth = std::max(maxWidth, fm.horizontalAdvance(line));
        int lineHeight = fm.height();
        int imgWidth = maxWidth;
        int imgHeight = lineHeight * lineCount;
        if(imgWidth <= 0 || imgHeight <= 0){
            QMessageBox::critical(this, "Ошибка", QString("Некорректные размеры: %1x%2").arg(imgWidth).arg(imgHeight));
            return;
        }
        QImage image(imgWidth, imgHeight, QImage::Format_ARGB32);
        image.fill(Qt::black);
        QPainter painter(&image);
        painter.setFont(font);
        if(m_imgBlackWhite){
            painter.setPen(Qt::white);
            int y = fm.ascent();
            for(const QString &line : lines){
                painter.drawText(0, y, line);
                y += fm.height();
            }
        } else {
            QTextDocument doc;
            doc.setHtml(m_imgAsciiDisplay->toHtml());
            doc.setDefaultFont(font);
            doc.drawContents(&painter);
        }
        painter.end();
        QString fileName = QFileDialog::getSaveFileName(this, "Сохранить как изображение", "", "Изображения (*.png *.jpg *.bmp)");
        if(!fileName.isEmpty()){
            if(!(fileName.endsWith(".png") || fileName.endsWith(".jpg") || fileName.endsWith(".bmp")))
                fileName += ".png";
            if(image.save(fileName))
                QMessageBox::information(this, "Успех", QString("Изображение сохранено:\n%1").arg(fileName));
            else
                QMessageBox::critical(this, "Ошибка", QString("Не удалось сохранить изображение:\n%1").arg(fileName));
        }
    }

    void onImgZoomChanged(int value) {
        QFont font = m_imgAsciiDisplay->font();
        font.setPointSize(value);
        m_imgAsciiDisplay->setFont(font);
    }

    // Слоты для вкладки "Видео в ASCII"
    void updateVideoCharset(int index) {
        QString preset = m_videoPresetCombo->itemData(index).toString();
        m_videoCharsetEdit->setText(preset);
    }

    void openVideo() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите видео", "", "Видео файлы (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)");
        if(!fileName.isEmpty()){
            m_currentVideoPath = fileName;
            QMessageBox::information(this, "Выбрано видео", m_currentVideoPath);
        }
    }

    void startPreprocessing() {
        if(m_currentVideoPath.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Сначала выберите видеофайл.");
            return;
        }
        int w = m_videoSpinWidth->value();
        QString chars = m_videoCharsetEdit->text();
        if(chars.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }
        m_videoAsciiDisplay->clear();
        m_btnPreprocPlay->setEnabled(false);
        m_btnStop->setEnabled(true);
        m_progressVideo->setValue(0);
        if(m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->wait();
            delete m_preprocThread;
        }
        m_preprocThread = new PreprocessingThread(m_currentVideoPath, w, chars, m_videoBlackWhite, this);
        connect(m_preprocThread, &PreprocessingThread::finished, this, &AsciiArtApp::onPreprocessingFinished);
        connect(m_preprocThread, &PreprocessingThread::progress, this, &AsciiArtApp::onPreprocessingProgress);
        m_preprocThread->start();
    }

    void onPreprocessingFinished(const std::vector<QString> &frames, double fps) {
        m_asciiFrames = frames;
        m_videoFps = fps;
        m_videoLength = frames.size();
        if(m_preprocThread) {
            m_preprocThread->wait();
            delete m_preprocThread;
            m_preprocThread = nullptr;
        }
        if(m_asciiFrames.empty() || m_videoLength == 0){
            QMessageBox::warning(this, "Ошибка", "Не удалось получить кадры из видео.");
            m_btnPreprocPlay->setEnabled(true);
            m_btnStop->setEnabled(false);
            return;
        }
        m_progressVideo->setValue(100);
        m_videoStartTime = QDateTime::currentMSecsSinceEpoch();
        m_currentFrameIndex = 0;
        m_videoAsciiDisplay->clear();
        m_playTimer->start();
        m_player->setSource(QUrl::fromLocalFile(m_currentVideoPath));
        m_player->play();
    }

    void onPreprocessingProgress(int processed, int total) {
        if(total > 0) {
            int percentage = processed * 100 / total;
            m_progressVideo->setValue(percentage);
        } else {
            m_progressVideo->setValue(0);
        }
    }

    void showNextFrame() {
        qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
        qint64 elapsed = currentTime - m_videoStartTime;
        int frameIndex = static_cast<int>(elapsed / 1000.0 * m_videoFps);
        if(frameIndex >= static_cast<int>(m_videoLength)) {
            stopVideo();
            return;
        }
        if(frameIndex != m_currentFrameIndex) {
            if(m_videoBlackWhite){
                m_videoAsciiDisplay->setStyleSheet("background-color: black; color: white;");
                m_videoAsciiDisplay->setPlainText(m_asciiFrames[frameIndex]);
            } else {
                m_videoAsciiDisplay->setStyleSheet("background-color: black;");
                m_videoAsciiDisplay->setHtml(m_asciiFrames[frameIndex]);
            }
            m_currentFrameIndex = frameIndex;
        }
    }

    void stopVideo() {
        m_playTimer->stop();
        if(m_player)
            m_player->stop();
        m_btnPreprocPlay->setEnabled(true);
        m_btnStop->setEnabled(false);
        if(m_preprocThread) {
            m_preprocThread->stop();
            m_preprocThread->wait();
            delete m_preprocThread;
            m_preprocThread = nullptr;
        }
    }

    void onVideoZoomChanged(int value) {
        QFont font = m_videoAsciiDisplay->font();
        font.setPointSize(value);
        m_videoAsciiDisplay->setFont(font);
    }

    void saveVideoWithAudio() {
        try {
            if(m_asciiFrames.empty()){
                QMessageBox::warning(this, "Ошибка", "Нет обработанных кадров для сохранения видео.");
                return;
            }
            QString fileName = QFileDialog::getSaveFileName(this, "Сохранить видео", "", "Видео файлы (*.mp4)");
            if(fileName.isEmpty())
                return;
            if(!fileName.endsWith(".mp4", Qt::CaseInsensitive))
                fileName += ".mp4";

            QFont font = m_videoAsciiDisplay->font();
            QFontMetrics fm(font);
            int charWidth = fm.averageCharWidth();
            int expectedWidth = m_videoSpinWidth->value();
            int width = charWidth * expectedWidth;
            QString firstFrame = m_asciiFrames[0];
            QStringList lines = m_videoBlackWhite ? firstFrame.split("\n") : firstFrame.split("<br>");
            int height = fm.height() * lines.size();

            if(width <= 0 || height <= 0){
                QMessageBox::critical(this, "Ошибка", QString("Некорректные размеры видео: %1x%2").arg(width).arg(height));
                return;
            }

            QString tempVideo = QDir::tempPath() + "/temp_video.avi";
            int fourcc = cv::VideoWriter::fourcc('X','V','I','D');
            cv::VideoWriter out(tempVideo.toStdString(), fourcc, m_videoFps, cv::Size(width, height));
            if(!out.isOpened()){
                QMessageBox::critical(this, "Ошибка", "Не удалось инициализировать VideoWriter.");
                return;
            }
            for (const auto &frameText : m_asciiFrames) {
                QImage image(width, height, QImage::Format_ARGB32);
                image.fill(Qt::black);
                QPainter painter(&image);
                painter.setFont(font);
                if(m_videoBlackWhite){
                    painter.setPen(Qt::white);
                    int y = fm.ascent();
                    QStringList textLines = frameText.split("\n");
                    for(const QString &line : textLines) {
                        painter.drawText(0, y, line.left(expectedWidth));
                        y += fm.height();
                    }
                } else {
                    QTextDocument doc;
                    doc.setHtml(frameText);
                    doc.setDefaultFont(font);
                    doc.setTextWidth(width);
                    doc.drawContents(&painter);
                }
                painter.end();
                QImage rgbImage = image.convertToFormat(QImage::Format_RGB32);
                // Преобразование QImage в cv::Mat
                cv::Mat mat(rgbImage.height(), rgbImage.width(), CV_8UC4, const_cast<uchar*>(rgbImage.bits()), rgbImage.bytesPerLine());
                cv::Mat frameBGR;
                cv::cvtColor(mat, frameBGR, cv::COLOR_BGRA2BGR);
                out.write(frameBGR);
            }
            out.release();

            // Используем QProcess для вызова ffmpeg
            QString ffmpegPath = "ffmpeg";
            QProcess process;
            process.start(ffmpegPath, QStringList() << "-version");
            process.waitForFinished();
            if(process.exitCode() != 0) {
#ifdef Q_OS_WIN
                QString exeDir = QCoreApplication::applicationDirPath();
                ffmpegPath = exeDir + "/ffmpeg/bin/ffmpeg.exe";
#else
                QString exeDir = QCoreApplication::applicationDirPath();
                ffmpegPath = exeDir + "/ffmpeg/bin/ffmpeg";
#endif
                if(!QFile::exists(ffmpegPath)) {
                    QMessageBox::critical(this, "Ошибка", QString("FFmpeg не найден.\n- Системный FFmpeg отсутствует в PATH.\n- Локальный FFmpeg не найден по пути: %1").arg(ffmpegPath));
                    return;
                }
            }
            QString audioFile = QDir::tempPath() + "/temp_audio.mp3";
            QStringList audioCmd;
            audioCmd << "-y" << "-i" << m_currentVideoPath << "-vn" << "-acodec" << "mp3" << audioFile;
            process.start(ffmpegPath, audioCmd);
            process.waitForFinished();
            if(process.exitCode() != 0) {
                QString errorMsg = process.readAllStandardError();
                QMessageBox::critical(this, "Ошибка", QString("Не удалось извлечь аудио:\n%1").arg(errorMsg));
                return;
            }
            QStringList mergeCmd;
            mergeCmd << "-y" << "-i" << tempVideo << "-i" << audioFile << "-c:v" << "libx264" << "-c:a" << "aac" << "-shortest" << fileName;
            process.start(ffmpegPath, mergeCmd);
            process.waitForFinished();
            if(process.exitCode() != 0) {
                QString errorMsg = process.readAllStandardError();
                QMessageBox::critical(this, "Ошибка", QString("Не удалось объединить видео и аудио:\n%1").arg(errorMsg));
                return;
            }
            QFile::remove(tempVideo);
            QFile::remove(audioFile);
            QMessageBox::information(this, "Успех", QString("Видео сохранено:\n%1").arg(fileName));
        } catch (...) {
            QMessageBox::critical(this, "Ошибка", "Произошла ошибка при сохранении видео.");
        }
    }

    // Слоты для вкладки "GIF в ASCII"
    void updateGifCharset(int index) {
        QString preset = m_gifPresetCombo->itemData(index).toString();
        m_gifCharsetEdit->setText(preset);
    }

    void openGif() {
        QString fileName = QFileDialog::getOpenFileName(this, "Выберите GIF", "", "GIF файлы (*.gif)");
        if(!fileName.isEmpty()){
            m_currentGifPath = fileName;
            QMessageBox::information(this, "Выбран GIF", m_currentGifPath);
        }
    }

    void startPreprocessingGif() {
        if(m_currentGifPath.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Сначала выберите GIF-файл.");
            return;
        }
        int w = m_gifSpinWidth->value();
        QString chars = m_gifCharsetEdit->text();
        if(chars.isEmpty()){
            QMessageBox::warning(this, "Ошибка", "Набор символов пуст.");
            return;
        }
        m_gifAsciiDisplay->clear();
        m_btnPreprocGif->setEnabled(false);
        m_btnStopGif->setEnabled(true);
        m_progressGif->setValue(0);
        if(m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->wait();
            delete m_gifPreprocThread;
        }
        m_gifPreprocThread = new PreprocessingThread(m_currentGifPath, w, chars, m_gifBlackWhite, this);
        connect(m_gifPreprocThread, &PreprocessingThread::finished, this, &AsciiArtApp::onGifPreprocessingFinished);
        connect(m_gifPreprocThread, &PreprocessingThread::progress, this, &AsciiArtApp::onGifPreprocessingProgress);
        m_gifPreprocThread->start();
    }

    void onGifPreprocessingFinished(const std::vector<QString> &frames, double fps) {
        m_gifAsciiFrames = frames;
        m_gifFps = fps;
        m_gifLength = frames.size();
        if(m_gifPreprocThread) {
            m_gifPreprocThread->wait();
            delete m_gifPreprocThread;
            m_gifPreprocThread = nullptr;
        }
        if(m_gifAsciiFrames.empty() || m_gifLength == 0){
            QMessageBox::warning(this, "Ошибка", "Не удалось получить кадры из GIF.");
            m_btnPreprocGif->setEnabled(true);
            m_btnStopGif->setEnabled(false);
            return;
        }
        m_progressGif->setValue(100);
        m_gifStartTime = QDateTime::currentMSecsSinceEpoch();
        m_currentGifFrameIndex = 0;
        m_gifAsciiDisplay->clear();
        m_gifPlayTimer->start();
    }

    void onGifPreprocessingProgress(int processed, int total) {
        if(total > 0) {
            int percentage = processed * 100 / total;
            m_progressGif->setValue(percentage);
        } else {
            m_progressGif->setValue(0);
        }
    }

    void showNextGifFrame() {
        if(m_gifLength == 0)
            return;
        qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
        qint64 elapsed = currentTime - m_gifStartTime;
        int frameIndex = static_cast<int>((elapsed / 1000.0 * m_gifFps)) % m_gifLength;
        if(frameIndex != m_currentGifFrameIndex) {
            if(m_gifBlackWhite){
                m_gifAsciiDisplay->setStyleSheet("background-color: black; color: white;");
                m_gifAsciiDisplay->setPlainText(m_gifAsciiFrames[frameIndex]);
            } else {
                m_gifAsciiDisplay->setStyleSheet("background-color: black;");
                m_gifAsciiDisplay->setHtml(m_gifAsciiFrames[frameIndex]);
            }
            m_currentGifFrameIndex = frameIndex;
        }
    }

    void stopGif() {
        m_gifPlayTimer->stop();
        m_btnPreprocGif->setEnabled(true);
        m_btnStopGif->setEnabled(false);
        if(m_gifPreprocThread) {
            m_gifPreprocThread->stop();
            m_gifPreprocThread->wait();
            delete m_gifPreprocThread;
            m_gifPreprocThread = nullptr;
        }
    }

    void onGifZoomChanged(int value) {
        QFont font = m_gifAsciiDisplay->font();
        font.setPointSize(value);
        m_gifAsciiDisplay->setFont(font);
    }

    void saveGif() {
        try {
            if(m_gifAsciiFrames.empty()){
                QMessageBox::warning(this, "Ошибка", "Нет обработанных кадров для сохранения GIF.");
                return;
            }
            QString fileName = QFileDialog::getSaveFileName(this, "Сохранить GIF", "", "GIF файлы (*.gif)");
            if(fileName.isEmpty())
                return;
            if(!fileName.endsWith(".gif", Qt::CaseInsensitive))
                fileName += ".gif";

            QFont font = m_gifAsciiDisplay->font();
            QFontMetrics fm(font);
            int expectedWidth = m_gifSpinWidth->value();
            QString firstFrame = m_gifAsciiFrames[0];
            QStringList lines = m_gifBlackWhite ? firstFrame.split("\n") : firstFrame.split("<br>");
            int lineCount = lines.size();
            int charWidth = fm.averageCharWidth();
            int width = charWidth * expectedWidth;
            int height = fm.height() * lineCount;

            QString tempVideo = QDir::tempPath() + "/temp_gif_video.mp4";
            int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
            cv::VideoWriter out(tempVideo.toStdString(), fourcc, m_gifFps, cv::Size(width, height));
            for (const auto &frameText : m_gifAsciiFrames) {
                QImage image(width, height, QImage::Format_ARGB32);
                image.fill(Qt::black);
                QPainter painter(&image);
                painter.setFont(font);
                if(m_gifBlackWhite) {
                    painter.setPen(Qt::white);
                    int y = fm.ascent();
                    QStringList textLines = frameText.split("\n");
                    for(const QString &line : textLines) {
                        painter.drawText(0, y, line.left(expectedWidth));
                        y += fm.height();
                    }
                } else {
                    QTextDocument doc;
                    doc.setHtml(frameText);
                    doc.setDefaultFont(font);
                    doc.setTextWidth(width);
                    doc.drawContents(&painter);
                }
                painter.end();
                QImage rgbImage = image.convertToFormat(QImage::Format_RGB32);
                cv::Mat mat(rgbImage.height(), rgbImage.width(), CV_8UC4, const_cast<uchar*>(rgbImage.bits()), rgbImage.bytesPerLine());
                cv::Mat frameBGR;
                cv::cvtColor(mat, frameBGR, cv::COLOR_BGRA2BGR);
                out.write(frameBGR);
            }
            out.release();

            QString ffmpegPath = "ffmpeg";
            QProcess process;
            process.start(ffmpegPath, QStringList() << "-version");
            process.waitForFinished();
            if(process.exitCode() != 0) {
#ifdef Q_OS_WIN
                QString exeDir = QCoreApplication::applicationDirPath();
                ffmpegPath = exeDir + "/ffmpeg/bin/ffmpeg.exe";
#else
                QString exeDir = QCoreApplication::applicationDirPath();
                ffmpegPath = exeDir + "/ffmpeg/bin/ffmpeg";
#endif
                if(!QFile::exists(ffmpegPath)) {
                    QMessageBox::critical(this, "Ошибка", QString("FFmpeg не найден.\n- Системный FFmpeg отсутствует в PATH.\n- Локальный FFmpeg не найден по пути: %1").arg(ffmpegPath));
                    return;
                }
            }
            QStringList cmd;
            cmd << "-y" << "-i" << tempVideo << "-vf" << QString("fps=%1,scale=%2:-1:flags=lanczos").arg(m_gifFps).arg(width) << fileName;
            process.start(ffmpegPath, cmd);
            process.waitForFinished();
            QFile::remove(tempVideo);
            if(process.exitCode() == 0)
                QMessageBox::information(this, "Успех", QString("GIF сохранён:\n%1").arg(fileName));
            else
                QMessageBox::critical(this, "Ошибка", QString("Не удалось сохранить GIF.\n%1").arg(process.readAllStandardError()));
        } catch (...) {
            QMessageBox::critical(this, "Ошибка", "Произошла ошибка при сохранении GIF.");
        }
    }

    // Инициализация пользовательского интерфейса
    void initImageTab() {
        QVBoxLayout *layout = new QVBoxLayout;
        m_imageTab->setLayout(layout);

        QHBoxLayout *topLayout = new QHBoxLayout;
        QPushButton *btnOpenImg = new QPushButton("Открыть изображение");
        connect(btnOpenImg, &QPushButton::clicked, this, &AsciiArtApp::openImage);
        topLayout->addWidget(btnOpenImg);

        QGroupBox *widthGroup = new QGroupBox("Ширина");
        QFormLayout *widthForm = new QFormLayout;
        m_imgSpinWidth = new QSpinBox;
        m_imgSpinWidth->setRange(10, 800);
        m_imgSpinWidth->setValue(80);
        widthForm->addRow("Символов:", m_imgSpinWidth);
        widthGroup->setLayout(widthForm);
        topLayout->addWidget(widthGroup);

        QGroupBox *charsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout *charsetLayout = new QVBoxLayout;
        m_imgCharsetEdit = new QLineEdit(".,:;i1tfLCG08@");
        charsetLayout->addWidget(m_imgCharsetEdit);
        m_imgPresetCombo = new QComboBox;
        m_imgPresetCombo->addItem("Default: .,:;i1tfLCG08@", ".,:;i1tfLCG08@");
        m_imgPresetCombo->addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_imgPresetCombo->addItem("Preset 2", ".:-=+*#%@");
        m_imgPresetCombo->addItem("Preset 3", "@%#*+=-:. ");
        connect(m_imgPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AsciiArtApp::updateImgCharset);
        charsetLayout->addWidget(m_imgPresetCombo);
        charsetGroup->setLayout(charsetLayout);
        topLayout->addWidget(charsetGroup);

        QCheckBox *imgBwCheckbox = new QCheckBox("Черно-белый режим");
        connect(imgBwCheckbox, &QCheckBox::toggled, [this](bool checked){ m_imgBlackWhite = checked; });
        topLayout->addWidget(imgBwCheckbox);

        QPushButton *btnConvertImg = new QPushButton("Конвертировать");
        connect(btnConvertImg, &QPushButton::clicked, this, &AsciiArtApp::convertImageToAscii);
        topLayout->addWidget(btnConvertImg);

        layout->addLayout(topLayout);

        QHBoxLayout *zoomLayout = new QHBoxLayout;
        QLabel *zoomLabel = new QLabel("Масштаб:");
        m_imgZoomSlider = new QSlider(Qt::Horizontal);
        m_imgZoomSlider->setRange(5, 30);
        m_imgZoomSlider->setValue(m_monospaceFont.pointSize());
        connect(m_imgZoomSlider, &QSlider::valueChanged, this, &AsciiArtApp::onImgZoomChanged);
        zoomLayout->addWidget(zoomLabel);
        zoomLayout->addWidget(m_imgZoomSlider);
        layout->addLayout(zoomLayout);

        m_imgAsciiDisplay = new QTextEdit;
        m_imgAsciiDisplay->setReadOnly(true);
        m_imgAsciiDisplay->setFont(m_monospaceFont);
        m_imgAsciiDisplay->setLineWrapMode(QTextEdit::NoWrap);
        m_imgAsciiDisplay->setStyleSheet("background-color: black;");
        layout->addWidget(m_imgAsciiDisplay);

        m_progressImage = new QProgressBar;
        m_progressImage->setValue(0);
        m_progressImage->setFixedHeight(10);
        layout->addWidget(m_progressImage);

        QPushButton *btnSaveImg = new QPushButton("Сохранить HTML");
        connect(btnSaveImg, &QPushButton::clicked, this, &AsciiArtApp::saveHtmlImage);
        layout->addWidget(btnSaveImg);

        QPushButton *btnSaveImgAsPic = new QPushButton("Сохранить как изображение");
        connect(btnSaveImgAsPic, &QPushButton::clicked, this, &AsciiArtApp::saveImageAsPicture);
        layout->addWidget(btnSaveImgAsPic);
    }

    void initVideoTab() {
        QVBoxLayout *layout = new QVBoxLayout;
        m_videoTab->setLayout(layout);

        QHBoxLayout *controlsLayout = new QHBoxLayout;
        QPushButton *btnOpenVideo = new QPushButton("Открыть видео");
        connect(btnOpenVideo, &QPushButton::clicked, this, &AsciiArtApp::openVideo);
        controlsLayout->addWidget(btnOpenVideo);

        QGroupBox *videoWidthGroup = new QGroupBox("Ширина");
        QFormLayout *videoWidthForm = new QFormLayout;
        m_videoSpinWidth = new QSpinBox;
        m_videoSpinWidth->setRange(10, 400);
        m_videoSpinWidth->setValue(78);
        videoWidthForm->addRow("Символов:", m_videoSpinWidth);
        videoWidthGroup->setLayout(videoWidthForm);
        controlsLayout->addWidget(videoWidthGroup);

        QGroupBox *videoCharsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout *videoCharsetLayout = new QVBoxLayout;
        m_videoCharsetEdit = new QLineEdit("@%#*+=-:. ");
        videoCharsetLayout->addWidget(m_videoCharsetEdit);
        m_videoPresetCombo = new QComboBox;
        m_videoPresetCombo->addItem("Default: @%#*+=-:. ", "@%#*+=-:. ");
        m_videoPresetCombo->addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_videoPresetCombo->addItem("Preset 2", ".:-=+*#%@");
        m_videoPresetCombo->addItem("Preset 3", ".,:;i1tfLCG08@");
        connect(m_videoPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AsciiArtApp::updateVideoCharset);
        videoCharsetLayout->addWidget(m_videoPresetCombo);
        videoCharsetGroup->setLayout(videoCharsetLayout);
        controlsLayout->addWidget(videoCharsetGroup);

        QCheckBox *videoBwCheckbox = new QCheckBox("Черно-белый режим");
        connect(videoBwCheckbox, &QCheckBox::toggled, [this](bool checked){ m_videoBlackWhite = checked; });
        controlsLayout->addWidget(videoBwCheckbox);

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

        QHBoxLayout *videoZoomLayout = new QHBoxLayout;
        QLabel *videoZoomLabel = new QLabel("Масштаб:");
        m_videoZoomSlider = new QSlider(Qt::Horizontal);
        m_videoZoomSlider->setRange(5, 30);
        m_videoZoomSlider->setValue(m_monospaceFont.pointSize());
        connect(m_videoZoomSlider, &QSlider::valueChanged, this, &AsciiArtApp::onVideoZoomChanged);
        videoZoomLayout->addWidget(videoZoomLabel);
        videoZoomLayout->addWidget(m_videoZoomSlider);
        layout->addLayout(videoZoomLayout);

        m_videoAsciiDisplay = new QTextEdit;
        m_videoAsciiDisplay->setReadOnly(true);
        m_videoAsciiDisplay->setFont(m_monospaceFont);
        m_videoAsciiDisplay->setLineWrapMode(QTextEdit::NoWrap);
        m_videoAsciiDisplay->setStyleSheet("background-color: black;");
        layout->addWidget(m_videoAsciiDisplay);

        QPushButton *btnSaveVideo = new QPushButton("Сохранить видео");
        connect(btnSaveVideo, &QPushButton::clicked, this, &AsciiArtApp::saveVideoWithAudio);
        layout->addWidget(btnSaveVideo);
    }

    void initGifTab() {
        QVBoxLayout *layout = new QVBoxLayout;
        m_gifTab->setLayout(layout);

        QHBoxLayout *controlsLayout = new QHBoxLayout;
        QPushButton *btnOpenGif = new QPushButton("Открыть GIF");
        connect(btnOpenGif, &QPushButton::clicked, this, &AsciiArtApp::openGif);
        controlsLayout->addWidget(btnOpenGif);

        QGroupBox *gifWidthGroup = new QGroupBox("Ширина");
        QFormLayout *gifWidthForm = new QFormLayout;
        m_gifSpinWidth = new QSpinBox;
        m_gifSpinWidth->setRange(10, 400);
        m_gifSpinWidth->setValue(78);
        gifWidthForm->addRow("Символов:", m_gifSpinWidth);
        gifWidthGroup->setLayout(gifWidthForm);
        controlsLayout->addWidget(gifWidthGroup);

        QGroupBox *gifCharsetGroup = new QGroupBox("Набор символов");
        QVBoxLayout *gifCharsetLayout = new QVBoxLayout;
        m_gifCharsetEdit = new QLineEdit("@%#*+=-:. ");
        gifCharsetLayout->addWidget(m_gifCharsetEdit);
        m_gifPresetCombo = new QComboBox;
        m_gifPresetCombo->addItem("Default: @%#*+=-:. ", "@%#*+=-:. ");
        m_gifPresetCombo->addItem("Preset 1", " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$");
        m_gifPresetCombo->addItem("Preset 2", ".:-=+*#%@");
        m_gifPresetCombo->addItem("Preset 3", ".,:;i1tfLCG08@");
        connect(m_gifPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AsciiArtApp::updateGifCharset);
        gifCharsetLayout->addWidget(m_gifPresetCombo);
        gifCharsetGroup->setLayout(gifCharsetLayout);
        controlsLayout->addWidget(gifCharsetGroup);

        QCheckBox *gifBwCheckbox = new QCheckBox("Черно-белый режим");
        connect(gifBwCheckbox, &QCheckBox::toggled, [this](bool checked){ m_gifBlackWhite = checked; });
        controlsLayout->addWidget(gifBwCheckbox);

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

        QHBoxLayout *gifZoomLayout = new QHBoxLayout;
        QLabel *gifZoomLabel = new QLabel("Масштаб:");
        m_gifZoomSlider = new QSlider(Qt::Horizontal);
        m_gifZoomSlider->setRange(5, 30);
        m_gifZoomSlider->setValue(m_monospaceFont.pointSize());
        connect(m_gifZoomSlider, &QSlider::valueChanged, this, &AsciiArtApp::onGifZoomChanged);
        gifZoomLayout->addWidget(gifZoomLabel);
        gifZoomLayout->addWidget(m_gifZoomSlider);
        layout->addLayout(gifZoomLayout);

        m_gifAsciiDisplay = new QTextEdit;
        m_gifAsciiDisplay->setReadOnly(true);
        m_gifAsciiDisplay->setFont(m_monospaceFont);
        m_gifAsciiDisplay->setLineWrapMode(QTextEdit::NoWrap);
        m_gifAsciiDisplay->setStyleSheet("background-color: black;");
        layout->addWidget(m_gifAsciiDisplay);

        QPushButton *btnSaveGif = new QPushButton("Сохранить GIF");
        connect(btnSaveGif, &QPushButton::clicked, this, &AsciiArtApp::saveGif);
        layout->addWidget(btnSaveGif);
    }

private:
    // Основные элементы интерфейса
    QTabWidget *m_tabWidget;
    QWidget *m_imageTab;
    QWidget *m_videoTab;
    QWidget *m_gifTab;
    QFont m_monospaceFont;

    // Элементы вкладки "Изображение в ASCII"
    QSpinBox *m_imgSpinWidth;
    QLineEdit *m_imgCharsetEdit;
    QComboBox *m_imgPresetCombo;
    QTextEdit *m_imgAsciiDisplay;
    QProgressBar *m_progressImage;
    QSlider *m_imgZoomSlider;
    QString m_currentImagePath;
    bool m_imgBlackWhite;

    // Элементы вкладки "Видео в ASCII"
    QSpinBox *m_videoSpinWidth;
    QLineEdit *m_videoCharsetEdit;
    QComboBox *m_videoPresetCombo;
    QTextEdit *m_videoAsciiDisplay;
    QProgressBar *m_progressVideo;
    QSlider *m_videoZoomSlider;
    QPushButton *m_btnPreprocPlay;
    QPushButton *m_btnStop;
    QString m_currentVideoPath;
    QMediaPlayer *m_player;
    QAudioOutput *m_audioOutput;
    QTimer *m_playTimer;
    qint64 m_videoStartTime;
    int m_currentFrameIndex;
    double m_videoFps;
    size_t m_videoLength;
    std::vector<QString> m_asciiFrames;
    bool m_videoBlackWhite;
    PreprocessingThread *m_preprocThread;

    // Элементы вкладки "GIF в ASCII"
    QSpinBox *m_gifSpinWidth;
    QLineEdit *m_gifCharsetEdit;
    QComboBox *m_gifPresetCombo;
    QTextEdit *m_gifAsciiDisplay;
    QProgressBar *m_progressGif;
    QSlider *m_gifZoomSlider;
    QPushButton *m_btnPreprocGif;
    QPushButton *m_btnStopGif;
    QString m_currentGifPath;
    QTimer *m_gifPlayTimer;
    qint64 m_gifStartTime;
    int m_currentGifFrameIndex;
    double m_gifFps;
    size_t m_gifLength;
    std::vector<QString> m_gifAsciiFrames;
    bool m_gifBlackWhite;
    PreprocessingThread *m_gifPreprocThread;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    AsciiArtApp window;
    window.show();
    return app.exec();
}
