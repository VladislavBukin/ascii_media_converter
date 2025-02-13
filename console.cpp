#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Функция для преобразования изображения (Mat) в ASCII-арт строку
string convertMatToAscii(const Mat &img, int desiredWidth, const string &asciiChars) {
    int originalWidth = img.cols;
    int originalHeight = img.rows;
    double aspect = static_cast<double>(originalHeight) / originalWidth;
    // Коэффициент 0.55 корректирует соотношение сторон символа в консоли
    int newHeight = static_cast<int>(desiredWidth * aspect * 0.55);
    if(newHeight < 1) newHeight = 1;
	 
	Mat resized;
    resize(img, resized, Size(desiredWidth, newHeight));

    ostringstream oss;
    for (int i = 0; i < resized.rows; i++) {
        for (int j = 0; j < resized.cols; j++) {
            Vec3b color = resized.at<Vec3b>(i, j);
            int b = color[0], g = color[1], r = color[2];
            // Преобразование в оттенок серого по формуле яркости
            double gray = 0.299 * r + 0.587 * g + 0.114 * b;
			//double gamma = 3.8;
			//double corrected = pow(gray/255.0, gamma) * 255.0;
            int index = static_cast<int>(gray/ 255 * (asciiChars.size() - 1));
            char ch = asciiChars[index];
            // Вывод символа с 24-битной раскраской (ANSI escape sequence)
            oss << "\033[38;2;" << r << ";" << g << ";" << b << "m" << ch << "\033[0m";
        }
        oss << "\n";
    }
    return oss.str();
}

// Функция для очистки консоли с помощью ANSI-кодов
void clearConsole() {
    // Очистка экрана и перевод курсора в начало
    cout << "\033[2J\033[H";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <путь_к_файлу> [ширина_ascii]\n";
        cout << "  <путь_к_файлу> - путь к изображению, GIF или видео\n";
        cout << "  [ширина_ascii] - количество символов по ширине (по умолчанию: 80)\n";
        return 1;
    }

    string inputFile = argv[1];
    int desiredWidth = (argc >= 3) ? atoi(argv[2]) : 80;
    // Набор символов: от «тёмных» (более плотных) к «светлым» (менее плотным)
    string asciiChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
//	string asciiChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/*#MW&8%B@$";
//	string asciiChars = "@%#*+=-:. ";
//	string asciiChars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
//	string asciiChars = "@%#*+=-:;',`. ";
//	string asciiChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
//	string asciiChars = " .:-=+*#%@";
//	string asciiChars = "@%#*oO+=-:. ";
//string asciiChars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
    // Определяем расширение файла
    string fileExtension = "";
    size_t dotPos = inputFile.find_last_of('.');
    if (dotPos != string::npos) {
        fileExtension = inputFile.substr(dotPos + 1);
        transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(), ::tolower);
    }

    // Списки расширений для видео/GIF и изображений
    vector<string> videoExt = {"gif", "mp4", "avi", "mov", "mkv", "wmv"};
    vector<string> imageExt = {"jpg", "jpeg", "png", "bmp", "tiff"};

    bool isVideo = false;
    if (find(videoExt.begin(), videoExt.end(), fileExtension) != videoExt.end()) {
        isVideo = true;
    } else if (find(imageExt.begin(), imageExt.end(), fileExtension) != imageExt.end()) {
        isVideo = false;
    } else {
        // Если расширение неизвестно, пытаемся открыть как видео
        VideoCapture cap(inputFile);
        if (cap.isOpened()) {
            isVideo = true;
            cap.release();
        } else {
            isVideo = false;
        }
    }

	if (isVideo) {
		// Обработка видео или GIF
		if (fileExtension == "gif") {
			// Зацикливаем воспроизведение GIF
			while (true) {
				VideoCapture cap(inputFile);
				if (!cap.isOpened()) {
					cerr << "Ошибка: не удалось открыть файл " << inputFile << endl;
					return 1;
				}
				double fps = cap.get(CAP_PROP_FPS);
				if (fps <= 0) fps = 10;

				Mat frame;
				while (cap.read(frame)) {
					if (frame.empty()) break;
					string asciiFrame = convertMatToAscii(frame, desiredWidth, asciiChars);
					clearConsole();
					cout << asciiFrame << flush;
					this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000.0 / fps)));
				}
				cap.release();
			}
		} else {
			// Обработка обычного видео (без зацикливания)
			VideoCapture cap(inputFile);
			if (!cap.isOpened()) {
				cerr << "Ошибка: не удалось открыть файл " << inputFile << endl;
				return 1;
			}
			double fps = cap.get(CAP_PROP_FPS);
			if (fps <= 0) fps = 10;
			Mat frame;
			while (cap.read(frame)) {
				if (frame.empty()) break;
				string asciiFrame = convertMatToAscii(frame, desiredWidth, asciiChars);
				clearConsole();
				cout << asciiFrame << flush;
				this_thread::sleep_for(chrono::milliseconds(static_cast<int>(1000.0 / fps)));
			}
			cap.release();
		}
	} else {
		// Обработка статичного изображения
		Mat img = imread(inputFile, IMREAD_COLOR);
		if (img.empty()) {
			cerr << "Ошибка: не удалось загрузить изображение " << inputFile << endl;
			return 1;
		}
		string asciiImage = convertMatToAscii(img, desiredWidth, asciiChars);
		cout << asciiImage;
	}

	return 0;
}
