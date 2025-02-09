#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Проверка аргументов: обязательно передан путь к изображению
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению> [ширина] [charset]\n";
        std::cout << "  <путь_к_изображению> - путь к входному изображению (например, image.jpg)\n";
        std::cout << "  [ширина]             - количество символов по ширине (по умолчанию: 80)\n";
        std::cout << "  [charset]            - набор символов (по умолчанию: \"@%#*+=-:. \")\n";
        return 1;
    }

    // Чтение аргументов командной строки
    std::string inputFile = argv[1];
    int desiredWidth = (argc >= 3) ? std::atoi(argv[2]) : 80;
    std::string asciiChars = (argc >= 4) ? argv[3] : "@%#*+=-:. ";

    // Загружаем изображение в формате BGR (OpenCV по умолчанию)
    cv::Mat img = cv::imread(inputFile, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение \"" << inputFile << "\"\n";
        return 1;
    }

    // Вычисляем новые размеры с учётом соотношения сторон и коррекции для консоли
    int originalWidth = img.cols;
    int originalHeight = img.rows;
    double aspect = static_cast<double>(originalHeight) / originalWidth;
    int newHeight = static_cast<int>(desiredWidth * aspect * 0.55);
    if (newHeight < 1) newHeight = 1;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(desiredWidth, newHeight));

    // Для каждого пикселя выводим символ с соответствующим цветом.
    // ANSI-escape код для 24-битного цвета имеет формат: "\033[38;2;<r>;<g>;<b>m"
    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            cv::Vec3b color = resized.at<cv::Vec3b>(i, j);
            int b = color[0];
            int g = color[1];
            int r = color[2];
            // Вычисляем оттенок серого для выбора символа
            double gray = 0.299 * r + 0.587 * g + 0.114 * b;
            int index = static_cast<int>(gray / 255 * (asciiChars.size() - 1));
            char ch = asciiChars[index];
            // Выводим символ с цветом
            std::cout << "\033[38;2;" << r << ";" << g << ";" << b << "m" << ch;
        }
        // Сброс цвета и переход на новую строку
        std::cout << "\033[0m" << "\n";
    }

    return 0;
}
