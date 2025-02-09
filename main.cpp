#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Проверка количества аргументов
    if (argc < 2) {
        std::cout << "Использование: " << argv[0] << " <путь_к_изображению> [выходной_файл] [ширина] [charset]\n";
        std::cout << "  <путь_к_изображению> - путь к входному файлу (например, image.jpg)\n";
        std::cout << "  [выходной_файл]     - имя файла для сохранения результата (по умолчанию: output.html)\n";
        std::cout << "  [ширина]             - количество символов по ширине (по умолчанию: 80)\n";
        std::cout << "  [charset]            - набор символов (по умолчанию: \"@%#*+=-:. \")\n";
        return 1;
    }

    // Парсинг аргументов
    std::string inputFile = argv[1];
    std::string outputFile = (argc >= 3) ? argv[2] : "output.html";
    int desiredWidth = (argc >= 4) ? std::atoi(argv[3]) : 80;
    std::string asciiChars = (argc >= 5) ? argv[4] : "@%#*+=-:. ";

    // Загружаем изображение с помощью OpenCV
    cv::Mat img = cv::imread(inputFile, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение \"" << inputFile << "\"\n";
        return 1;
    }

    // Вычисляем новые размеры с учетом соотношения сторон и корректирующего коэффициента
    int originalWidth = img.cols;
    int originalHeight = img.rows;
    double aspect = static_cast<double>(originalHeight) / originalWidth;
    int newHeight = static_cast<int>(desiredWidth * aspect * 0.55);
    if (newHeight < 1) newHeight = 1;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(desiredWidth, newHeight));

    // Формируем HTML-код с использованием <span> для задания цвета каждого символа
    std::stringstream html;
    html << "<html>\n<head>\n<meta charset=\"UTF-8\">\n<title>ASCII Art</title>\n</head>\n<body style=\"background-color: black;\">\n<pre style=\"font: 10px/5px monospace;\">\n";

    // Для каждого пикселя формируем символ с цветом
    for (int i = 0; i < resized.rows; i++) {
        for (int j = 0; j < resized.cols; j++) {
            cv::Vec3b color = resized.at<cv::Vec3b>(i, j);
            int b = color[0];
            int g = color[1];
            int r = color[2];
            double gray = 0.299 * r + 0.587 * g + 0.114 * b;
            int index = static_cast<int>(gray / 255 * (asciiChars.size() - 1));
            char ch = asciiChars[index];
            html << "<span style=\"color: rgb(" << r << "," << g << "," << b << ")\">" << ch << "</span>";
        }
        html << "<br>\n";
    }
    html << "</pre>\n</body>\n</html>";

    // Сохраняем HTML в выходной файл в текущей папке
    std::ofstream out(outputFile);
    if (!out) {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << outputFile << "\n";
        return 1;
    }
    out << html.str();
    out.close();

    std::cout << "ASCII art успешно сохранён в файле \"" << outputFile << "\"\n";
    return 0;
}
