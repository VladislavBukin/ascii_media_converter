cmake_minimum_required(VERSION 3.16)

project(AsciiArtConverter VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Автоматическая генерация MOC, UIC и RCC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Widgets Multimedia REQUIRED)
find_package(OpenCV REQUIRED)

# Исключаем opencv_viz, если он присутствует
list(REMOVE_ITEM OpenCV_LIBS opencv_viz)

set(SOURCES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets Qt6::Multimedia ${OpenCV_LIBS})
