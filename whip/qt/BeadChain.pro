QT       += core gui widgets
CONFIG   += c++17
TARGET    = BeadChain
TEMPLATE  = app

SOURCES += \
    src/main.cpp \
    src/Bead.cpp \
    src/ChainSimulator.cpp \
    src/ChainWidget.cpp \
    src/MainWindow.cpp

HEADERS += \
    src/Bead.h \
    src/ChainSimulator.h \
    src/ChainWidget.h \
    src/MainWindow.h
