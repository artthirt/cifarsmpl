#-------------------------------------------------
#
# Project created by QtCreator 2017-02-13T10:31:13
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = cifarsmpl
TEMPLATE = app

INCLUDEPATH += $$PWD/models/

SOURCES += main.cpp\
        mainwindow.cpp \
    widgetcifar.cpp \
    models/cifar_reader.cpp \
    models/cifar_train.cpp \
    models/convnnf.cpp

HEADERS  += mainwindow.h \
    widgetcifar.h \
    models/cifar_reader.h \
    models/cifar_train.h \
    models/convnnf.h

FORMS    += mainwindow.ui \
    widgetcifar.ui

win32{
QMAKE_CXXFLAGS += /openmp
}

unix{
QMAKE_CXXFLAGS += -fopenmp
LIBS += -lgomp
}

include(ct/ct.pri)
