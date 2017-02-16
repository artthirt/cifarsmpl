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
    models/convnnf.cpp \
    drawcnvweight.cpp \
    qt_work_mat.cpp \
    models/gpu_train.cpp

HEADERS  += mainwindow.h \
    widgetcifar.h \
    models/cifar_reader.h \
    models/cifar_train.h \
    models/convnnf.h \
    drawcnvweight.h \
    qt_work_mat.h \
    models/gpu_train.h

FORMS    += mainwindow.ui \
    widgetcifar.ui \
    drawcnvweight.ui

win32{
    QMAKE_CXXFLAGS += /openmp
}

unix{
    QMAKE_CXXFLAGS += -fopenmp
    LIBS += -lgomp
}

CONFIG(debug, debug|release){
    DST = "debug"
}else{
    DST = "release"
}

UI_DIR = tmp/$$DST/ui
OBJECTS_DIR = tmp/$$DST/obj
RCC_DIR = tmp/$$DST/rcc
MOC_DIR = tmp/$$DST/moc


include(ct/ct.pri)
include(gpu/gpu.pri)
