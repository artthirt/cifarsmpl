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
    models/cifar_reader.cpp

HEADERS  += mainwindow.h \
    widgetcifar.h \
    models/cifar_reader.h

FORMS    += mainwindow.ui \
    widgetcifar.ui

include(ct/ct.pri)
