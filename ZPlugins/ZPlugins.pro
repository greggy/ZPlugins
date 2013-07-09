#-------------------------------------------------
#
# Project created by QtCreator 2013-06-24T15:10:30
#
#-------------------------------------------------

QT       -= core gui

TARGET = ZPlugins
TEMPLATE = lib

DEFINES += ZPLUGINS_LIBRARY

SOURCES += \
    Plugin.cpp \
    gstzdenoiser.cpp \
    gstzcartoon2.cpp \
    gstzcartoon.cpp

HEADERS +=\
    gstzdenoiser.h \
    gstzcartoon2.h \
    gstzcartoon.h

symbian {
    MMP_RULES += EXPORTUNFROZEN
    TARGET.UID3 = 0xE6C8A36A
    TARGET.CAPABILITY = 
    TARGET.EPOCALLOWDLLDATA = 1
    addFiles.sources = ZPlugins.dll
    addFiles.path = !:/sys/bin
    DEPLOYMENT += addFiles
}

unix:!symbian {
    maemo5 {
        target.path = /opt/usr/lib
    } else {
        target.path = /usr/lib
    }
    INSTALLS += target
}

INCLUDEPATH += /opt/gstreamer-sdk/include/
INCLUDEPATH += /opt/gstreamer-sdk/include/gstreamer-0.10/
INCLUDEPATH += /opt/gstreamer-sdk/include/glib-2.0/
INCLUDEPATH += /opt/gstreamer-sdk/lib/glib-2.0/include/
INCLUDEPATH += /opt/gstreamer-sdk/include/libxml2/

CONFIG += link_pkgconfig
PKGCONFIG += gstreamer-0.10
