#------------------------------------------------
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
    gstzcartoon.cpp \
    test.cu \
    zcartoon_transform.cu

#SOURCES -= \
#    gstzcartoon.cu

HEADERS +=\
    gstzdenoiser.h \
    gstzcartoon2.h \
    gstzcartoon.h \
    utils.h

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

# CUDA
CUDA_SOURCES += test.cu

# Project dir and outputs
PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/../ZPlugins-build
DESTDIR = $$PROJECT_DIR/../ZPlugins-build

# Path to cuda toolkit install
CUDA_DIR = /usr/lib/nvidia-cuda-toolkit
# GPU architecture
CUDA_ARCH = sm_20
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# include paths
INCLUDEPATH += $$CUDA_DIR/include
# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib

LIBS += -lcuda -lcudart
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda
