#!/bin/bash

gst-launch-0.10 --gst-plugin-path=../www/ZPlugins/ZPlugins-build -v filesrc location=car-300x226.jpg ! decodebin2 ! imagefreeze ! ffmpegcolorspace ! zcartoon ! ffmpegcolorspace ! autovideosink
