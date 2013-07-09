#!/bin/bash

gst-launch-0.10 --gst-plugin-path=./ZPlugins-build uridecodebin uri=http://docs.gstreamer.com/media/sintel_trailer-480p.webm ! ffmpegcolorspace ! zcartoon ! ffmpegcolorspace ! autovideosink
