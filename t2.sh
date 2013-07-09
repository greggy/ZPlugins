#!/bin/bash

gst-launch-0.10 --gst-plugin-path=./ZPlugins-build videotestsrc is-live=true ! video/x-raw-rgb,bpp=32,depth=32,framerate=\(fraction\)25/1,width=320,height=240 ! zcartoon ! ffmpegcolorspace ! autovideosink

