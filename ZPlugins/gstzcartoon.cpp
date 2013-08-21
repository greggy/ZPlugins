/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2013  <<user@hostname.org>>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-zcartoon
 *
 * FIXME:Describe zcartoon here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! zcartoon ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <cuda.h>
#include <sys/param.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>

#include "gstzcartoon.h"

GST_DEBUG_CATEGORY_STATIC (gst_zcartoon_debug);
#define GST_CAT_DEFAULT gst_zcartoon_debug


/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,    
    GST_STATIC_CAPS( GST_VIDEO_CAPS_BGRA )
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

GST_BOILERPLATE (Gstzcartoon, gst_zcartoon, GstElement,
    GST_TYPE_ELEMENT);

static void gst_zcartoon_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_zcartoon_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_zcartoon_set_caps (GstPad * pad, GstCaps * caps);
static GstFlowReturn gst_zcartoon_chain (GstPad * pad, GstBuffer * buf);

/* GObject vmethod implementations */

static void
gst_zcartoon_base_init (gpointer gclass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (gclass);

  gst_element_class_set_details_simple(element_class,
    "zcartoon",
    "FIXME:Generic",
    "FIXME:Generic Template Element",
    " <<user@hostname.org>>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_factory));
}

/* initialize the zcartoon's class */
static void
gst_zcartoon_class_init (GstzcartoonClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_zcartoon_set_property;
  gobject_class->get_property = gst_zcartoon_get_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, static_cast < GParamFlags >G_PARAM_READWRITE));
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_zcartoon_init (Gstzcartoon * filter,
    GstzcartoonClass * gclass)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_setcaps_function (filter->sinkpad,
                                GST_DEBUG_FUNCPTR(gst_zcartoon_set_caps));
  gst_pad_set_getcaps_function (filter->sinkpad,
                                GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));
  gst_pad_set_chain_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_zcartoon_chain));

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_pad_set_getcaps_function (filter->srcpad,
                                GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));

  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);
  filter->silent = FALSE;
}

static void
gst_zcartoon_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstzcartoon *filter = GST_ZCARTOON (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_zcartoon_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstzcartoon *filter = GST_ZCARTOON (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstElement vmethod implementations */

/* this function handles the link with other elements */
static gboolean
gst_zcartoon_set_caps (GstPad * pad, GstCaps * caps)
{
  Gstzcartoon *filter;
  GstPad *otherpad;

  filter = GST_ZCARTOON (gst_pad_get_parent (pad));
  otherpad = (pad == filter->srcpad) ? filter->sinkpad : filter->srcpad;
  gst_object_unref (filter);

  return gst_pad_set_caps (otherpad, caps);
}

static void
mytransform (GstBuffer * buf)
{
  gint width, height;
  GstCaps * caps;
  gint pixelPos;
  guint8 *data;

  caps = GST_BUFFER_CAPS(buf);

  const GstStructure *str;

  str = gst_caps_get_structure (caps, 0);
  gst_structure_get_int (str, "width", &width);
  gst_structure_get_int (str, "height", &height);

  //g_print("Video size %d x %d\n", width, height);

  data = GST_BUFFER_DATA (buf);

  for (int i=0; i<height; i++) {
    for (int j=0; j<width; j++) {
      pixelPos = (i + j * height) * 4; // Index of pixel
      data[pixelPos + 2] = 0; // BGRA format
    }
  }

  /* Free the buffer */
  //gst_buffer_unref (buf);
}


/* Cartoon algorithm
 * -----------------
 * Mask radius = radius of pixel neighborhood for intensity comparison
 * Threshold   = relative intensity difference which will result in darkening
 * Ramp        = amount of relative intensity difference before total black
 * Blur radius = mask radius / 3.0
 *
 * Algorithm:
 * For each pixel, calculate pixel intensity value to be: avg (blur radius)
 * relative diff = pixel intensity / avg (mask radius)
 * If relative diff < Threshold
 *   intensity mult = (Ramp - MIN (Ramp, (Threshold - relative diff))) / Ramp
 *   pixel intensity *= intensity mult
 */

static void
zcartoon_transform_cpu (GstBuffer * buf)
{
  gint i_width, i_height;
  GstCaps * caps;
  gint m_pixelPos, n_pixelPos;
  guint8 *data;
  GstBuffer *o_buf;
  guint8 *o_data;
  gint m_mask_radius = 7;
  gdouble m_threshold = 1.0;
  gdouble m_ramp = 0.1;

  caps = GST_BUFFER_CAPS(buf);

  const GstStructure *str;

  str = gst_caps_get_structure (caps, 0);
  gst_structure_get_int (str, "width", &i_width);
  gst_structure_get_int (str, "height", &i_height);

  //g_print("Video size %d x %d\n", width, height);

  data = GST_BUFFER_DATA (buf);

  // copy buf to o_buf
  o_buf = gst_buffer_copy (buf);
  o_data = GST_BUFFER_DATA (o_buf);

  gdouble size = m_mask_radius * m_mask_radius;

  gint center = m_mask_radius / 2 + 1,
          width = i_width - center,
          height = i_height - center,
          top = m_mask_radius / 2;

  for(gint y = center; y < height; y++){
    for(gint x = center; x < width; x++){

      m_pixelPos = (y * i_width + x) * 4; // main pixel

      // get neighbour pixels
      gint i = 0;
      gdouble sumR = 0, sumB = 0, sumG = 0;
      for(gint iX = x-top; i < m_mask_radius; ++i, ++iX){

        gint j = 0;
        for(gint iY = y-top; j < m_mask_radius; ++j, ++iY){

          n_pixelPos = (iY * i_width + iX) * 4; // neighbour pixel
          sumR += o_data[n_pixelPos + 2];
          sumB += o_data[n_pixelPos + 0];
          sumG += o_data[n_pixelPos + 1];
          //g_print("x: %d; y: %d; iX: %d; iY: %d; m_pixel: %d; n_pixel: %d\n", x, y, iX, iY, m_pixelPos, n_pixelPos);
        }
      }

      sumR /= size;
      sumB /= size;
      sumG /= size; 

      gdouble red = o_data[m_pixelPos + 2],
              blue = o_data[m_pixelPos + 0],
              green = o_data[m_pixelPos + 1];
      //g_print("red: %f, blue: %f, green: %f; pixel: %d\n", red, blue, green, m_pixelPos);

      gdouble koeffR = red / sumR,
              koeffB = blue / sumB,
              koeffG = green / sumG;

      if(koeffR < m_threshold)
          red *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffR)))/m_ramp);

      if(koeffB < m_threshold)
          blue *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffB)))/m_ramp);

      if(koeffG < m_threshold)
          green *= ((m_ramp - MIN(m_ramp,(m_threshold - koeffG)))/m_ramp);

      if (red > 255 || blue > 255 || green > 255)
          g_print("red: %f, blue: %f, green: %f;\n", red, blue, green);

      data[m_pixelPos + 2] = red;
      data[m_pixelPos + 0] = blue;
      data[m_pixelPos + 1] = green;

    }
  }

  /* Free the buffer */
  //gst_buffer_unref (buf);
  gst_buffer_unref (o_buf);
}


guint64 RealTime;
gint NumberFrames = 0;
gfloat ElapsedTimeSum = 0;

extern void zcartoon1_transform( guint8 *data, gint width, gint height );
extern void zcartoon2_transform( guint8 *data, gint width, gint height );
extern void zcartoon3_transform( guint8 *data, gint width, gint height );
extern void zcartoon4_transform( guint8 *data, gint width, gint height );
extern void simple_transform( guint8 *data, gint width, gint height );
extern void boxfilter1_transform( guint8 *data, gint width, gint height );
extern void boxfilter2_transform( guint8 *data, gint width, gint height );
//extern int test( gint len );

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_zcartoon_chain (GstPad * pad, GstBuffer * buf)
{
  Gstzcartoon *filter;
  gint width, height;
  GstCaps *caps;
  guint8 *data;
  GstBuffer *o_buf;
  guint8 *o_data;
  glong size;
  gint len;

  filter = GST_ZCARTOON (GST_OBJECT_PARENT (pad));

  if (filter->silent == FALSE)
    //g_print ("I'm plugged, therefore I'm in.\n");
    //mytransform (buf);
    //zcartoon_transform_cpu (buf);

    caps = GST_BUFFER_CAPS(buf);

    const GstStructure *str;

    str = gst_caps_get_structure (caps, 0);
    gst_structure_get_int (str, "width", &width);
    gst_structure_get_int (str, "height", &height);
    //g_print("Video size %d x %d\n", width, height);

    data = GST_BUFFER_DATA (buf);

    NumberFrames++;
    RealTime = g_get_monotonic_time();

    //zcartoon1_transform ( data, width, height );
    //zcartoon2_transform ( data, width, height );
    //zcartoon3_transform ( data, width, height );
    //zcartoon4_transform ( data, width, height );
    boxfilter1_transform ( data, width, height );
    //boxfilter2_transform ( data, width, height );
    //simple_transform ( data, width, height );
    //test( len );

    gfloat ElapsedTime = (g_get_monotonic_time() - RealTime) / 1000.0; // milliseconds
    ElapsedTimeSum += ElapsedTime;
    if (NumberFrames % 100 == 0){
        g_print("Video with size %dx%d processed %f frames in second, about %f ms for frame.\n",
                width, height, 1000.0 / (ElapsedTimeSum / 100.0), ElapsedTimeSum / 100.0);
        ElapsedTimeSum = 0;
    }

  /* just push out the incoming buffer without touching it */
  return gst_pad_push (filter->srcpad, buf);
}
