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
 * SECTION:element-zchroma
 *
 * FIXME:Describe zchroma here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! zchroma ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>
#include <math.h>

#include "gstzchroma.h"

GST_DEBUG_CATEGORY_STATIC (gst_zchroma_debug);
#define GST_CAT_DEFAULT gst_zchroma_debug

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

GST_BOILERPLATE (GstZChroma, gst_zchroma, GstElement,
    GST_TYPE_ELEMENT);

static void gst_zchroma_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_zchroma_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_zchroma_set_caps (GstPad * pad, GstCaps * caps);
static GstFlowReturn gst_zchroma_chain (GstPad * pad, GstBuffer * buf);

/* GObject vmethod implementations */

static void
gst_zchroma_base_init (gpointer gclass)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (gclass);

  gst_element_class_set_details_simple(element_class,
    "ZChroma",
    "FIXME:Generic",
    "FIXME:Generic Template Element",
    " <<user@hostname.org>>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_factory));
}

/* initialize the zchroma's class */
static void
gst_zchroma_class_init (GstZChromaClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_zchroma_set_property;
  gobject_class->get_property = gst_zchroma_get_property;

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
gst_zchroma_init (GstZChroma * filter,
    GstZChromaClass * gclass)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_setcaps_function (filter->sinkpad,
                                GST_DEBUG_FUNCPTR(gst_zchroma_set_caps));
  gst_pad_set_getcaps_function (filter->sinkpad,
                                GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));
  gst_pad_set_chain_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_zchroma_chain));

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_pad_set_getcaps_function (filter->srcpad,
                                GST_DEBUG_FUNCPTR(gst_pad_proxy_getcaps));

  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);
  filter->silent = FALSE;
}

static void
gst_zchroma_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstZChroma *filter = GST_ZCHROMA (object);

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
gst_zchroma_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstZChroma *filter = GST_ZCHROMA (object);

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
gst_zchroma_set_caps (GstPad * pad, GstCaps * caps)
{
  GstZChroma *filter;
  GstPad *otherpad;

  filter = GST_ZCHROMA (gst_pad_get_parent (pad));
  otherpad = (pad == filter->srcpad) ? filter->sinkpad : filter->srcpad;
  gst_object_unref (filter);

  return gst_pad_set_caps (otherpad, caps);
}


// http://www.swageroo.com/wordpress/how-to-program-a-gaussian-blur-without-using-3rd-party-libraries/
void gaussianWeights(const float sigma, int filterWidth, float **filter)
{
    int center = filterWidth / 2;
    int i = 0;
    *filter = new float[filterWidth * filterWidth];

    for (int a = -center; a <= center; a++)
    {
        for (int b = -center; b <= center; b++)
        {
            int x = abs(a);
            int y = abs(b);
            (*filter)[i] = 1.0f / (2.0f * M_PI * pow(sigma, 2)) * expf(-((pow(x, 2) + pow(y, 2)) / (2.0f * pow(sigma, 2))));
            i++;
        }
    }

    return;
}


guint64 RealTime3;
gint NumberFrames3 = 0;
gfloat ElapsedTimeSum3 = 0;

/* chain function
 * this function does the actual processing
 */
extern void zchroma_transform( guint8 *data, gint width, gint height, gint filterWidth, float *d_filter );

static GstFlowReturn
gst_zchroma_chain (GstPad * pad, GstBuffer * buf)
{
  GstZChroma *filter;
  gint width, height;
  GstCaps *caps;
  guint8 *data;
  int filterWidth = 3;
  const float sigma = 1.5;
  float *d_filter;

  filter = GST_ZCHROMA (GST_OBJECT_PARENT (pad));

  if (filter->silent == FALSE)
    //g_print ("I'm plugged, therefore I'm in.\n");
    caps = GST_BUFFER_CAPS(buf);

    const GstStructure *str;

    str = gst_caps_get_structure (caps, 0);
    gst_structure_get_int (str, "width", &width);
    gst_structure_get_int (str, "height", &height);
    size_t buf_size = GST_BUFFER_SIZE (buf);
    //g_print("Video with size %dx%d, buffer size: %d.\n", width, height, buf_size);

    data = GST_BUFFER_DATA (buf);

  gaussianWeights( sigma, filterWidth, &d_filter );

  NumberFrames3++;
  RealTime3 = g_get_monotonic_time();

  zchroma_transform( data, width, height, filterWidth, d_filter );

  gfloat ElapsedTime3 = (g_get_monotonic_time() - RealTime3) / 1000.0; // milliseconds
  ElapsedTimeSum3 += ElapsedTime3;
  if (NumberFrames3 % 100 == 0){
    g_print("Video with size %dx%d processed %f frames in second, about %f ms for frame.\n",
            width, height, 1000.0 / (ElapsedTimeSum3 / 100.0), ElapsedTimeSum3 / 100.0);
    ElapsedTimeSum3 = 0;
  }

  /* just push out the incoming buffer without touching it */
  return gst_pad_push (filter->srcpad, buf);
}


