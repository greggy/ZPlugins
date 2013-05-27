// DSPlugin.cpp : Defines the exported functions for the DLL application.
//
#include "gstzcartoon.h"
#include "gstzcartoon2.h"
#include "gstzdenoiser.h"

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template zdsaudio' with your description
   */
  
	if (!gst_element_register (plugin, "zcartoon", GST_RANK_NONE, GST_TYPE_ZCARTOON) ||
		!gst_element_register (plugin, "zcartoon2", GST_RANK_NONE, GST_TYPE_ZCARTOON2) ||
		! gst_element_register (plugin, "zdenoiser", GST_RANK_NONE, GST_TYPE_ZDENOISER)		
		){
		return FALSE;
	}
	
  return TRUE;
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "zplugins"
#endif

/* gstreamer looks for this structure to register zdsaudios
 *
 * exchange the string 'Template zdsaudio' with your zdsaudio description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "zplugins",
    "ZPlugins",
    plugin_init,
    "0.1",
    "LGPL",
    "GStreamer",
    "http://gstreamer.net/"
)

