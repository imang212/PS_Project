import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

class RtspMediaFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super().__init__()
        self.set_launch(
            "( libcamerasrc ! video/x-raw,format=NV12,width=1536,height=864,framerate=30/1 "
            "! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast "
            "! rtph264pay name=pay0 pt=96 )"
        )
        self.set_shared(True)

server = GstRtspServer.RTSPServer()
factory = RtspMediaFactory()
mounts = server.get_mount_points()
mounts.add_factory("/test", factory)
server.attach(None)

print("RTSP server running at rtsp://localhost:8554/test")

loop = GLib.MainLoop()
loop.run()

"""
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.37.205:8554/hailo_stream latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
"""