import time
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstApp, GLib

#_ = GstApp

# gst-launch-1.0 pulsesrc ! audioconvert ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! fakesink silent = TRUE

# Only channel zero of a 6-cheannel ReSpeaker, the audioconvert hopefully
# does nothing. YES: checked it, also works without, second line
# gst-launch-1.0 alsasrc device="hw:4" ! deinterleave name=d d.src_0 ! audioconvert ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! fakesink silent = TRUE
# gst-launch-1.0 alsasrc device="hw:4" ! deinterleave name=d d.src_0 ! wavenc ! filesink location="foo.wav"

# TODO: is there any value in the additional things in pipeline 1?
#PIPELINE1 = """pulsesrc ! audioconvert ! audio/x-raw,format=S16LE,channels=1,rate=16000 ! queue ! appsink sync=true max-buffers=1 drop=true name=sink emit-signals=true"""

# rate default is 16k
# For ReSpeaker, picks up result channel of demo mode (channel 0)
# Output is in format audio/x-raw,format=S16LE,channels=6,rate={rate}
PIPELINE_RESPEAKER="""pulsesrc ! audio/x-raw,format=S16LE,channels=6,rate={} ! deinterleave name=d d.src_0 ! appsink name=sink emit-signals=true"""

# For, e.g., Sennheiser headset (stereo, 44100Hz)
PIPELINE_PULSE = """pulsesrc ! audioconvert ! audio/x-raw,format=S16LE,channels=1,rate={} ! appsink name=sink emit-signals=true"""

#PIPELINE=PIPELINE_RESPEAKER
PIPELINE=PIPELINE_PULSE

class GstreamerMicroSink(object):

    def on_new_sample(self, app_sink):
        sample = app_sink.pull_sample()
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("Could not map buffer data!")
        self.callback(map_info.data, map_info.size)
        buffer.unmap(map_info)
        return False

    def __init__(self, callback=lambda buffer: True, pipeline_spec=PIPELINE, rate=16000):
        Gst.init(None)
        self.main_loop = GLib.MainLoop()
        self.pipeline = Gst.parse_launch(pipeline_spec.format(rate))
        self.appsink = self.pipeline.get_by_name("sink")
        self.callback = callback
        cb = lambda appsink: self.on_new_sample(appsink)
        self.handler_id = self.appsink.connect("new-sample", cb)

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.main_loop.quit()

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()


def test(data, size):
    print("got ", type(data), " size ", len(data), " size ", size)

if __name__ == '__main__':
    gms = GstreamerMicroSink(callback=test)
    gms.start()
    time.sleep(3)
    gms.stop()
