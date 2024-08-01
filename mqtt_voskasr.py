#!/usr/bin/env python3

import sys
import asyncio
import logging
import yaml
import wave
import time
import resampy
import numpy as np

from vosk import Model, KaldiRecognizer, SetLogLevel

import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import gstmicpipeline as gm

# configure logger
logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

MAX_RECONNECTS = 40
RECONNECT_WAIT = 5  # SECONDS


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def current_milli_time():
    return round(time.time() * 1000)

class VoskMicroServer():

    #BUFFER_SIZE = 1536
    BUFFER_SIZE = 512

    def __init__(self, config):
        self.pid = "voskasr"
        self.audio_dir = "audio/"
        self.language = "de"

        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.channels = 1

        self.config = config
        if 'asr_sample_rate' in config:
            self.asr_sample_rate = config['asr_sample_rate']
        if 'audio_dir' in config:
            self.audio_dir = config['audio_dir']
        if 'language' in config:
            self.language = config['language']
        self.topic = self.pid + '/asrresult'
        if self.language:
            self.topic += '/' + self.language
        self.loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self.__init_mqtt_client()
        #self.model_path = "/opt/models/kaldi_models/vosk-model-en-us-0.22"
        self.model_path = "/opt/models/kaldi_models/vosk-model-de-0.6"
        self.__init_recognizer()

        # for monitoring (eventually)
        self.am = None
        self.wf = None

    def __init_recognizer(self):
        self.asr_model = Model(lang=self.language, model_path=self.model_path)
        self.recognizer = KaldiRecognizer(self.asr_model, self.asr_sample_rate)
        self.recognizer.SetMaxAlternatives(5)
        self.recognizer.SetWords(True)
        logger.info("Vosk model initialized")

    def __init_mqtt_client(self):
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        # self.client.username_pw_set(self.mqtt_username, self.mqtt_password)
        # self.client.on_connect = self.__on_mqtt_connect

    def wav_filename(self):
        return self.audio_dir + 'chunk-%d.wav' % (time.time())

    def open_wave_file(self, path, sample_rate):
        """Opens a .wav file.
        Takes path, number of channels and sample rate.
        """
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        return wf

    def asrmon_filename(self):
        return self.audio_dir + 'asrmon-%d.wav' % (time.time())

    def open_asrmon_file(self, path):
        """Opens a .wav file.
        Takes path, number of channels and sample rate.
        """
        am = wave.open(path, 'wb')
        am.setnchannels(1)
        am.setsampwidth(2)
        am.setframerate(self.asr_sample_rate)
        return am

    def writeframes(self, audio):
        if self.wf:
            self.wf.writeframes(audio)

    def resample(self, frame, channels, sample_rate):
        if channels == 1 and sample_rate == self.asr_sample_rate:
            return frame
        frame = np.frombuffer(frame, dtype=np.int16)
        if channels > 1:
            # numpy slicing:
            # take every i'th value: frame[start:stop:step]
            frame = frame[self.usedchannel::channels]
        if sample_rate != self.asr_sample_rate:
            frame = resampy.resample(frame, sample_rate, self.asr_sample_rate)
            frame = frame.astype(np.int16)
        return frame.tobytes()

    def callback(self, indata, frames, time_block, status):
        """This is called (from a separate thread) for each audio block."""
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait,
                                       bytes(indata))

    def mqtt_connect(self):
        self.client.connect(self.config['mqtt_address'])
        self.client.loop_start()

    def mqtt_disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


    # Send a result returned from the ASR to the MQTT topic
    def check_result(self, transcribe):
        data = json.loads(transcribe)
        print("FINAL: " + transcribe + " ", end='')
        if data and 'text' in data:
            text = data['text']
            print(text)
            if text != '' and text != 'einen' and text != 'bin' and text != 'the':
                # TODO: not sure if we need this, maybe the MQTT message id is
                # enough?
                data['start'] = self.voice_start if self.voice_start else -1
                data['end'] = current_milli_time()
                self.voice_start = None
                print(data)
                self.client.publish(self.topic, json.dumps(data, indent=None))

    def send_frames(self, audio):
        if self.recognizer.AcceptWaveform(audio):
            self.check_result(self.recognizer.Result())
        else:
            # partial result in self.recognizer.PartialResult()
            partial = json.loads(self.recognizer.PartialResult())
            if partial["partial"]:
                self.voice_start = current_milli_time()
                print(partial["partial"])
            else:
                if self.voice_start:
                    #self.check_result(self.recognizer.FinalResult())
                    self.voice_start = None
            pass

    async def audio_loop(self):
        logger.info(f'sample_rate: {self.asr_sample_rate}')
        window_size_samples = VoskMicroServer.BUFFER_SIZE
        self.voice_start = None
        is_voice = None
        while True:
            audio = await self.audio_queue.get()
            asr_audio = self.resample(audio, self.channels, self.sample_rate)
            if self.am:
                self.am.writeframes(asr_audio)
            self.send_frames(asr_audio)
            if not is_voice:
                if self.voice_start:
                    print('<', end='', flush=True)
                    is_voice = True
                    if "monitor_asr" in self.config:
                        self.wf = self.open_wave_file(
                        self.asrmon_filename(), self.asr_sample_rate)
            else:
                if not self.voice_start:
                     print('>', end='', flush=True)
                     is_voice = False
                     if self.wf:
                        self.wf.close()
                        self.wf = None
            if is_voice and self.wf:
                self.wf.writeframes(asr_audio)


    async def run_micro(self):
        cb = lambda inp, frames: self.callback(inp, frames, None, None)
        pipeline = self.config["pipeline"] if "pipeline" in self.config \
            else gm.PIPELINE
        try:
            with gm.GstreamerMicroSink(callback=cb, pipeline_spec=pipeline) \
                 as device:
                if "monitor_mic" in self.config:
                    self.am = self.open_wave_file(self.wav_filename(),
                                                  self.sample_rate)

                print("Connecting to MQTT broker")
                self.mqtt_connect()
                await self.audio_loop()
        finally:
            print('Disconnecting...')
            if self.am:
                self.am.close()
            self.mqtt_disconnect()

async def main(args):
    #if len(args) < 1:
    #    sys.stderr.write('Usage: %s <config.yaml> [audio_file(s)]\n' % args[0])
    #    sys.exit(1)

    config = { 'mqtt_address':'localhost',
               'monitor_mic':True,
               'monitor_asr':False
              }
    if len(args) >= 1:
        with open(args[0], 'r') as f:
            config = yaml.safe_load(f)

    ms = VoskMicroServer(config)

    logging.basicConfig(level=logging.INFO)
    await ms.run_micro()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
