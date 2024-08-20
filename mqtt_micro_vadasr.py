#!/usr/bin/env python3

import sys
import asyncio
import logging
import yaml
import wave
import time
import torch
import resampy
import numpy as np

from vosk import Model, KaldiRecognizer, SetLogLevel

import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import gstmicpipeline as gm

from vad_iterator import VADIterator

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

def init_jit_model(model_path: str, device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

class VoskMicroServer():

    MAX_BUF_RETENTION = 40
    MIN_SPEECH_DETECTS = 3
    MIN_SILENCE_DETECTS = 30
    #BUFFER_SIZE = 1536
    BUFFER_SIZE = 512

    def __init__(self, config):
        self.pid = "voskasr"
        self.audio_dir = "audio/"
        self.language = "en-us"

        self.channels = 1
        self.usedchannel = 0
        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.buffers_queued = 6

        self.model_path = None

        self.config = config
        if 'asr_sample_rate' in config:
            self.asr_sample_rate = config['asr_sample_rate']
        if 'channels' in config:
            self.channels = config['channels']
        if 'use_channel' in config:
            self.usedchannel = config['use_channel']
        if 'audio_dir' in config:
            self.audio_dir = config['audio_dir']
        if 'language' in config:
            self.language = config['language']
        if 'model_path' in config:
            self.model_path = config['model_path']
        if 'buffers_queued' in config:
            self.buffers_queued = config['buffers_queued']
        self.topic = self.pid + '/asrresult'
        if self.language:
            slashpos = self.language.find('_')
            suff = self.language
            if slashpos >= 0:
                suff = suff[:slashpos]
            self.topic += '/' + suff
        self.loop = asyncio.get_running_loop()
        self.audio_queue = asyncio.Queue()
        self.__init_mqtt_client()
        # create 100 ms buffer with silence (2 bytes per sample): / 1000 * 100
        self.silence_buffer = bytearray(int(1000 * 2 * (self.asr_sample_rate / 1000)))
        # load silero VAD model
        model = init_jit_model(model_path='silero_vad.jit')

        vad_config = config['vad'] if 'vad' in config else dict()
        #print(type(vad_config['threshold']))
        self.vad_iterator = VADIterator(model, **vad_config)
        #self.threshold = 0.5
        #print(f'{self.asr_sample_rate} {self.sample_rate} {self.channels}')

        self.__init_recognizer()

        # for monitoring (eventually)
        self.am = None
        self.wf = None

    def __init_recognizer(self):
        self.asr_model = Model(lang=self.language, model_path=self.model_path)
        self.recognizer = KaldiRecognizer(self.asr_model, self.asr_sample_rate)
        self.recognizer.SetMaxAlternatives(10)
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
        if channels > 1:
            # numpy slicing:
            # take every i'th value: frame[start:stop:step]
            frame = frame[self.usedchannel::channels]
        if sample_rate != self.asr_sample_rate:
            frame = resampy.resample(frame, sample_rate, self.asr_sample_rate)
            frame = frame.astype(np.int16)
        return frame

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
    def check_result(self, transcribe, voice_start):
        data = json.loads(transcribe)
        # PRELIMINARY SOLUTION FOR MULTIPLE ALTERNATIVES
        if data and 'alternatives' in data:
            data = data['alternatives'][0]
        if data and 'text' in data:
            text = data['text']
            if text != '' and text != 'einen' and text != 'bin' and text != 'the':
                # TODO: not sure if we need this, maybe the MQTT message id is
                # enough?
                data['start'] = voice_start
                data['end'] = current_milli_time()
                print(data)
                self.client.publish(self.topic, json.dumps(data, indent=None))
                return data['end']
        return voice_start

    def send_frames(self, audio, voice_start):
        if self.recognizer.AcceptWaveform(audio):
            return self.check_result(self.recognizer.Result(), voice_start)
        else:
            # partial result in self.recognizer.PartialResult()
            pass
        return voice_start

    async def audio_loop(self):
        logger.info(f'sample_rate: {self.asr_sample_rate}')
        is_voice = None
        window_size_samples = VoskMicroServer.BUFFER_SIZE
        framesqueued = window_size_samples * self.buffers_queued
        voice_buffers = [0] * framesqueued
        out_buffer = []
        while True:
            while len(voice_buffers) < window_size_samples + framesqueued:
                # this is a byte buffer
                audio = await self.audio_queue.get()
                frame = np.frombuffer(audio, dtype=np.int16)
                # monitor what comes in
                if self.am:
                    self.am.writeframes(audio)
                voice_buffers.extend(frame.tolist())

            ichunk = voice_buffers[framesqueued:framesqueued + window_size_samples]
            chunk = np.array(ichunk, dtype=np.int16)
            vadbuf = chunk / 32768
            speech_dict = self.vad_iterator(vadbuf, return_seconds=True)
            if speech_dict:
                #print(f'{speech_dict}')
                if "start" in speech_dict:
                    is_voice = current_milli_time()
                    print('<', end='', flush=True)
                    # monitor what is sent to the ASR
                    if "monitor_asr" in self.config:
                        self.wf = self.open_wave_file(
                        self.asrmon_filename(), self.asr_sample_rate)
                    # add queued buffers to the outbuffer
                    out_buffer = voice_buffers[:framesqueued]
                    audiodata = np.array(out_buffer, dtype=np.int16).tobytes()
                    self.writeframes(audiodata)
                    is_voice = self.send_frames(audiodata, is_voice)
                elif "end" in speech_dict:
                    if not is_voice:
                        print('VAD end ignored')
                        break
                    voice_start = is_voice
                    is_voice = None
                    print('>', end='', flush=True)
                    self.writeframes(chunk.tobytes())
                    voice_start = self.send_frames(chunk.tobytes(), voice_start)
                    #self.writeframes(self.silence_buffer)
                    self.check_result(self.recognizer.FinalResult(), voice_start)
                    out_buffer = []
                    if self.wf:
                        self.wf.close()
                        self.wf = None
            voice_buffers = voice_buffers[window_size_samples:]
            if is_voice:
                self.writeframes(chunk.tobytes())
                is_voice = self.send_frames(chunk.tobytes(), is_voice)


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
               'monitor_asr':True,
               'language': 'en-us'
              }
    if len(args) >= 1:
        with open(args[0], 'r') as f:
            config.update(yaml.safe_load(f))

    ms = VoskMicroServer(config)

    logging.basicConfig(level=logging.INFO)
    await ms.run_micro()

if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:]))
