
#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import argparse
import csv
import asyncio
import logging
import yaml
import wave
import time
import torch
import resampy
import numpy as np
import queue
import traceback
from threading import Thread
import requests

from dataclasses import is_dataclass, asdict
from vosk import Model, KaldiRecognizer, SetLogLevel

import json
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

import gstmicpipeline as gm

from vad_iterator import VadState

MICRO="microphone"

# configure logger
logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

modroot = Path('.') / 'models'

# enable debugging at httplib level (requests->urllib3->http.client)
# You will see the REQUEST, including HEADERS and DATA, and RESPONSE
# with HEADERS but without DATA.
# The only thing missing will be the response.body which is not logged.

# import http.client as http_client
# http_client.HTTPConnection.debuglevel = 1
#
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True


def int_or_str(text):
    """Try to convert to int, return original object if not possible."""
    try:
        return int(text)
    except ValueError:
        return text


def current_milli_time():
    """
    Return the unix time in milliseconds.

    The audio_time argument is deliberately ignored!
    """
    return round(time.time() * 1000)


def audio_milli_time(audio_time):
    """Audio time is processed audio in seconds."""
    return round(audio_time * 1000)


def open_wave_file(path, sample_rate, channels):
    """ Monitor input to .wav file, Takes path, sample rate, an no. channels.
    """
    wf = wave.open(path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    return wf


class VoskMicroServer():
    MAX_BUF_RETENTION = 40
    MIN_SPEECH_DETECTS = 3
    MIN_SILENCE_DETECTS = 30
    BUFFER_SIZE = 512

    def __init__(self, config, transcription_file=None):
        self.pid = "voskasr"
        self.topics = {}  # string to fn or (fn, qos)

        self.audio_dir = "audio/"
        self.language = "de"

        self.channels = 1
        self.usedchannel = 0
        self.sample_rate = 16000
        self.asr_sample_rate = 16000
        self.buffers_queued = 6

        self.loop: asyncio.AbstractEventLoop
        self.is_running = True
        self.audio_source = MICRO

        if self.from_micro():
            self.timestamp_fn = current_milli_time
            self.audio_queue = asyncio.Queue()
        else:
            self.timestamp_fn = audio_milli_time
            self.audio_queue = asyncio.Queue(maxsize=1)

        self.config = config
        if 'asr_sample_rate' in config:
            self.asr_sample_rate = config['asr_sample_rate']
        if 'sample_rate' in config:
            self.sample_rate = config['sample_rate']
        if 'channels' in config:
            self.channels = config['channels']
        if 'use_channel' in config:
            self.usedchannel = config['use_channel']
        if 'audio_dir' in config:
            self.audio_dir = config['audio_dir']
        if 'language' in config:
            self.language = config['language']
        if 'buffers_queued' in config:
            self.buffers_queued = config['buffers_queued']
        self.topic = self.pid + '/asrresult'
        if self.language:
            self.topic += '/' + self.language
        self.initial_prompt = ''
        self.transcription_file = transcription_file
        self.__init_mqtt_client()
        self.silence_buffer = bytearray(self.asr_sample_rate * 2)
        # initialize silero VAD model
        vad_config = config.get('vad', dict())
        self.vad_state = VadState(modroot / 'silero_vad.jit', buffered=False, **vad_config)

        self.__init_recognizer()

        # for monitoring (eventually)
        self.am = None
        self.wf = None

    def __init_recognizer(self):
        if 'vosk' not in self.config:
            logger.error('no vosk config section: minimally specify model name or path')
            sys.exit(1)
        vosk_config = self.config['vosk']
        model_path = Path(vosk_config['model_path'])
        if not model_path.is_absolute():
            model_path = modroot / 'kaldi-models' / model_path
        self.asr_model = Model(lang=self.language,
                               model_path=str(model_path.absolute()))
        self.recognizer = KaldiRecognizer(self.asr_model, self.asr_sample_rate)
        self.recognizer.SetMaxAlternatives(1)
        self.recognizer.SetWords(True)
        logger.info("Vosk model initialized")

    valid_mqtt_keys = {'host', 'port', 'keepalive', 'bind_address', 'bind_port'
                        'clean_start'}

    def __init_mqtt_client(self):
        if 'mqtt' not in self.config:
            self.config['mqtt'] = { 'host':'localhost' }
        mqtt_config = self.config['mqtt']
        for key in mqtt_config:
            if key not in self.__class__.valid_mqtt_keys:
                del(key, mqtt_config)
        self.client: mqtt.Client
        self.client = mqtt.Client(CallbackAPIVersion.VERSION2)
        if 'username' in mqtt_config and 'password' in mqtt_config:
            self.client.username_pw_set(mqtt_config['username'],
                                        mqtt_config['password'])
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.prompt_topic = self.pid + '/set_prompt'
        self.topics[self.prompt_topic] = self._on_prompt_msg

    def _on_prompt_msg(self, client, userdata, message):
        self.initial_prompt = message.payload
        logger.info(f'new prompt: {self.initial_prompt}')

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        logger.debug(f'CONNACK received with code {reason_code}')
        # subscribe to all registered topics/callbacks
        for topic in self.topics:
            qos = 0
            if topic is tuple:
                qos = topic[1]
                topic = topic[0]
            self.client.subscribe(topic, qos)

    def _on_message(self, client, userdata, message):
        logger.debug(f"Received message {str(message.payload)} on topic {message.topic} with QoS {str(message.qos)}")
        if message.topic not in self.topics:
            self.topics[message.topic] = None
            for topic in self.topics:
                if mqtt.topic_matches_sub(topic, message.topic):
                    self.topics[message.topic] = self.topics[topic]
        cb = self.topics[message.topic]
        if cb is not None:
            if cb is tuple:
                cb = cb[0]  # second is qos
            cb(client, userdata, message)
        return

    def from_micro(self):
        return self.audio_source == MICRO

    def wav_filename(self):
        return self.audio_dir + f'source-{current_milli_time():014d}.wav'

    def asrmon_filename(self, suffix):
        return self.audio_dir + f'chunk-{suffix:014}.wav'

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
        self.client.connect(**self.config['mqtt'])
        self.client.loop_start()

    def mqtt_disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def send_transcription(self, trans: dict):
        """TODO: adapt"""
        if self.client:
            self.client.publish(self.topic, json.dumps(trans, indent=None))
        if self.transcription_file:
            # massage the information in trans into the right format
            text = ""
            for segment in trans['segments']:
                text += segment['text'] + ' '
            trans.pop('info')
            trans.pop('segments')
            trans['text'] = text.strip()
            trans['source'] = self.audio_source
            self.transcription_file.writerow(trans)

    def transcribe_success(self, segment, audio_segment):
        """TODO: adapt"""
        if 'text' in segment:
            logger.info("[%.2fs -> %.2fs] %s"
                        % (segment['start'], segment['end'], segment['text']))
        self.send_transcription(segment)


    def bytes2intlist(self, audio):
        frame = np.frombuffer(audio, dtype=np.int16)
        # monitor what comes in
        # data will be self.sample_rate, mono, np.int16 ndarray
        frame = self.resample(frame, self.channels, self.sample_rate)
        # print('d', np.shape(data))
        # print('vb1', len(voice_buffers))
        return frame.tolist()

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
                data['source'] = self.audio_source
                print(data)
                # TODO: to be compatible with asrident, we would need the
                # concatenated buffers, too...
                self.transcribe_success(data, None)
                return data['end']
        return voice_start

    def send_frames(self, audio, voice_start):
        buf = np.array(audio, dtype=np.int16).tobytes()
        if self.wf:
            self.wf.writeframes(buf)
        if self.recognizer.AcceptWaveform(buf):
            return self.check_result(self.recognizer.Result(), voice_start)
        else:
            # partial result in self.recognizer.PartialResult()
            logger.debug(f'{self.recognizer.PartialResult()}')
        return voice_start

    async def microphone_loop(self):
        logger.info(f'sample_rate: {self.asr_sample_rate}')
        start_time = None
        while self.is_running:
            audio = await self.audio_queue.get()
            if self.am:
                self.am.writeframes(audio)
            state, buffer = self.vad_state.add_audio(self.bytes2intlist(audio))
            if state == "start":
                start_time = current_milli_time()
                if self.config.get('monitor_asr', False):
                    self.wf = open_wave_file(self.asrmon_filename(start_time),
                                             self.asr_sample_rate, 1)

                self.send_frames(buffer, start_time)
            elif state == "continue":
                self.send_frames(buffer, start_time)
                pass
            elif state == "end":
                self.send_frames(buffer, start_time)
                self.send_frames(self.silence_buffer, start_time)
                if self.wf:
                    self.wf.close()
                    self.wf = None
                start_time = None
        logger.info("Leaving audio_loop")


    def cb(self, inp, frames):
        self.callback(inp, frames, None, None)

    async def run_micro(self):
        self.inputfile = None
        self.loop = asyncio.get_running_loop()
        pipeline = self.config["pipeline"] if "pipeline" in self.config \
            else gm.PIPELINE
        try:
            self.device = gm.GstreamerMicroSink(callback=self.cb,
                                                pipeline_spec=pipeline)
            self.device.start()
            logger.info("Connecting to MQTT broker")
            self.mqtt_connect()
            if self.config.get('monitor_mic', False):
                with open_wave_file(self.wav_filename(),
                                    self.sample_rate, self.channels) as self.am:
                    await self.microphone_loop()
            else:
                self.am = None
                await self.microphone_loop()
        finally:
            self.device.stop()
            logger.info('Disconnecting...')
            self.mqtt_disconnect()

    def stop(self):
        self.is_running = False
        self.audio_queue.put_nowait(self.silence_buffer)
        self.device.stop()

    def read_file(self, file):
        with wave.open(file, "rb") as wf:
            self.channels = wf.getnchannels()
            self.sample_rate = wf.getframerate()
            buffer_size = int(self.sample_rate * 0.2)  # 0.2 seconds of audio
            filebuf = bytearray()
            while True:
                data = wf.readframes(buffer_size)
                if len(data) > 0:
                    filebuf.extend(data)
                else:
                    # all timestamps in milliseconds
                    now = int(time.time() * 1000)
                    buf = self.bytes2intlist(filebuf)
                    # milliseconds since sample_rate is Hz
                    duration = int(float(len(buf) * 1000) /
                                   (self.sample_rate * self.channels))
                    self.transcription_queue.put((buf, now, now + duration))
                    self.transcribe()
                    break

    def read_file_vad(self, file):
        with wave.open(file, "rb") as wf:
            self.channels = wf.getnchannels()
            self.sample_rate = wf.getframerate()
            buffer_size = 512

            # samples --> milliseconds since sample_rate is Hz
            sample2time = 1000.0 / (self.sample_rate * self.channels)
            processed_samples = 0

            active = True
            while active:
                data = wf.readframes(buffer_size)
                if len(data) == 0:
                    data = self.silence_buffer
                    active = False

                state, buffer = self.vad_state.add_audio(self.bytes2intlist(data))
                if state == "start":
                    start_time = int(processed_samples * sample2time)
                elif state == "continue":
                    pass
                elif state == "end":
                    duration = int(len(buffer) * sample2time)
                    self.transcription_queue.put((buffer, start_time,
                                                  start_time + duration))
                    self.transcribe()
                processed_samples += len(data)

    def run(self, config, files, mqtt, vad):
        if mqtt:
            print("Connecting to MQTT broker")
            self.mqtt_connect()
        # self.loop = asyncio.get_running_loop()
        # self.processing = asyncio.create_task(self.audio_loop())
        # self.processing.add_done_callback(
        #    lambda t: logger.info("processing loop finished"))
        for file in files:
            curr_audio_source = self.audio_source
            try:
                p = Path(file)
                self.audio_source = p.name
                logger.info("Processing {}".format(self.audio_source))
                self.is_running = False   # leave transcribe when queue
                p = str(p) # currently wave.open does not accept Path objects
                if vad:
                    self.read_file_vad(p)
                else:
                    self.read_file(p)
            finally:
                self.audio_source = curr_audio_source


    def stop_batch(self):
        self.is_running = False
        print('Disconnecting...')
        if self.am:
            self.am.close()
        self.mqtt_disconnect()


def process_files(server_class, config_file, files, output_dir, mqtt, vad):
    config = load_config(config_file)
    root = output_dir if output_dir else "outputs"
    if root[:-1] != os.sep:
        root += os.sep
    outdir = root
    if not os.path.exists(outdir):
        logger.info("Creating {}".format(outdir))
        os.makedirs(outdir)

    with open(outdir + 'batch.csv', 'w') as csvfile:
        fieldnames = ['source', 'start', 'end', 'text',
                      'embedid', 'speaker', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        config['audio_dir'] = outdir + os.sep
        ms = server_class(config, transcription_file=writer)
        ms.run(config, files, mqtt, vad)


def load_config(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)


def main(server_class):
    parser = argparse.ArgumentParser(
        prog='Vosk Server',
        description='Listen to microphone and transcribe input ' +
        'or analyse a set of files',
        epilog='')
    parser.add_argument("-c", "--config", metavar='config', type=str,
                        required=True, help='config file')
    parser.add_argument("-o", "--output-dir", metavar='output_dir', type=str,
                        required=False, help='output directory for chunks')
    parser.add_argument("-m", "--mqtt", action='store_true',
                        required=False,
                        help='send mqtt messages in batch processing')
    parser.add_argument("-v", "--vad", action='store_true',
                        required=False,
                        help='use VAD for file processing')
    parser.add_argument('files', metavar='files', type=str, nargs='*')
    args = parser.parse_args()
    if (args.files):
        process_files(server_class,
                      args.config, args.files, args.output_dir,
                      args.mqtt, args.vad)
    else:
        config = load_config(args.config)
        ms = server_class(config)

        logging.basicConfig(level=logging.INFO)
        try:
            asyncio.run(ms.run_micro())
        except Exception as ex:
            logger.error("Exception: " + str(ex))
            traceback.print_exc()
            ms.stop()


if __name__ == '__main__':
    # one level up for src/
    #modroot = Path(sys.argv[0]).parent.parent.absolute() / 'models'
    main(VoskMicroServer)
