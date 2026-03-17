import logging
from pathlib import Path
import torch
import numpy as np

logger = logging.getLogger(__file__)


def init_jit_model(model_path: Path, device='cpu'):
    if isinstance(device, str):
        device = torch.device(device)
    logger.info(f"Init VAD model on {device}")
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


class VADIterator:
    def __init__(self,
                 model_path: Path,
                 device = 'cpu',
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = init_jit_model(model_path, device)
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()
        #print(speech_prob)
        #sys.stdout.flush()

        # temp_end is there is to skip small intervals of silence without
        # signalling that the speech signal has ended!
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None


class VadState:
    #MIN_SPEECH_DETECTS = 3
    #MIN_SILENCE_DETECTS = 30
    BUFFER_SIZE = 512

    def __init__(self, model_path, buffers_queued = 6, buffered = True,
                 **kwargs):
        self.framesqueued = VadState.BUFFER_SIZE * buffers_queued
        self.vad_iterator = VADIterator(model_path, **kwargs)
        self.is_voice = False
        self.voice_buffers = [0] * self.framesqueued
        self.buffer = [] if buffered else None

    def _is_voice(self) -> bool:
        return self.is_voice

    def add_audio(self, audio_as_intlist):
        self.voice_buffers.extend(audio_as_intlist)
        if len(self.voice_buffers) < VadState.BUFFER_SIZE + self.framesqueued:
            return None, None

        # only check the last vad_state.BUFFER_SIZE samples for the VAD
        # we queue additional data to be able to provide data from the past
        ichunk = self.voice_buffers[self.framesqueued:
                                    self.framesqueued + VadState.BUFFER_SIZE]
        vadbuf = np.array(ichunk, dtype=np.int16) / 32768
        speech_dict = self.vad_iterator(vadbuf, return_seconds=True)
        #print(f'{speech_dict}')
        transcription_buffer = None
        voice_state = None
        if speech_dict:
            if "start" in speech_dict:
                # arg ist processed time in milliseconds
                logger.debug('<')
                voice_state = "start"
                self.is_voice = True
                # add queued buffers to the outbuffer: old + ichunk
                ichunk = self.voice_buffers[:self.framesqueued
                                            + VadState.BUFFER_SIZE]
                if isinstance(self.buffer, list):
                    self.buffer.extend(ichunk)
                else:
                    transcription_buffer = ichunk
            elif "end" in speech_dict:
                if not self._is_voice():
                    logger.warning('VAD end ignored')
                    return "no_speech", None
                logger.debug('>')
                voice_state = "end"
                self.is_voice = False
                if isinstance(self.buffer, list):
                    self.buffer.extend(ichunk)
                    transcription_buffer = self.buffer
                    self.buffer.clear()
                else:
                    transcription_buffer = ichunk
        elif self._is_voice():
            voice_state = "continue"
            if isinstance(self.buffer, list):
                self.buffer.extend(ichunk)
            else:
                transcription_buffer = ichunk
        else:
            voice_state = "no_speech"
        # in any case, we processed BUFFER_SIZE samples
        self.voice_buffers = self.voice_buffers[VadState.BUFFER_SIZE:]
        # samples = audio_buffer length / 2 (int16: 2 bytes, mono)
        return voice_state, transcription_buffer
