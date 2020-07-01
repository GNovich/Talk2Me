import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
from io import BytesIO
import sounddevice as sd
import numpy as np
import librosa
import argparse
import speech_recognition as sr
import pickle
from scipy import signal
import matplotlib.pylab as plt
plt.switch_backend('TkAgg')

class Recorder:
    """
        the recorde class is incharge of fetching mic interactions
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic_id = sr.Microphone.list_microphone_names().index('Built-in Microphone')
        self.mic = sr.Microphone(device_index=self.mic_id)

    def recognize_speech_from_mic(self, adjust=False, timeout=None, phrase_time_limit=None):
        """Transcribe speech from recorded from `microphone`.

        Returns a dictionary with three keys:
        "success": a boolean indicating whether or not the API request was
                   successful
        "error":   `None` if no error occured, otherwise a string containing
                   an error message if the API could not be reached or
                   speech was unrecognizable
        "transcription": `None` if speech could not be transcribed,
                   otherwise a string containing the transcribed text
        """
        # check that recognizer and microphone arguments are appropriate type
        if not isinstance(self.recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")

        if not isinstance(self.mic, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")

        # adjust the recognizer sensitivity to ambient noise and record audio
        # from the microphone
        with self.mic as source:
            voice_not_captured = True
            while voice_not_captured:
                try:
                    if adjust:
                        print('calibrating...')
                        self.recognizer.adjust_for_ambient_noise(source, duration=3)
                        print('go:')
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    voice_not_captured = False
                except Exception as e:
                    print("Didn't get that! Let's try again:\n")

        # set up the response object
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        # try recognizing the speech in the recording
        # if a RequestError or UnknownValueError exception is caught,
        #     update the response object accordingly
        try:
            response["transcription"] = self.recognizer.recognize_google(audio)
            response["audio"] = audio
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"

        return response


class VoiceProducer:
    """
        The VoiceProducer is initiated by a wav sample
        after hence it is given text by 'say' or 'get_wavs' and provides wav back
        it can also play the wavs with 'play_wav'
    """
    def __init__(self, init_wav_fpath, enc_model_fpath, syn_model_dir, voc_model_fpath):
        self.init_sentences = {
            "let's play": "let us play repeat after me! Use around 10 words per sentence.",
            #"let's play 2": "If you want to re-capture your voice, say only: record again",
        }
        self.embbeding = None
        self.sampling_rate = None
        self.synthesizer = Synthesizer(syn_model_dir.joinpath("taco_pretrained"))
        encoder.load_model(enc_model_fpath)
        vocoder.load_model(voc_model_fpath)
        self.voice_capture(init_wav_fpath)
        self.pre_made_sentences = dict(zip(self.init_sentences.keys(),
            self.get_waves(list(self.init_sentences.values()))))
        self.wav_stack = []
        self.wav_stack_size = 0

    def save_embbeding(self):
        pickle.dump(open('curr_emb.pkl', 'wb'), self.embbeding)

    def load_embbeding(self, path='curr_emb.pkl'):
        self.embbeding = pickle.load(open(path, 'rb'))

    def preprocess_wav(self, init_wav_fpath):
        original_wav, self.sampling_rate = librosa.load(init_wav_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, self.sampling_rate)
        return preprocessed_wav

    def voice_capture(self, init_wav_fpath):
        # There are many functions and parameters that the speaker encoder interfaces.
        self.embbeding = encoder.embed_utterance(self.preprocess_wav(init_wav_fpath))
        print("Created the embedding!")

    def say(self, text:str):
        wav = self.get_waves([text])[0]
        self.play_wav(wav)

    def get_waves(self, texts:list):
        specs = self.synthesizer.synthesize_spectrograms(texts, [self.embbeding])

        ## Generating the waveforms
        # Remember, the longer the spectrogram, the more time-efficient the vocoder.
        generated_wav = [vocoder.infer_waveform(spec) for spec in specs]

        ## Post-generation
        # a bug with sounddevice makes the audio cut one second earlier, so we pad it.
        res_wavs = [np.pad(wav, (0, self.synthesizer.sample_rate), mode="constant") for wav in generated_wav]
        return res_wavs

    def update_embbeding(self, wav):
        if self.embbeding is None:
            return
        wav = self.preprocess_wav(BytesIO(wav.get_wav_data()))
        pad = self.sampling_rate // 8
        self.wav_stack.append(np.pad(wav, (pad, pad)))
        self.wav_stack_size += librosa.get_duration(wav)
        if self.wav_stack_size > 10:
            long_wav = np.concatenate(self.wav_stack)
            new_embbeding = encoder.embed_utterance(long_wav)
            self.embbeding = np.mean([new_embbeding, self.embbeding], 0)
            self.wav_stack = []
            self.wav_stack_size = 0

    def smooth(self, wav, N=1, Wn=(0.01, 0.9)):
        # N Filter order
        # Wn Cutoff frequency
        B, A = signal.butter(N, Wn, btype='bandpass', output='ba')
        return signal.filtfilt(B, A, wav)

    def play_wav(self, wav, smoothed=False):
        # Play the audio (non-blocking)
        sd.stop()
        if smoothed:
            wav = self.smooth(wav)
        sd.play(wav, self.synthesizer.sample_rate, blocking=True)


# TODO conversational class - the chatbot
# https://github.com/microsoft/DialoGPT
# https://github.com/huggingface/transfer-learning-conv-ai # shit for brains

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-init", "--init_fpath", type=str,
                        default="saved_audio/init_audio_file.wav",
                        help="Path to a saved init")
    parser.add_argument("-save_init", "--save_init_Wav", type=int,
                        default=1)
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    args = parser.parse_args()

    print('')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", end='\n')
    print("Interactive mic generation loop", end='\n\n')

    # Get the reference audio filepath
    print("Setup", end='\n')
    recorder = Recorder()
    in_wav_fpath = args.init_fpath
    if not os.path.exists(in_wav_fpath):
        print("Reference voice: talk clearly for 20 seconds about anything you'd like. "
              "better yet, read a paragraph from a book!")
        init_wav = recorder.recognize_speech_from_mic(adjust=True, phrase_time_limit=20)
        in_wav_fpath = "saved_audio/init_audio_file.wav"
        with open(in_wav_fpath, "wb") as file:
            file.write(init_wav["audio"].get_wav_data())
        print('got it, you can stop now...')
    voicer = VoiceProducer(Path(in_wav_fpath), args.enc_model_fpath, args.syn_model_dir, args.voc_model_fpath)
    if not args.save_init_Wav:
        os.remove(in_wav_fpath)

    # now is the "conversation" part!
    # currently it's just a "reapeat after me" game
    print(voicer.init_sentences["let's play"])
    voicer.play_wav(voicer.pre_made_sentences["let's play"])
    print('')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", end='\n')

    while True:
        print("talk to me!")
        res = recorder.recognize_speech_from_mic(phrase_time_limit=10)
        print("You said: ", res['transcription'])
        voicer.say(res['transcription'])
        # voicer.update_embbeding(res["audio"])
        print('')



