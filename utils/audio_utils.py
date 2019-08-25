import wave, sys, pyaudio
import matplotlib.pyplot as plt

def play_wav(wf):
    paud = pyaudio.PyAudio()
    chunk = 1024
    stream = paud.open(format = paud.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    data = wf.readframes(chunk)
    while data != '':
        stream.write(data)
        data = wf.readframes(chunk)

def graph_wav(wave_data, wav_name, wav_num):
    fig, axs = plt.subplots(1, 1, figsize=(10, 2))
    axs.plot(wave_data);
    axs.set_title('Audio Signal for image {}: {}'.format(wav_num, wav_name))
    #axs[1].plot(encoding[0]);
    #axs[1].set_title('NSynth Encoding')
    #plt.close(fig)