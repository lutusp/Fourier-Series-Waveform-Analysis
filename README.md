# Fourier Series Waveform Analysis & Video Demonstration

![Square Wave Exammple](./images/Screenshot%20from%20square_wave_time.mp4.png)


This project generates tutorial videos of square and sawtooth waves in both time and frequency domains, complete with sound. Its purpose is to familiarize students with:

* The relationship between time-domain and frequency-domain waveform representations.
* The experience of seeing, as well as hearing, an increase in the number of generated Fourier terms.
* Some math background on the above topics.
* How to generate tutorial videos with sound in Python.

The core algorithm relies on a Fourier expansion to generate square-wave and sawtooth-wave series, with varying numbers of terms, to show how the number of series terms affects the generated waveform. The accompanying audio track shows how the number of Fourier terms changes the perceived audio spectrum.

The program "square_sawtooth_wave_generator.py" created four videos when run. Just take these steps:

* `$ pip install -r requirements.txt`
* `$ sudo apt install ffmpeg # usually not necessary`

Then, to generate the four tutorial videos and place them in the directory "renders":
* `python3 ./square_sawtooth_wave_generator.py`

The Fourier series expressions for the square and sawtooth waveforms are [Here](./Equations.pdf).

Until now I've only ever seen an animation that increased the number of Fourier terms as a visual experience. These videos add the dimension of sound, and the sound of increasing higher harmonics is not to be missed.

* Installation:

Download a release ZIP file.
Move the unpacked contents into a dedicated directory.
Move into that directory.
Take thee steps:

`$ pip install -r requirements.txt`
`$ sudo apt install ffmpeg # not usually necessary`

Then, to generate the videos:

`$ python3 ./square_sawtooth_wave_generator.py`

After the program runs, tne directory "renders" will contain four videos, each with sound, of square and sawtooth wave generation with a progressively increasing number of Fourier terms.

* Vibe coding

As usual lately, I created these videos with the help of Claude Code and a particularly effective locally-installed LLM: unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M , installed on a system with an RTX 4090.
Prior to recent times I would spend days creating videos like this. Now, maybe four hours from idea to finished video.
I have included Claude Code's internal data, so those who want to pick up this project with minimal effort should be able to tell CC to read the available data before starting work.
