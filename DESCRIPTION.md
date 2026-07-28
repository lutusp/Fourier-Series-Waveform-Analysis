# Square and Sawtooth Waveform Video

This project should be written Python and generate videos showing the gradual evolution of two waveforms -- a square wave and a sawtooth wave. As the video evolves, the program will increase the number of terms in the series, render that change to the evolving video, and use the sound the new result creates for the sound track of the video.

Altogether four videos should be created, all with sound tracks. Two will show the evolution of the square waveform in the time and frqeuency domains, the other the sawtooth waveform. Initially make the videos ten seconds in length and show the evolution of the waveforms over 128 terms of the infinite series. Each video will display two full cycles of the waveform and have a sound track with frequency of 110 Hz, with a waveform generated directly by the wave shape of the wave-generating functions.

The videos will have a duration of ten seconds, 300 frames, and begin with n = 1 and end with n = 128 of the terms required by the waveforms. The audio will be modulated by the generated waveform and the video frame will render two full cycles of the current waveforms using matplotlib.

In summary, four videos, each ten seconds (300 frames), with a matplotlib render of an evolving waveform and spectrum, each with an accompanying sound track that uses the evolving waveform to modify a 110 Hz signal.

To start things off I have created a boilerplate Python script named "square_triangle_wave_generator.py" that you should fill out with your own code.

The finished videos should be placed in the "renders" subdirectory.
