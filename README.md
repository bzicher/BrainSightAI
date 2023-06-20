# BrainSightAI
AI classifier that predicts brain states from physiological recordings

## Overview
This project focuses on using AI algorithms to predict brain state from electroencephalography (EEG) or peripheral electromyography (EMG) recordings.

### Dataset
Electroencephalography (EEG) and electromyography (EMG) data was recorded from 9 participants. Seven participants have peripheral recordings from the tibialis anterior using 4 high-density surface EMG grids, while two participants have recordings using intramuscular multi-electrode grid data as well. 
The EMG signals were decomposed into constituent spike trains using previously validated methods.
The main experiment was divided into three identical parts, each made up of 35 trials. Each trial was 8s long, during which participants had to react to auditory cues. Initially, subjects had to maintain a 10% maximum voluntary contraction (MVC) force of ankle dorsiflexion, after which they heard a preparatory cue and 1s later a GO/NO-GO cue (50%/50%).
In 'GO' trials, the participant did a ballistic movement, while in 'NO-GO' trials, the constant force of 10% MVC had to be maintained. Three states can be extracted from this data, baseline (-2s to -1s relative to 'NO-GO' cue), preparatory (-1s to 0s) and preparation cancellation (0s to 1s).

To use this code:
1. Clone the repo
2. Create a new environment (optional)
3. Install the package using '*pip install -e .*'

## Notes on installation

#### Step 1
Create new (conda) env with python>=3.8

#### Step 2
install pytorch with cuda
*pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118*

(this installs numpy automatically)

#### Step 3
*pip install -r requirements.txt*

#### Step 4
install package
*pip install -e .*