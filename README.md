# FixationSaccadeClassifier

## Installation
 * pull repo
 * in the root folder run <code>pip3 install .</code>

## Usage

```
from fixation_saccade_classifier import IDTFixationSaccadeClassifier

classifier = IVTFixationSaccadeClassifier()
fixations, saccades, fixation_colors, saccades_colors = classifier.fit_predict(lx, ly)
```

## Methods

 * I-DT - Dispersion-Threshold Identification
 * I-VT - Velocity-Threshold Identification
 * I-HMM - HMM Identification
 * I-AOI - Area-of-Interest Identification

source: [Identifying Fixations and Saccades in Eye-Tracking Protocols](https://www.researchgate.net/publication/220811146_Identifying_fixations_and_saccades_in_eye-tracking_protocols)

## Example of usage
see [example.ipynb](https://github.com/appuzanova/FixationSaccadeClassifier/blob/master/example/example.ipynb)
 
