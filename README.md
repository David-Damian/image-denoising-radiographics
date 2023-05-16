# Image denoising for radiographics.
We implemented a generative neural network to **enhance** the image quality of radiographs that are poorly taken, wrinkled, or have other artifacts. To simulate these artifacts, we are incorporating Poisson noise, Gaussian noise, and other similar techniques into the images.

Also we use advanced tools of AWS cloud for training the model.

# Data
We use the [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/) which consist of 40k multi-view radiographic images of upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist. 
