# Find-Best-Learning-Rate

This repo aims to find the best learning rate for my deep learning research.

### Usage

Download the full repo and just run:

```
$ python3 template.py
```

It is an example for MNIST data, the lr\_schedule and Loss changes can be found in the two jpg files.

Lr schedule:

![lr_schedule](https://github.com/Ecohnoch/Find-Best-Learning-Rate/blob/master/lr_schedule.jpg)

MNIST Loss changes:

![output](https://github.com/Ecohnoch/Find-Best-Learning-Rate/blob/master/output.jpg)



### Face Recognition

Src Repo: [https://github.com/Ecohnoch/Tiny-Face-Recognition](https://github.com/Ecohnoch/Tiny-Face-Recognition)

Set lr schedule:

![lr_schedule](https://github.com/Ecohnoch/Find-Best-Learning-Rate/blob/master/Face_Recognition/lr_schedule.jpg)

Loss changes:

![output](https://github.com/Ecohnoch/Find-Best-Learning-Rate/blob/master/Face_Recognition/output.jpg)

So we can find 1e-4 is the best original learning rate for my face recognition.

To be continued.

# Reference

```
 Smith L N . Cyclical Learning Rates for Training Neural Networks[J]. Computer Science, 2015:464-472.
```