---THE COMMAND TO RUN MY CODE---

python3 imagenet_finetune.py


---WHAT I HAVE DONE---

1.In the data augment section, I add a transform "random horizontal flip" using torchvision.transforms. This transform can horizontally flip the given image randomly with a given probability, which in this case I set to 0.5.

2.Change the optimizer to "Adam", which is more popular currently.

3.Run validation of the model on test set of the dataset. I run for 20 epochs and save the model with the best validation error.

And I also plot a figure of training error and validation error, which is showed in the report.
