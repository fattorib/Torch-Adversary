# Targeted Adversarial Attacks in PyTorch
Experimenting with [Targeted Adversarial Attacks](https://en.wikipedia.org/wiki/Adversarial_machine_learning) on ImageNet models in PyTorch. 

# Model 
Attacks were performed on ResNet34. It was mainly picked as it was a good performer on ImageNet and it was small enough work with locally on an Nvdidia GTX 1070. 

# Method
Two methods are implemented:
- Untargeted: Fast Gradient Sign Attack. Implemented in ```create_FGSM```
- Targeted: Performing gradient descent on loss function between image and desired class. Implemented in ```create_adversary```

# Images
The 5 test images were all found from the ImageNet website and downloaded from their respective links. All images were resized and cropped prior to being fed into the model for better visualization.

<p float="left">
  <img src="https://github.com/fattorib/Torch-Adversary/blob/main/images/Cat.jpg" width="250" />
  <img src="https://github.com/fattorib/Torch-Adversary/blob/main/images/Fungus.jpg" width="250" /> 
  <img src="https://github.com/fattorib/Torch-Adversary/blob/main/images/Rock.jpg" width="250" />
  <img src="https://github.com/fattorib/Torch-Adversary/blob/main/images/Beach.jpg" width="250" />
  <img src="https://github.com/fattorib/Torch-Adversary/blob/main/images/bird.JPEG" width="250" />
</p>


# Output
Applying a targeted attack with label 707: Pay-Phone, Pay-Station. ```create_adversary``` was used with a learning rate of 0.1 and 1000 epochs. 

![Screenshot](https://github.com/fattorib/Torch-Adversary/blob/ImageNet/images/Out.PNG)
