# Targeted Adversarial Attacks in PyTorch
Experimenting with [Targeted Adversarial Attacks](https://en.wikipedia.org/wiki/Adversarial_machine_learning) on ImageNet models in PyTorch. 

# Model 
Attacks were performed on ResNet34. It was mainly picked as it was a good performer on ImageNet and it was small enough work with locally on an Nvdidia GTX 1070. 

# Method
Two methods are implemented:
- Untargeted: Fast Gradient Sign Attack. Implemented in ```create_FGSM```
- Targeted: Performing gradient descent on loss function between image and desired class. Implemented in ```create_adversary```

# Images

# Output
Applying a targeted attack with label 707: Pay-Phone, Pay-Station. ```create_adversary``` a learning rate of 0.1 and the images were "trained" over 1000 epochs. 
![Screenshot](https://github.com/fattorib/Torch-Adversary/blob/ImageNet/images/Out.PNG)
