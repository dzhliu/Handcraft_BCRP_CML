# Handcraft_BCRP_CML

Handcraft n backdoor critical routine paths with random mask

1. pre-train with square pattern (intensity:1) 

2. find backdoor routing (based on weight/activation value, top-k (k=0.05) in each layer) using backdoor samples (using training data)

3. manipulate weights in backdoor routing (1.05x)

4. using the mask with a $\alpha$ intensity (test data)

### Step 1: Train clean pre-train model 
Train clean model:
    python main.py 

### Step 2: Train backdoor model
Train backdoor model:
	python train_attack.py

### Step 3: Handcraft BCRP attack
HBCRP attack:
    python HBCRP_attack_activation.py
### Todo:
square pattern
CIFAR-10 VGG-11 ResNet