<h1 align="center">Building GPT2 From Scratch</h1>

Build a lightweight GPT2 from scratch, train it on movie reviews, and complete movie reviews based on the prompt words.

### For Windows
```
conda create -n koyui_gpt2 python=3.10 -y
conda activate koyui_gpt2
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirments.txt
```

### Device
My device (RTX 3060 Laptop GPU with 6GB VRAM and i7-12700H CPU) is capable of training.

### TODO
- ✓ Implement BBPE and build vocabulary on these reviews. (Tested)
- ✓ Build GPT2 Training Framework.
- ✓ Validation at the end of each training loop.
- [] Try to train a model with proper hyperparameters and get a good test result
- [] Conduct evaluations