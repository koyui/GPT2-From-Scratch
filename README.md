<h1 align="center">Building GPT2 From Scratch</h1>

Build a lightweight GPT2 from scratch, train it on movie reviews, and complete movie reviews based on the prompt words.

### For Windows
I have only tested the code on ``Windows 11``.

All dependencies:
> python=3.10      
> CUDA=11.7     
> torch=2.0.1       
> numpy=1.26.4     
> bert-score=0.3.2         
> transformers=2.2.0           
> tqdm  
> regex      
> pandas        
> natsort       
> omegaconf         
> matplotlib
```
conda create -n koyui_gpt2 python=3.10 -y
conda activate koyui_gpt2
pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Tokenizer
I have tried to implement a bbpe tokenizer, another tokenzier is from pretrained bert-base-chinese `https://hf-mirror.com/google-bert/bert-base-chinese/tree/main`

### Project Structure
```
ckpts/
|--gpt2_time_stamp/         # Checkpoints here
configs/
|--config.yaml              # Project configs
data/
|--dataset.py               # GPT2 Dataset
|--utils.py                 # Collate Fn
deps/
|--bert-base-chinese/
|--reviews.csv              # Training Data
|--merge/map_rules.json     # Tokenizer Rules
models/
|--model.py                 # GPT2 Models
test/
|--input.txt
|--result.txt
utils/
|--bbpe.py                  # BBPE Class
|--custom_bert_tokenizer.py # BertTokenizer Wrapper
|--visualize_bbpe_vocab.py  # Visualization Tools
README.md
requirements.txt/
trainer.py                  # MAIN
```

### Run
Run `python trainer.py` to train/test/evaluate your GPT2 model.

### Device
My device (RTX 3060 Laptop GPU with 6GB VRAM and i7-12700H CPU) is capable of training.

### Acknowledgments

Thanks to the following references that I refer to and benefited from:
- [Tokenizer_gpt2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py): Implement detail(regex) on bbpe tokenizer.
- [kaggle: what is BBPE](https://www.kaggle.com/code/binfeng2021/what-is-bbpe-tokenizer-behind-llms): Implementation on BBPE tokenizer.
- [Let's reproduce GPT-2(124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3870s): Implementation on GPT2

### TODO
- ✓ Implement BBPE and build vocabulary on these reviews. (Tested)
- ✓ Build GPT2 Training Framework.
- ✓ Validation at the end of each training loop.
- ✓ Conduct evaluations
- ✓ Try to train a model with proper hyperparameters and get some proper test results