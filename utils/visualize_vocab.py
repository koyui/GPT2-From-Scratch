from utils.bbpe import BBPE
import pandas as pd

bbpe = BBPE(2048)
bbpe.from_file('./deps/merge_rule_2048.json')
for i in range(256, 2048):
    print(bbpe.vocab[i].decode('utf-8', errors='ignore'))