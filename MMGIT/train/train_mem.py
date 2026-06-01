import sys
sys.path.append(".")
sys.path.append("./utils")
sys.path.append("..")
from graphllava_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from train import _train

if __name__ == "__main__":
    _train()