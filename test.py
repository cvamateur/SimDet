from modeling import SingleStageDetector
from config import Configs as cfg

ckpt = r"ckpt/lst.pt"

detector = SingleStageDetector(cfg)
