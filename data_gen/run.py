from utils.commons.hparams import hparams, set_hparams
import importlib

# ii=0

def binarize():
    binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    print("| Binarizer: ", binarizer_cls)
    binarizer_cls().process()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648515174
    set_hparams()
    binarize()
