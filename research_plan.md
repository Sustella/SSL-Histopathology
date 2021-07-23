# Research Plan

## Purposes
1. SSL with vs without STRAP (https://github.com/rikiyay/style-transfer-for-digital-pathology)
1. ViT vs ResNet(18 or 50)
    - for ViT, we may want to use MoBy SWIN, MoBy/DINO DeiT
1. skip comparison for aggregation for now... (Attention-MIL, CLAM, Dual-Stream MIL, Transformer)

## Datasets
1. SSL pretrain datasets
    - need to be diverse rather than big according to this [paper](https://arxiv.org/abs/2011.13971)
    - TCGA currently unavailable
    - Patch
        - BACH 400patch/30wsi (https://zenodo.org/record/3632035#.YPsFwVNKheg)
        - ~~BreaKHis needRequest (https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)~~
        - ~~BreastPathQ MayNotAvailable? (https://breastpathq.grand-challenge.org/)~~
        - ICPR2014-Mitosis (https://mitos-atypia-14.grand-challenge.org/Donwload/)
        - NCT-CRC-HE-100k-NONNORM (https://zenodo.org/record/1214456#.YPsH8lNKheg)
    - WSI
        - BACH (see above)
        - CAMELYON16 (https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M?resourcekey=0-FREBAxB4QK4bt9Zch_g5Mg)
        - CAMELYON17 (https://drive.google.com/drive/folders/0BzsdkU4jWx9BaXVHSXRJTnpLZU0?resourcekey=0-tyfGzeoOMAWlP_ogPt_4pw)
        - TUPAC16 (https://tupac.grand-challenge.org/Dataset/)
        - CTPAC (https://wiki.cancerimagingarchive.net/x/mIREAQ)
        - TCIA_Others (https://www.cancerimagingarchive.net/histopathology-imaging-on-tcia/)
            - such as
                - NLST (https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial)
                - SLN-Breast (https://wiki.cancerimagingarchive.net/display/Public/Breast+Metastases+to+Axillary+Lymph+Nodes)
1. Downstream tasks (first tile-level -> next wsi-level)
    - tile-level classification
        - WILDS (> PatchCamelyon)
        - NCT-CRC-HE-100K/CRC-VAL-HE-7K
    - wsi-level classification
        - CAMELYON16/17
        - PANDA

## Implementation
1. SSL
    - DINO ViT: https://github.com/facebookresearch/dino
    - MoBy SWIN/DeiT & DINO DeiT: https://github.com/SwinTransformer/Transformer-SSL
    - SimCLR: https://github.com/binli123/dsmil-wsi
1. Aggregation
    - Attention MIL: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    - Dual-Stream MIL: https://github.com/binli123/dsmil-wsi
    - CLAM: https://github.com/mahmoodlab/CLAM
    - Transformer: need implementation (refer to Tara's)

