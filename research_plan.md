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
        - ~~ICPR2014-Mitosis (https://mitos-atypia-14.grand-challenge.org/Donwload/)~~
        - NCT-CRC-HE-100k-NONNORM (https://zenodo.org/record/1214456#.YPsH8lNKheg)
    - WSI
        - BACH (see above)
        - CAMELYON16 (https://drive.google.com/drive/folders/0BzsdkU4jWx9Bb19WNndQTlUwb2M?resourcekey=0-FREBAxB4QK4bt9Zch_g5Mg)
        - CAMELYON17 (https://drive.google.com/drive/folders/0BzsdkU4jWx9BaXVHSXRJTnpLZU0?resourcekey=0-tyfGzeoOMAWlP_ogPt_4pw)
        - TUPAC16 (https://tupac.grand-challenge.org/Dataset/)
        - TCIA_Collections (https://www.cancerimagingarchive.net/histopathology-imaging-on-tcia/)
            - CTPAC (https://www.cancerimagingarchive.net/collections/) -> filter with CTPAC
            - ~~NLST (https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial)~~
            - SLN-Breast (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763339)
        - PANDA  (https://www.kaggle.com/c/prostate-cancer-grade-assessment)

    - only using WSI datasets? or using both Patch and WSI datasets? -> WSI datasets + Kather
    - tile on 20x fixed? Or mixture of 20x and 40x (i.e., max magnif. level)? -> latter
    - randomly sample 100 tiles/WSI for the WSI datasets except for PANDA, 10 tiles/WSI for PANDA (100 x 9000 + 10 x 11000 = 900,000 + 110,000 = 1,010,000)
    - tile_size: 224

1. Metadata for datasets
    - CTPAC/CCRCC: 782 svs, 20x
    - CTPAC/GBM: 510 svs, 20x
    - CTPAC/COAD: 372 svs, 40x
    - CTPAC/LSCC: 1081 svs, 20x
    - CTPAC/BRCA: 653 svs, 20x and 40x
    - CTPAC/OV: 221 svs, 40x
    - CTPAC/SAR: 307 svs, 20x (some are about 30x)
    - CTPAC/CM: 411 svs, 20x (some are about 30x)
    - CTPAC/LUAD: 1137 svs, 20x (some are 40x and about 30x)
    - CTPAC/PDA: 557 svs, 20x
    - CTPAC/UCEC: 883 svs, 20x (some are 40x and about 30x)
    - CTPAC/HNSCC: 390 svs, 20x
    - CAMELYON16: 159+111+129 tif, 40x
    - CAMELYON17: 1000 tif, 40x
    - SLN-Breast: 130 svs, 20x
    - PANDA: 10616 tiff, 20x (biopsy)
    - BACH: 40 svs, 20x
    - TUPAC16: 821 svs, 40x (some are 20x)
    - total number of WSI: 20310

    - NCT-CRC-HE-100k-NONNORM: 100k tiles, 20x, 224x224

1. Downstream tasks (first tile-level -> next wsi-level)
    - tile-level classification
        - WILDS (https://worksheets.codalab.org/worksheets/0xb44731cc8e8a4265a20146c3887b6b90)
        - NCT-CRC-HE-100K/CRC-VAL-HE-7K (see above)
    - wsi-level classification
        - CAMELYON16/17 (see above)
        - PANDA (see above)

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

