# DocParseNet
DocParseNet is an innovative multi-modal deep learning architecture specifically designed for parsing and annotating document images, particularly those derived from Right of Way (ROW) agreement PDFs. It's a pioneering solution in the field of document image analysis, offering exceptional efficiency and accuracy.
![DocParseNet-Model](https://github.com/ahmad-shirazi/DocParseNet/assets/105120174/8e210a55-b605-4038-87f7-2191c06b33b5)


# Abstract:
The automated annotation of scanned documents remains a challenge, requiring a balance between the twin imperatives of computational efficiency and annotation accuracy. DocParseNet innovatively addresses this challenge by fusing advanced deep learning, active learning, and multi-modal learning to process both textual and visual data. This model transcends traditional OCR and semantic segmentation, capturing the intricate dynamics of text-image interplay and preserving the contextual nuances essential for deciphering complex document structures. Our empirical evaluations demonstrate that DocParseNet significantly outstrips conventional models, achieving mIoU scores of 49.12 on validation and 49.78 on the test set, reflecting its superior accuracy in discerning complex textual and graphical content inherent in Right-of-Way (RoW) documents. We achieve a notable accuracy improvement of approximately 58% compared to state-of-the-art baseline models and an 18% accuracy gain compared to the UNext baseline by integrating modified UNet with OCR DistilBERT embeddings. Remarkably, DocParseNet achieves these results with a modest parameter count of 2.8 million. These metrics, particularly when juxtaposed with the  0.034 TFLOPs (BS=1), highlight DocParseNet's capability to deliver high-performance document annotation while maintaining computational efficiency. Our Model reduces model parameters by approximately 1/25th, accelerating training speed by around 5 times compared to state-of-the-art baselines. The model's adaptability and scalability are thus well-suited for diverse, real-world RoW document processing applications, setting a new benchmark for automated annotation systems.


# Key Features:
Multi-modal Architecture: Combines the strengths of various neural network models to handle complex document structures.
Specialized for ROW Agreements: Tailored to parse and annotate key elements in ROW agreement documents like "State," "County," "Agreement Title," "Grantee Company," and "Grantor Company."
Efficient Training on Limited Data: Achieves high performance with a dataset of only 400 annotated ROW agreement PDFs converted to images.
Long-Range Dependency Learning: Utilizes Shifted MLP Encoding Block (Tok-MLP) to capture broader contextual information across the document.
Dataset:
The model is trained on a carefully curated dataset of 400 ROW agreement PDFs, converted into image format and annotated with precision. This dataset is split in an 8-1-1 ratio for training, validation, and testing, ensuring robust training and accurate evaluation.

Usage:
DocParseNet is ideal for researchers and practitioners working on document image analysis, especially those dealing with legal agreements. It can be adapted for various document types and formats, making it a versatile tool in the field.

Get Started:
Clone the repo and follow the setup instructions to start using DocParseNet for your document parsing needs. Comprehensive documentation is provided to help you understand and utilize the full capabilities of this system.

## 1- Data Format

Make sure to put the files in the following structure (e.g., the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

## 2- Training and Validation

1. Train the model.
```
python train.py --dataset <dataset training path> --datasetv <dataset validation path> --arch DocParsNet --name docparsnet1 --img_ext .png --mask_ext .png --lr 0.01 --epochs 1700 --early_stopping -1 --input_w 1024 --input_h 1024 --b 64
```
2. Evaluate the mdoel.
```
python val.py --name <exp name>
```

Contributions:
Contributions are welcome! If you have ideas for improvements or want to adapt the model for different document types, please feel free to fork the repository, make changes, and submit a pull request.


# Results
![IOUs](https://github.com/ahmad-shirazi/DocParseNet/assets/105120174/842d0f80-d264-48ea-aac1-8b182b94a3f3)
![IOUs-TFLOPs](https://github.com/ahmad-shirazi/DocParseNet/assets/105120174/921f66d3-d110-4d83-a204-77089c4befa1)



| Method                     | mIoU  | AT    | State | County | Grantor | Grantee | TFLOPs |
|----------------------------|-------|-------|-------|--------|---------|---------|--------|
| Segformer-B5 [xie2021segformer] | 34.81 | 37.36 | 38.72 | 44.28  | 8.81    | 44.87   | 0.39   |
| UNext [valanarasu2022unext]     | 42.04 | 36.16 | 53.81 | 52.01  | 24.48   | 43.73   | 0.06   |
| **DocParseNet**            | **49.78** | **43.06** | **65.66** | **53.90**  | **36.14**   | **50.12**   | **0.04**  |

*AT: Agreement Title. For each field, we mark the highest score in bold.*

