# DocParseNet
DocParseNet is an innovative multi-modal deep learning architecture specifically designed for parsing and annotating document images, particularly those derived from Right of Way (ROW) agreement PDFs. It's a pioneering solution in the field of document image analysis, offering exceptional efficiency and accuracy.

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

Contributions:
Contributions are welcome! If you have ideas for improvements or want to adapt the model for different document types, please feel free to fork the repository, make changes, and submit a pull request.
