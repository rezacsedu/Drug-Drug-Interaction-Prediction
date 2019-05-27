## DDI prediction based on KG embeddings and Conv-LSTM network
Implementation of our paper titled "Drug-Drug Interaction Prediction Based on Knowledge Graph Embeddings and Convolutional-LSTM Network" submitted The 10th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics(ACM BCB), 2019.

In this paper, we propose a new method for predicting potential DDIs by encompassing over 12,000 drug features from DrugBank, PharmGKB, and KEGG drugs with the help of knowledge graph(KGs). 

In our pipeline, we extract feature vector representation of drugs from the KGs, using various embedding techniques such as RDF2Vec, TransE, KGloVe, SimplE, CrossE, and PyTorch-BigGraph(PBG). The embedded vectors are then used to train different prediction models.

## Requirements
* Python 3
* Keras 
* TensorFlow.

## Citation request
    @inproceedings{karim2019ddiconvlstm,
        title={Drug-Drug Interaction Prediction Based on Knowledge Graph Embeddings and Convolutional-LSTM Network},
        author={Md. Rezaul Karim, Michael Cochez, Joao Bosco Jares, Mamtaz Uddin, Stefan Decker, and Oya Beyan},
        booktitle={Proceedings of ACM BCB, ACM, New York, NY, USA, 10 pages},
        year={2019}
    }

## Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
