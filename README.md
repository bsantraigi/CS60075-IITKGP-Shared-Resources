# CS60075(Spring 2021) Shared Tasks

## Tutorials given in Course

1. [**Pytorch Tutorial**](https://github.com/nayakt/Neural-POS-Tagger)
2. [**Notebook for Word-embeddings analogy:**](https://colab.research.google.com/drive/1DxC5AnIFuu9Maan_D23yoE6sW4Ez2Uhw?usp=sharing#scrollTo=yr64uvq1E3E-)
3. [**Notebook for Bias in word-embeddings:**](https://colab.research.google.com/drive/11PyFecG4gaWBrP1FB6nHUROktwUSUCbI?usp=sharing#scrollTo=SkOPpPRMEp1M)
4. 

## Important Points to Consider:

* **Ideal team size**: 4-5 members
* [SemEval 2021 **Task 1**](https://sites.google.com/view/lcpsharedtask2021): Cannot be taken by teams of size 6.
* [SemEval 2021 **Task 8**](https://competitions.codalab.org/competitions/25770): Subtasks 1 and 3 (if team size <= 5), Subtasks 1, 3 and 5 (if team size is 6)
* [SemEval 2021 **Task 11**](https://ncg-task.github.io/): Subtasks 1 and 2 (if team size <= 5), Subtasks 1, 2 and 3 (if team size is 6)

## Deadlines

> TBA (Tentative Deadline: 7th April)

## Deliverables

1. A project report of upto 4-page with following sections in the same order. Use either latex or docx format from ACL.
    * Title
    * Subtask ID + Group Details (Names, Roll Numbers, Group Name)
    * Introduction
    * Task Details
    * Approach / Model Architectures
    * Experiments
    * Results / Discussions
1. A public GitHub repo for the submitted code. *Codes will be checked for plagiarism.*
    * Repo Name Format: `CS60075-Team-X-Task-Y`
    * Example: `CS60075-Team-3-Task-11`
1. A brief presentation video of upto 10 mins explaining the task, approach and results.

## Useful Resources for the Tasks

### Where to look? (Keywords, Concepts, Packages)

If you are unfamiliar with the most common concepts of NLP and ML, you can start searching for the following concepts. These are list of keywords without any specific structure.

1. Word Embeddings: Word2Vec, Glove, Gensim, Spacy
2. Contextual Word Embeddings (works better than simple word embeddings): BERT, ELMo, Huggingface, AllenNLP
3. Metric Calculation: scikit-learn package
1. Classifiers: tf-idf classifier, bag-of-words, Naive Bayes, Classification using BERT
1. Text Generation: Sequence-to-Sequence, GPT-2, BART
1. Best tutorial websites: TowardsDataScience, Medium, Huggingface Documentation

### Free coding platforms for ML

1. Kaggle Notebooks
1. Google Colab

### Task 11 (Papers, Tutorials, etc.)

* Information Extraction: OpenIE
   * https://demo.allennlp.org/open-information-extraction/open-information-extraction
   * https://github.com/dair-iitd/OpenIE-standalone
* Entity Extraction: 
   * https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
   * https://www.nltk.org/book/ch07.html

### General Resource

#### Basic Materials

**ML for NLP Tutorials:**

1. [Dive into Deep Learning — Dive into Deep Learning 0.8.0 documentation](https://d2l.ai/)
2. [fast.ai Code-First Intro to Natural Language Processing](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)
4. [Fine-tuning a BERT/Pretrained-Model in Pytorch](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)
5. [Nice IMDB-Sentiment Analysis Tutorials](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/)
6. [English to Hindi Transliteration using Seq2Seq Model](https://bsantraigi.github.io/tutorial/2019/08/31/english-to-hindi-transliteration-using-seq2seq-model.html)
7. [Deep Sentiment Analysis](https://bsantraigi.github.io/tutorial/2019/08/31/deep-sentiment-analysis.html)
8. [Text Classification using Naive Bayes Method](https://bsantraigi.github.io/2019/07/13/text-classification-using-naive-bayes-method.html)
9. [Training a Language Model with a Xtra-Small Transformer (Transformer-XS)](https://bsantraigi.github.io/tutorial/2019/07/08/training-xtra-small-transformer-language-model.html)

#### Great Seq2Seq Tutorials

1. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
 It talks about the paper [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762). This presents a **Sequence to Sequence** architecture for **Neural Machine Translation**
2. **Chatbot Tutorial:** [**Chatbot Tutorial**](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
1. [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
2. [Transformer — PyTorch master documentation](https://pytorch.org/docs/master/generated/torch.nn.Transformer.html)
3. [huggingface/transformers: 🤗Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.](https://github.com/huggingface/transformers)

#### Other Helpful Blog Posts:

- [How do Transformers Work in NLP? A Guide to the Latest State-of-the-Art Models](https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/)
- [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time.](http://jalammar.github.io/illustrated-transformer/)
- [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591)
- [Transformers from scratch](http://www.peterbloem.nl/blog/transformers)
- [What is a Transformer? - Inside Machine learning](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- FastAI tutorial on Transformers (with code): [The Transformer for language translation (NLP video 18)](https://www.youtube.com/watch?v=KzfyftiH7R8)
- [https://github.com/fawazsammani/chatbot-transformer/blob/master](https://github.com/fawazsammani/chatbot-transformer/blob/master/models.py)PYTORCH CHATBOT TRANSFORMER IMPLEMENTATION
- Good read for various decoding techniques: [https://huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)
