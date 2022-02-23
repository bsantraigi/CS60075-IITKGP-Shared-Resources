# CS60075 Shared Resources

## Deadlines

> 10th March

### Deliverables:
Delivery of baseline model results, and next steps to be taken.
Submission:
1. A 2 page proposal (2 page of content, extra references) which details the following. Use either latex or docx format from ACL.
    * Templates: Docx and Latex templates can be downloaded here. You can edit using latex template on overleaf also.
    * Title
    * Subtask ID + Group Details (Names, Roll Numbers, Group Name)
    * Individual Contributions of Students (Must be explicitly mentioned for each student)
    * Task Description
    * Baseline Approaches
    * Next steps
    * Results of baselines
    * References: Upto 1 page of references can be added. You should cite to any model/library that you are using from any existing paper.

> 20th March

### Deliverables:
Discussion of steps taken since last deliverable. 1 page ad-on results from previous steps. Discussion of progress.

> 6th April

### Deliverables:
1. A 3-5 page report (NO MORE THAN 5 PAGES ALLOWED, including References) with following sections in the same order. Use either latex or docx format from ACL.
Templates: Docx and Latex templates can be downloaded here. You can edit using latex template on overleaf also.
    * Title
    * Subtask ID + Group Details (Names, Roll Numbers, Group Name)
    * Group ID
    * Codalab Team ID, if different from Group ID
    * Individual Contributions of Students (Must be explicitly mentioned for each student)
    * Task Description
    * Approach / Model Architectures
    * Metrics used
    * Experiments
    * Results / Discussions
    * Difficulty faced / Path not taken
    * References: Upto 1 page of references can be added. You should cite to any model/library that you are using from any existing paper.
    * Screenshots of codalab submission page (Participate Tab -> Submit-View Results) or the leaderboard submission page. Include all phases for which you have submitted (dev/eval + test). Account name should be visible on the top right.

## Task Description, Baseline model and where to start
### Multilingual Question Answering (MQA)
https://github.com/facebookresearch/MLQA 
https://github.com/google-research-datasets/tydiqa
Task is : Giving an answer from a passage based on the question.
* During training you can use more than one language data. 
* Use Multingual pre-trained language model
* Members: 3-4 members

Students are requested to implement the base models using [mBERT](https://huggingface.co/bert-base-multilingual-cased) and [XLM](https://huggingface.co/xlm-roberta-base) models of the [tydi-QA](https://github.com/google-research-datasets/tydiqa) gold passage dataset. Please find the below points to train the multilingual model. You can use this [code](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering).
1. Train your model using the Tydi-QA gold passage train dataset(which contains 9 languages) and evaluate the Tydia-QA dataset dev data (only Hindi and Bengali)
2. Extract English, Hindi, and Bengali from the Tydi-QA gold passage train dataset to train the base models, and evaluate the Tydia-QA dataset dev data (only Hindi and Bengali). 

<!-- * Information Extraction: OpenIE
   * https://demo.allennlp.org/open-information-extraction/open-information-extraction
   * https://github.com/dair-iitd/OpenIE-standalone
* Entity Extraction: 
   * https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
   * https://www.nltk.org/book/ch07.html
 -->

## Rules of the Shared Task
* All solutions should be self-coded, plagiarism is strictly prohibited and strict action will be taken against the entire team found plagiarizing. Solutions from any current repository, any blog or any source on the internet is included in this.
* While your solution is not expected to be the best, we want to see the unique solutions you create in terms of features used, improvements made, extensive result analysis etc. 
* Regular discussions with a TA or PG sir are expected and encouraged. There would be bonus marks for this.
* Marks will be allotted based on ranking on the leaderboard, innovative ideas, and understanding of tasks and models involved. While the leaderboard will be present and you are encouraged to view it as a competition, final marks will not depend only on your ranking. We will also evaluate the efforts and the approach the team has made.
* For each task there would be a separate leaderboard.
* For a task, no more than 5 teams can join.

## Tutorials given in Course

1. [**NLTK Tutorial**](https://colab.research.google.com/drive/1OMMp7vGMhqDbkJMMefrX22yenL0tom_0?usp=sharing)
2. [**Notebook for Word-embeddings analogy:**](https://colab.research.google.com/drive/1DxC5AnIFuu9Maan_D23yoE6sW4Ez2Uhw?usp=sharing#scrollTo=yr64uvq1E3E-)
3. [**Notebook for Bias in word-embeddings:**](https://colab.research.google.com/drive/11PyFecG4gaWBrP1FB6nHUROktwUSUCbI?usp=sharing#scrollTo=SkOPpPRMEp1M)
4. [**Pytorch Tutorial**](https://colab.research.google.com/drive/10ejOXFuD8IBfGHmoydIfc4OMqzivCiAT?usp=sharing)

## Useful Resources for the Tasks

### Where to look? (Keywords, Concepts, Packages)

If you are unfamiliar with the most common concepts of NLP and ML, you can start searching for the following concepts. These are list of keywords without any specific structure.

1. Word Embeddings: Word2Vec, Glove, Gensim, Spacy
2. Contextual Word Embeddings (works better than simple word embeddings): BERT, ELMo, Huggingface, AllenNLP
3. Metric Calculation: scikit-learn package
4. Classifiers: tf-idf classifier, bag-of-words, Naive Bayes, Classification using BERT
5. Text Generation: Sequence-to-Sequence, GPT-2, BART
6. Multilingual models: mBERT, MuRIL, XLM
7. Best tutorial websites: TowardsDataScience, Medium, Huggingface Documentation

### Free coding platforms for ML

1. Kaggle Notebooks
1. Google Colab


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
