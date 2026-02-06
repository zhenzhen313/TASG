## Target-specific attention and sentiment-guided graph neural network for target-oriented multimodal sentiment classification


---
### Abstract
Target-oriented multimodal sentiment classification is an important subtask of aspect-based sentiment analysis, which analyzes the sentiment polarity oriented to the target in a sentence-image pair. Previous research primarily relies on generic object detection, neglecting visual sentiment cues and suffering from semantic noise introduced by image-to-text conversions. Moreover, although existing approaches explore intra-modal and inter-modal feature interaction and enhancement for target-oriented representations, they leave potential for improvement in capturing complex relationships for target-level sentiment analysis. To address these limitations, we propose a target-oriented multimodal sentiment classification framework with Target-specific Attention and Sentiment-guided Graph neural network (TASG). For visual inputs, we decompose them into content-descriptive captions and sentiment-aware adjective-noun pairs. We design a target-specific attention module to enhance target-oriented features while mitigating noise from complex cross-modal conversions. Furthermore, we introduce a sentiment-guided graph neural network that constructs a semantic graph based on syntactic sentiment cues and multimodal information, allowing the model to explicitly model and refine target-sentiment relationships. Experimental results on standard Twitter benchmarks and the large-scale MASAD dataset demonstrate the effectiveness of TASG. The code will be made publicly available upon acceptance of the paper.


###  Overview
<p align="center">
  <img src="./images/overview.jpg" alt=" Overview of the proposed model.">
</p>

Figure shows the overview of our designed framework, which comprises four primary modules: 
1) Feature Extraction Module, employs the existing methods to generate textual information corresponding to the image features, and a PLM is utilized to extract text embeddings. 
2) Target-Specific Attention Module, it enhances the representation of targets in textual data, and it further refines the caption embeddings and auxiliary representation based on target information. 
3) Sentiment-Guided Graph Neural Network Module, it extracts sentiment cues from the multimodal data by applying sentiment filtering in conjunction with graph convolutional operations. 
4) Prediction Module, it integrates the extracted target and sentiment features, and employs a softmax layer to calculate the final sentiment predictions,. 


---
### Follow the steps below to run the code:
1. download [Bertweet-base](https://arxiv.org/abs/2005.10200), and put it in `./bertweet-base` directory
2. download [dataset](https://www.ijcai.org/proceedings/2019/751)
3. install packages (see `requirements.txt`)
4. run `bash scripts/*.sh`

---

```
