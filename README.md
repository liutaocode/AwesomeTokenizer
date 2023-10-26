# AwesomeMultiModalTokenizer

The two-stage training paradigm has been garnering significant attention, particularly in the era of Large Language Models (LLMs). The first stage involves training a semantic auto-encoder tokenizer, serving as the fundamental building block. Step two involves token manipulation with certain conditions, usually prompts. The role of tokenizers has varied beyond textual information to include other forms of data such as audio, images, and videos.

In this repository, we focus on audio, image, and video tokenizer. 

*Note: This repository is still under construction. Feel free to have a PR.*

## Text Tokenizer
Not discussed in this repo. Please refer to [URL1](https://github.com/huggingface/tokenizers) or [URL2](https://huggingface.co/docs/transformers/tokenizer_summary).

## Audio Tokenizer

Arxiv 2019,[VQ-Wav2Vec](https://arxiv.org/abs/1910.05453), [Code](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#vq-wav2vec)(Gumbel VQ)

TASLP 2021, [SoundStream](https://arxiv.org/abs/2107.03312), (*No Code*) (RVQ)

Arxiv 2022, [EnCodec](https://arxiv.org/abs/2210.13438),  [Code](https://github.com/facebookresearch/encodec) (RVQ)


## Image Tokenizer

**Image tokens** is also called **discrete visual tokens**.

NIPS 2017, [VQ-VAE](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)

CVPR 2021, [VQ-GAN](http://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf), [Code](https://github.com/dome272/VQGAN-pytorch)

ICLR 2022, [ViT-VQGAN](https://arxiv.org/pdf/2110.04627.pdf)

CVPR 2022, [RQ-GAN](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.pdf)

CVPR 2023, [Reg-VQ](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.pdf)

CVPR 2023, [Masked-VQ](http://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Not_All_Image_Regions_Matter_Masked_Vector_Quantization_for_Autoregressive_CVPR_2023_paper.pdf)

CVPR 2023, [Dynamic-VQ](http://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Towards_Accurate_Image_Coding_Improved_Autoregressive_Image_Generation_With_Dynamic_CVPR_2023_paper.pdf)

## Video Tokenizer

ICCV 2019, [VideoBERT](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_VideoBERT_A_Joint_Model_for_Video_and_Language_Representation_Learning_ICCV_2019_paper.pdf) (Tokenize the visual features using hierarchical kmeans)

CVPR 2023, [MAG-VIT](https://github.com/google-research/magvit), [Code](https://github.com/google-research/magvit)

Arxiv 2023, [MAG-VIT2](https://github.com/google-research/magvit) (LFQ:look-up free quantization)



## Applications

### Text-to-Image (T2I)

PMLR 2021, [DALL-E](https://arxiv.org/abs/2102.12092) (Based on Gumbel VQ)

Arxiv 2022, [Parti](https://arxiv.org/abs/2206.10789) (Based on ViT-VQGAN)

*Some T2I algorithms make use of a latent code, which is not listed here.*

### Audio Generation 

TASLP 2023, [Audio-LM](https://ieeexplore.ieee.org/abstract/document/10158503) (Based on SoundStream)

Arxiv 2023, [VALL-E](https://arxiv.org/abs/2301.02111), [Unofficial Code](https://github.com/lifeiteng/vall-e) (Based on Encodec)

Arxiv 2023, [SoundStorm](https://arxiv.org/abs/2305.09636) (Based on SoundStream)

Technique Report 2023, [AudioPaLM](https://arxiv.org/pdf/2306.12925.pdf)

### Talking Face Sythesis

ECCV 2022, [VQFR](https://arxiv.org/abs/2205.06803), [Code](https://github.com/TencentARC/VQFR) (It provides a pre-train VQ codebook on FFHQ datasets.)

AAAI 2023, [VPNQ](https://ojs.aaai.org/index.php/AAAI/article/view/25354/25126)

CVPR 2023, [CodeTalker](https://arxiv.org/abs/2301.02379), [Code](https://doubiiu.github.io/projects/codetalker/) (3D Discrete Motion Features.)

## Vector Quantization Repo

[VQ (lucidrains)](https://github.com/lucidrains/vector-quantize-pytorch/tree/master/vector_quantize_pytorch), This repo contains multiple implementations of VQ.

