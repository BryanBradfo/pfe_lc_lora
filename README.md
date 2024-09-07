# **On Efficient Constructions of Checkpoints for Deep Neural Networks** 🤖

<p align="center">
  <img src="img/checkpoint.webp" />
</p>

## **Overview** ✨
This project proposes an ingenious scheme, **leveraging the benefits** of [**Delta-LoRA**](https://arxiv.org/abs/2309.02411)  (a modified version of [LoRA](https://arxiv.org/abs/2106.09685): Low Rank Adaptation) and [**LC-checkpoint**](https://arxiv.org/abs/2009.13003), which is a **checkpointing scheme**. This innovative framework aims to facilitate the training of deep neural networks by **creating compressed checkpoints**. These checkpoints allow the training process to **resume from the last saved state** in the event of failures (such as gradient explosion or division by zero), thus **saving time and computational resources**.

## **Motivation** 🏋️
The code is driven by two main ambitions:
1. To create a framework that supports the training of deep neural networks with the ability to **create compressed checkpoints**, enabling the resumption of fine-tuning without starting from scratch in case of failures.
2. To establish a framework that also **prevents data poisoning**; thus, if malicious data is detected, training can resume from a model checkpoint that was last trained on clean data.

## **Technologies** 🧑‍💻

* [Python - matplotlib, pandas, seaborn, numpy, zlib](https://www.python.org/)
* [Pytorch](https://pytorch.org/)

## **Installation Instructions** ⌨️

### Option 1: Fresh Environment Setup 😁

For a fresh environment setup, follow these steps:
- Install the latest version of Python.
- Install the latest version of Visual Studio Code.
- Install Python extension, Jupyter Notebook on Visual Studio Code.
- Install Anaconda.
- Open a terminal and run the following commands:
  ```bash
  conda create --name py310 python=3.10
  conda activate py310
  conda install cudatoolkit -c anaconda -y
  nvidia-smi
  conda install pytorch-cuda=11.8 -c pytorch -c nvidia -y
  conda install pytorch torchvision torchaudio -c pytorch -c nvidia -y
  pip install pandas scipy matplotlib pathos wandb
- These installation steps are primarily for Windows but can be easily adapted for Linux and macOS by modifying the commands accordingly.

### Option 2: Use Predefined Environment ✨

Alternatively, you can use the predefined environment file to set up your environment more quickly:

- Clone the repository from GitHub.
- Open a terminal in the cloned repository directory.
- Run the following command:
  ```bash
  conda env create -f environment.yml
  conda activate py310
  ```
- This will create a new conda environment named `py310` and install all the necessary packages.

## **Usage** 👋

Once installed, you can run the scripts inside the project directory to start the training process and utilize the checkpointing mechanisms.

## **Additional Resources** 😊

- [Link to the project report in French](docs/_FR_PFE-Efficient_checkpointing_for_Deep_Neural_Networks.pdf)
- [Link to the project report in English](docs/_EN_PFE-Efficient_checkpointing_for_Deep_Neural_Networks.pdf)
- [Access to the slides presentation in French](docs/_FR_Presentation-checkpointing_efficace_pour_les_DNNs.pdf)

## **How to start ?** 🚨

Kindly be aware that the code has been crafted with maximum flexibility in mind. Nevertheless, there's a possibility that you may need to customize it to suit your particular use case and circumstances.

## **Contact** 📩

- bryan [dot] chen [at] etu [dot] toulouse-inp [dot] com / t0934135 [at] u [dot] nus [dot] edu

## **Acknowledgments** 🙏
This project was built with guidance and support from:

- Assoc Prof [Ooi Wei Tsang](https://www.comp.nus.edu.sg/cs/people/ooiwt/) (NUS)
- Asst Prof [Axel Carlier](https://ipal.cnrs.fr/axel-carlier-personal-page/) (INP-ENSEEIHT)
- PhD Student [Yannis Montreuil](https://ipal.cnrs.fr/yannis-montreuil-personal-page/) (UPMC, Sorbonne University)
- Scientist [Lai Xing Ng](https://ipal.cnrs.fr/lai-xing-ng/) (A*STAR Institute for Infocomm Research)

Special thanks to [CNRS@Create](https://www.cnrsatcreate.cnrs.fr/) for supporting this research project.

We express our gratitude to all their contributors and maintainers!

## **References** 📚 

1. Yu Chen, Zhenming Liu, Bin Ren & Xin Jin's [On Efficient Construction of Checkpoints.](https://arxiv.org/abs/2009.13003)

2. Shuyu Zhang, Donglei Wu, Haoyu Jin, Xiangyu Zou, Wen Xia & Xiaojia Huang's [QD-Compressor: a Quantization-based Delta Compression Framework for Deep Neural Networks](https://ieeexplore.ieee.org/document/9643728)

3. Amey Agrawal, Sameer Reddy, Satwik Bhattamishra, Venkata Prabhakara Sarath Nookala, Vidushi Vashishth, Kexin Rong & Alexey Tumanov's [DynaQuant: Compressing Deep Learning Training Checkpoints via Dynamic Quantization](https://arxiv.org/abs/2306.11800)

4. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang & Weizhu Chen's [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

6. Bojia Zi, Xianbiao Qi, Lingzhi Wang, Jianan Wang, Kam-Fai Wong, Lei Zhang's [Delta-LoRA: Fine-Tuning High-Rank Parameters with the Delta of Low-Rank Matrices](https://arxiv.org/abs/2309.02411) 

<!-- # LC-LoRA

### Introduction

Delta-compression framework for diverging branches in model training using Low-Rank Approximation (LoRA) and delta-encoding.
