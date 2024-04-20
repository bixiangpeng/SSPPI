# SSPPI (Continuously Updating......)
---
A repo for "SSPPI: Cross-modality enhanced protein-protein interaction prediction from sequence and structure perspectives".


## Contents

* [Abstracts](#abstracts)
* [Requirements](#requirements)
   * [Download projects](#download-projects)
   * [Configure the environment manually](#configure-the-environment-manually)
   * [Docker Image](#docker-image)
* [Usages](#usages)
   * [Project structure](#project-structure)
   * [Data preparation](#data-preparation)
   * [Training](#training)
   * [Pretrained models](#pretrained-models)
* [Results](#results)
   * [Experimental results](#experimental-results)
   * [Reproduce the results with single command](#reproduce-the-results-with-single-command)
* [Baseline models](#baseline-models)
* [NoteBooks](#notebooks)
* [Contact](#contact)

## Abstracts



![SSPPI architecture](https://github.com/bixiangpeng/SSPPI/blob/main/framework.png)


## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/SSPPI`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n SSPPI python=3.7 ` and `conda activate SSPPI`
   
   Install specified version of pytorch: ` conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
   
   Install other python packages:
   ```shell
   pip install -r requirements.txt \
   && pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
   && pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
   && pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   ```
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `23.3.1`.


* ### Docker Image

    We also provide the Dockerfile to build the environment, please refer to the Dockerfile for more details. Make sure you have Docker installed locally, and simply run following command:
   ```shell
   # Build the Docker image
   sudo docker build --build-arg env_name=SSPPI -t hisif-image:v1 .
   # Create and start the docker container
   sudo docker run --name hisif-con --gpus all -it hisif-image:v1 /bin/bash
   # Check whether the environment deployment is successful
   conda list 
   ```
  
##  Usages

* ### Project structure

   ```text

   ```
* ### Data preparation
  There are three benchmark datasets were adopted in this project, including two binary classification datasets (`Yeast and Multi-species`) and a multi-class classification dataset (`Multi-class`).

   1. __Download processed data__
   
      The data file (`data.zip`) of these three datasets can be downloaded from this [link](https://drive.google.com/file/). Uncompress this file to get a 'data' folder containing all the original data and processed data.
      
      🌳 Replacing the original 'data' folder by this new folder and then you can re-train or test our proposed model on Yeast, Multi-species or Multi-class.  
      
      🌳 For clarity, the file architecture of `data` directory is described as follows:
      
      ```text
       
      ```
   3. __Customize your data__

      

* ### Training
  After processing the data, you can retrain the model from scratch with the following command:
  ```text
  

   ```
   
* ### Pretrained models
   If you don't want to re-train the model, we provide pre-trained model parameters as shown below. 
<a name="pretrained-models"></a>

   
  
## Results

* ### Experimental results


   
* ### Reproduce the results with single command

## Baseline models


## NoteBooks


## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.

