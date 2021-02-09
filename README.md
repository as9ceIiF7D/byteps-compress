# Install 

## Prepare 

Environment:

- Ubuntu 18.04 LTS
- CUDA 10.2 
- cuDNN 7.6.5
- NCCL 2.7.6
- Python3.6 
- g++ 7.5 
- torch==1.7.0 
- torchvision==0.8.1
- mxnet-cu102==1.7.0 (we use this [build](https://repo.mxnet.io/dist/python/cu102/mxnet_cu102-1.7.0b20200813-py2.py3-none-manylinux2014_x86_64.whl))
- gluoncv==0.8.0

We recommend using [Amazon Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/), which has pre-installed most of the above libraries.

## Compile BytePS

Before compiling BytePS, please specify NCCL home directory. Otherwise, PyTorch extension might fail to compile. 

```sh
cd byteps
export BYTEPS_NCCL_HOME=/usr/local/cuda
python3 setup.py install --user
```

You can check the install by running 
```sh
python3 -c "import byteps.mxnet" 
python3 -c "import byteps.torch"
```

If there is no error message, then it works. 


# Reproduce

All experiments are conducted on [Amazon EC2 P3.16xlarge instances](https://aws.amazon.com/ec2/instance-types/p3/), each equipped with 8 16GB V100 GPUs and 25Gbps Ethernet. 

Please prepare the ip list of your hosts in `worker-hosts` and `server-hosts` file. For distributed training, it should contain multiple lines of ip. Since PS workers and servers are co-located in each node, two host files can the same. If you want to launch twice as many servers, just copy the ip list twice in `server-hosts`. 

You may also need to prepare a pem file to ssh to other nodes without password.

## ImageNet

We use `ImageRecord` format to store the dataset. Please refer to [GluonCV's document](https://cv.gluon.ai/build/examples_datasets/recordio.html) for more details.

We evaluate two representive CNN models: ResNet50_v2, and VGG-16. 

### Training 

[Training script](byteps/example/mxnet/train_gluon_imagenet_byteps_gc.py) comes from GluonCV, with a small modification to adopt gradient compression. 

To train, run 
```sh
cd example/mxnet
./train_imagenet.sh baseline 0.2
```

For ResNet50, we train with 8 Amazon EC2 P3.16xlarge. For VGG16, we train with 4 Amazon EC2 P3.16xlage.

**Hyper-Parameter**

The base LR refers to the learning rate with single node. We follow _`Linear Scaling Rule`_ proposed in [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf). For example, if the base LR is `0.2`, then with 8 Amazon EC2 P3.16xlarge, the actual learning rate should be `0.2 * 8 = 1.6`. We only change learning rate. We use the default values for other hyper-parameters.

**ResNet50**

We use a smaller learning rate for 1-bit, top-k and random-k. The total batch size is 4k.

| Algorithm                  | base LR |
| -------------------------- | ------- |
| NAG (FP32)                 | 0.2     |
| NAG (FP16)                 | 0.2     |
| Scaled 1-bit with EF       | 0.1     |
| Top-k (k=0.1%) with EF     | 0.1     |
| Random-k (k=1/32) with EF  | 0.1     |
| Linear Dithering (5 bits)  | 0.2     |
| Natural Dithering (3 bits) | 0.2     |

**VGG16**

We use a larger learning rate for 1-bit, top-k and random-k. The total batch size is 1k.

| Algorithm                  | base LR |
| -------------------------- | ------- |
| NAG (FP32)                 | 0.01    |
| NAG (FP16)                 | 0.01    |
| Scaled 1-bit with EF       | 0.015   |
| Top-k (k=0.1%) with EF     | 0.015   |
| Random-k (k=1/32) with EF  | 0.015   |
| Linear Dithering (5 bits)  | 0.01    |
| Natural Dithering (3 bits) | 0.01    |



## BERT 

The code comes from [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT), with a small modification to accommodate gradient compression.


### Pretraining 

We use mixed precision to accelerate pretraining with `apex` package. To install `apex`, run

```sh
cd apex 
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ 
```

We provide a highly-optimized version of LANS optimizer written in CUDA ([code](apex/csrc/multi_tensor_lans.cu)).

The preprocessed dataset is partitioned into 1536 shards.

For BERT-base, we train with 4 Amazon EC2 P3.16xlarge.

**Hyper-Parameter**

Following [LAMB](https://openreview.net/pdf?id=Syx4wnEtvH), we use _`Sqare Root Scaling Rule`_.

We use a larger learning rate for 1-bit in phase 1 and smaller learning rate in phase 2. For top-k, we use a smaller learning rate in both phases. We only change learning rate. We use the default values for other hyper-parameters.

Total batch size is 2k.

| Algorithm                       | LR (phase 1) | LR (phase 2) |
| ------------------------------- | ------------ | ------------ |
| LANS                            | 0.00125      | 0.00125      |
| CLAN(Scaled 1-bit with EF)      | 0.0014865    | 0.001051     |
| CLAN(Top-k (k=0.1%) with EF)    | 0.001051     | 0.001051     |
| CLAN(Linear Dithering (7 bits)) | 0.00125      | 0.00125      |


### Finetune

**SQuAD** 

We evaluate on SQuAD v1.1. We do not change any hyper-parameters. 

To finetune SQuAD, run
```sh
cd BERT
./scripts/run_squad.sh
```

**GLUE**

By default, we use a batch size of 32 and train for 3 epochs. But for MRPC dataset, it is a tiny dataset. We train it for 5 epochs. We find that in some cases, the loss might be NaN. If that happens, we will use FP32 to finetune instead of FP16.  

To finetune GLUE, run
```sh
cd BERT
./scripts/run_glue.sh 
```

