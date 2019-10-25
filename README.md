# Self-supervised GAN: Analysis and Improvement with Multi-class Minimax Game

Tensorflow code of the paper "Self-supervised GAN: Analysis and Improvement with Multi-class Minimax Game", NeurIPS 2019<br>
Ngoc-Trung Tran, Viet-Hung Tran, Ngoc-Bao Nguyen, Linxiao Yang, and Ngai-Man Cheung.

## Dependencies
* Python (2.7, 3.5, 3.6), Numpy, Tensorflow, SciPy, scikit-learn
* Recent NVIDIA GPUs

## Data
* Supporting datasets: MNIST, Stacked MNIST (or MNIST 1K) CIFAR-10/100, STL-10 and ImageNet 32x32.
* When the code runs first time, the dataset is automatically downloaded in `./output/` or your defined path `--out_dir=...`.

### Training and testing

* `ss_task`: `0` (no SS task), `1` (SS task), `2` (our MS task)
* `python msdistgan_[*].py --help` for more information.

#### MNIST

```
//Training with MS task
python msdistgan_mnist.py --ss_task=2
```
There is no evaluation method on this dataset.

#### Stacked MNIST

```
//Training with MS task
python msdistgan_mnist1k.py --k=2 --ss_task=2 --is_train=1
```

```
//Testing with MS task
python msdistgan_mnist1k.py --k=2 --ss_task=2 --is_train=0
```

* `k`: `4` (K/4 architecture), `2` (K/2 architecture), `1` (Full size) (Refer to network architectures of Unrolled GAN [1])

#### CelebA

```
//Training with MS task
python msdistgan_celeba.py --ss_task=2 --is_train=1
```

```
//Testing with MS task
python msdistgan_celeba.py --ss_task=2 --is_train=0
```

#### CIFAR-10/100

```
//Training with MS task on CIFAR-10 with Resnet and Hinge loss
python msdistgan_cifar.py --db_name=cifar10 --nnet_type=resnet --loss_type=hinge --ss_task=2 --data_source=./data/cifar10/ --is_train=1

//Training with MS task on CIFAR-100 with Resnet and Hinge loss
python msdistgan_cifar.py --db_name=cifar100 --nnet_type=resnet --loss_type=hinge --ss_task=2 --data_source=./data/cifar100/ --is_train=1
```

```
//Computing FID (10K-10K) of the pre-trained model of CIFAR-10
python msdistgan_cifar.py --db_name=cifar10 --nnet_type=resnet --loss_type=hinge --ss_task=2 --data_source=./data/cifar10/ --nb_test_real=10000 ----nb_test_fake=10000 --is_train=0

//Computing FID (10K-10K) of the pre-trained model of CIFAR-10
python msdistgan_cifar.py --db_name=cifar100 --nnet_type=resnet --loss_type=hinge --ss_task=2 --data_source=./data/cifar100/ --nb_test_real=10000 ----nb_test_fake=10000 --is_train=0
```

#### STL-10
```
//Training with MS task on STL-10 with Resnet and Hinge loss
python msdistgan_stl10.py --nnet_type=resnet --loss_type=hinge --ss_task=2 --is_train=1
```

```
//Computing FID (10K-10K) of the pre-trained model of STL-10
python msdistgan_stl10.py --nnet_type=resnet --loss_type=hinge --ss_task=2 --nb_test_real=10000 ----nb_test_fake=10000 --is_train=0
```

#### ImageNet 32x32

```
//Training with MS task on Imagenet 32x32 with Resnet and Hinge loss
python msdistgan_imagenet32.py --nnet_type=resnet --loss_type=hinge --ss_task=2 --is_train=1
```

```
//Computing FID (10K-10K) of the pre-trained model of Imagenet 32x32
python msdistgan_imagenet32.py --nnet_type=resnet --loss_type=hinge --ss_task=2 --nb_test_real=10000 ----nb_test_fake=10000 --is_train=0
```

### Citation

If you find this work useful in your research, please consider citing:

```
@InProceedings{tran_2019_neurips_gan,
  author = {Tran, Ngoc-Trung and Tran, Viet-Hung and Nguyen, Ngoc-Bao and Yang, Linxiao and Cheung, Ngai-Man},
  title = {Self-supervised GAN: Analysis and Improvement with Multi-class Minimax Game},
  booktitle = {NeurIPS},
  month = {December},
  year = {2019}
}
```

### References

[1] Unrolled Generative Adversarial Networks, ICLR 2016.

We're going to release TPU code of our model soon.

