# README

标签（空格分隔）： 计算fid说明

---

### Frechet Inception Distance(FID):provides an alternative approach . to quantify the quality of generated samples, they are  first embedded into a feature space given by(a specific layer) of Inception Net.
### 上面这句话的意思就是说，FID这种方法是用来衡量生成样本质量的，这些生成样本呢需要把他们嵌入到一个特征空间中（一个特殊的网络层）。
## 公式：FID(x, g) = ||μx - μg||22 + Tr(Σx + Σg - 2(ΣxΣg)1/2)
### 上面公式可以参考论文中公式，格式有点小问题。论文链接：[Are GANs Created Equal? A Large-Scale Study][1]


----------
## 本实验的思路：wgan文件夹中有各种GAN的变种，我们要做的就是要比较各种gan之间的性能指标，在checkpoint文件中已经有训练好的各种gan模型的参数 ，直接可以运行各个gan得到各自的样本即可，然后通过get_fid中方法来计算fid值。
## get_fidScore文件夹中有两个重要的文件夹get_fid和getNpz，getNpz文件夹中要做的事儿是：计算原始数据集(MNIST和fashion-mnist两个数据集)的均值和协方差，并存储下来（方便计算fid值的时候用）。get_fid文件夹中才是真正开始计算fid值，我们把生成样本放到那个Inception Net中，然后输出并计算均值和协方差（这个过程inception.py中已经完成任务了，我们只需要往其中传值就行了），刚刚getNpz中也已经得到了均值和协方差（也是需要通过inception.py来计算的），有了这两组均值和协方差之后，我们就可以按照公式来进行计算了。


----------
最后贴出我们的计算结果，仅供参考：

![gan实验结果.png-11.3kB][2]


  [1]: https://arxiv.org/abs/1711.10337
  [2]: http://static.zybuluo.com/NumberFairy/zvmkhlyjodhb29fwlxmpfzus/gan%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png