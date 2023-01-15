# 记录kaggle的必要commands
# 本地github push时显示
- fatal: unable to access 'https://github.com/tianyolanda/football_seg_kaggle.git/': Could not resolve host: github.com
- 解决：
1. clashX点击复制终端代理命令
2. 复制到终端里
3. 重试一次即可

# 删除全部后缀为.h5的文件
rm -d *.h5

# 当前console所在的目录位置，就是代码中保存下来的东西的存储位置！

# 每个session每周只能运行12h，（然后就需要换一个session了）。整个kaggle可以运行30h。可以换手机号注册继续运行

# 如何薅羊毛用kaggle的GPU跑自己代码 

!git clone https://github.com/tianyolanda/football_seg_kaggle.git

cp -r /kaggle/working/football_seg_kaggle/* .


## 难点在于：自己代码是由很多文件夹组成的
- 问题（1）如何复制到kaggle环境？
- 问题（2）script写代码时会出现import层级问题

## 问题（1）：
### 分析：网上很多人提供的方法是放到input中，类似数据集的上传，把代码上传上去。
但更新代码很麻烦，需要重新上传压缩文件。
因此我的思路是一步到位，把代码通过git放到output中（可随时更新）：
### 解决方案
- （1）使用git，把代码下载到/kaggle/working/中
!git clone https://github.com/tianyolanda/football_seg_kaggle.git

- （2）将git的都拷贝过来到output/文件夹
cp -r /kaggle/working/football_seg_kaggle/* .

## 问题(2)：
### 分析：git的操作导致了下一个问题（不太确定是不是我打开方式不对）：
 1. 我发现kaggle中唯一可运行、可以写代码的script，也就是你脸上的那个可以编辑的代码页面。这个script他是无法移动位置的。因为我在console里ls也看不到他，无法用mv命令把他挪走。
 2. 但我们git下载的是一个文件夹。里面的库是以该文件夹作为root，而script是在文件夹的外面一层，他俩差了一级！！所以会出现import层级问题
 例子： 比如 script是在 /kaggle/working/
 然鹅 其他代码都在/kaggle/working/football_seg_kaggle/
 而 他们在调用时是以football_seg_kaggle/作为当前文件夹的。
 3. 我们的目的是要在script中调用git文件夹的.py，一个直白的办法是，在调用的时候，每个包前面直接加上git文件夹的名。但其实这些.py中也有import，会互相调用、就要更改全部文件的import部分，太麻烦了。
### 解决方案 
 我想到了另一个简单完美的解决方法：
- (3)把git的文件夹的全部文件都拷贝过来到input/文件夹：
cp -r /kaggle/working/football_seg_kaggle/* .

这样所有文件就都在/kaggle/working/层级，调用无障碍~

## 现存其他问题：
1. git过来的文件是无法更改的
这个也么办法，任何方式上传的都没法改。
目前只能在本地改好，然后git pull下来最新的代码
好在这些文件基本是库，改的频率比较低。

2. tensorboard用不了
换另一个本地绘图的试试吧 

3. 看代码不方便
在本地看吧。。




