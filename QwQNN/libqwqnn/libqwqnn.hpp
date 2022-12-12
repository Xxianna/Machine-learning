//this file is the main header file of the library qwqnn
//______________________________________________________
//-------------THE ONLY AUTHOR ：Xxianna-----------------
//______________________________________________________
//please link these necessary libraries:
//libqwqtensor.lib or libqwqtensor_SSE4.lib
//  -fopenmp
//please link these libraries if you want better performance:
//libqwqtensor_SSE4.lib -msse4

//本库设计了神经网络类，用于定义和训练神经网络

#ifndef LIBQWQNN_HPP
#define LIBQWQNN_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "libqwqtensor\libqwqtensor.hpp"
//上述文件已经包含<memory>和<omp.h>

#define QWQNN_VERSION "0.2.1"

//file comment:
//Class Layer is the base class of all layers.
//Class Optimizer is the class to optimize the network with any optimizer linked.
//Class LossFunction is the class to calculate the loss of the network with any loss function linked.
//Class NeuralNetwork is the main class of this library.

//other actually useful classes:
//wait for update...

using namespace std;
namespace qwqnn
{
//考虑层有两种，一种是激活层，一种是全连接层（或者叫卷积层、池化层）也包含dropout层
//所需的功能：
//自动或手动产生输出维度
//记录上一层和下一层的指针（双向链表）
//记录输入维度、输出维度、自身所需的各种权重、参数等
//引入一种特殊的层，允许用户以任意手段操作数据，但：输入输出数目不变，误差无操作透传
//这种操作相当于在pytorch中，在forward中进行的非预制操作
//层类要能够：
//1.前向传播计算误差
//2.反向传播计算自身梯度，存储自身梯度
//3.更新自身权重
//4.传播误差，交给上一层
//对于激活层，没有权重要求。激活层和偏置层分开设计
//取消了上一代的偏置神经元设定。一方面造成了太多+1 -1的算式
//使得代码难以维护，另一方面，偏置神经元权重随着网络单层
//神经元的增加而占比越来越小，极其难以训练。
//在激活层设计偏置更为直观
class Layer
{
public:
    //variables
    string name="unknown"; //层的名字，用于输出（可自定义）
    string name_type="unknown"; //层的类型名，标准化
    Tensor INPUT; //输入数据.初始化一次赋值dim和shape，后续只更新data
    Tensor OUTPUT; //输出数据.初始化一次赋值dim和shape，后续只更新data
    Tensor WEIGHT; //权重数据.初始化一次赋值dim和shape，后续只更新data
    Tensor GRAD; //梯度数据.初始化一次赋值dim和shape，后续只更新data
    Tensor LOSS; //误差数据.初始化一次赋值dim和shape，后续只更新data
    Layer *prev=nullptr; //上一层的指针
    Layer *next=nullptr; //下一层的指针
    //还要有一些必要的备用变量（与是什么层无关的传播必须的变量）
    Tensor* Recent_Grad=nullptr; //最近的一些梯度，用法交给优化器
    Tensor* Recent_Weight=nullptr; //最近的一些权重，用法交给优化器
    //functions necessary
    //const Tensor& forward(const Tensor& input); //前向传播。返回OUTPUT，以此作为下一层的输入
    //以上是安全形式，但Tensor类里还要改很多才能实现，因此不用
    virtual Tensor forward(Tensor& input)=0; //前向传播。返回OUTPUT，以此作为下一层的输入
    virtual Tensor backward(Tensor& loss)=0; //反向传播。返回LOSS，以此作为上一层的误差
    //梯度已经算好了，也就是更新权重已经不需要知道层的具体结构了
    //因此，交给优化器即可
    //其他
    Layer& set_prev(Layer* prev_layer);//设置上一层.返回自身.用于链式调用
    Layer& set_next(Layer* next_layer);//设置下一层.返回自身.用于链式调用
    Layer& set_name(string name);
    //init default
    Layer();
    //定义过程是python可选参数的c++实现，也就是链式调用
    //但实际上以上参数对于一般的层来说是必须的
    //因此，这里只定义一个默认的构造函数
    virtual string print(); //返回一个字符串，自我介绍的一行
    //下面的函数用于保存权重
    //virtual char save_weight(ofstream& file);
    //virtual char load_weight(ifstream& file);
};
//创建一个DataLoader类，用于读取数据,完全由用户自定义
//这个类的作用是：读取数据，返回一组input和output
class DataLoader
{public:
    //只提供一个基础接口，返回一个input和output
    virtual bool get(Tensor* input,Tensor* output)=0;
    //如果返回false，说明数据读取完毕，可以结束训练了
};

//这是一个优化器类，所有优化器都受它调用
//优化器类要能够：
//1.接受一个层的指针，更新该层的梯度表，再调用层的更新函数
//2.这期间，要对梯度进行处理
//3.要能查优化器名称表来选定优化器，用于参数处理
class Optimizer
{
public:
    //variables
    string name_type="unknown"; //优化器的类型名，标准化.但优化器本身可以自定义
    double learning_rate=0.01; //学习率
    float momentum=0; //动量
    double decay=1; //衰减
    int epoch=0; //当前是第几个epoch
    //functions necessary
    virtual void update(Tensor& loss_end,Layer* tail)=0; //更新权重或者梯度
    Optimizer& set_learning_rate(double learning_rate); //设置学习率
    Optimizer& set_momentum(float momentum); //设置动量
    Optimizer& set_decay(double decay); //设置衰减
};

//这是一个损失函数类，所有损失函数都受它调用
//损失函数类要能够：
//通过神经网络的输出和标签，计算损失函数值
class LossFunction
{
public:
    //variables
    string name_type="unknown"; //损失函数的类型名，标准化.但损失函数本身可以自定义
    //functions necessary
    virtual Tensor loss(Tensor& output, Tensor& target)=0; //计算损失函数值 
};

//这是一个神经网络类
//神经网络类要能够：
//1.吃入一个层的指针，将其加入到层链表中
//2.吃入一个优化器的指针，将其存储
//3.吃入一个损失函数的指针，将其存储
//4.包含输入输出tensor格式等基本信息
//5.带有一些易用的接口，便于进行训练、预测、传播、梯度清零等
typedef double (*show_loss)(); //定义一个函数指针，用于显示误差
typedef struct{
    show_loss func;
    int interval;
}show_loss_struct;
class Network //继承只需要初始化网络，故不需要虚函数
{
public:
    //variables
    show_loss_struct show_loss_func; //显示误差的函数
    DataLoader *loader=nullptr; //数据加载器
    vector<Layer*> layers; //层链表
    Optimizer *optimizer=nullptr; //优化器
    LossFunction *lossfunc=nullptr; //损失函数
    Tensor INPUT; //输入tensor
    Tensor OUTPUT; //输出tensor
    Tensor loss_back; //损失函数的反向传播值
    int batch_size=1,epoch=1; //batch大小
    //functions necessary
    Network& set_show_loss(show_loss_struct show_loss); //设置显示误差的函数
    Network& add_layer(Layer *layer); //添加层
    Network& set_optimizer(Optimizer *optimizer); //设置优化器
    Network& set_lossfunc(LossFunction *lossfunc); //设置损失函数
    Network& set_batch_size(int batch_size);
    Network& set_epoch(int epoch);
    Network& set_input(Tensor& input); //设置输入tensor
    Network& set_output(Tensor& output); //设置输出tensor
    Network& set_dataloader(DataLoader *loader); //设置数据加载器
    Network& forward(); //前向传播
    Network& backward(); //反向传播
    void update(); //更新权重.包含了梯度清零
    //void clear_grad(); //清空梯度
    void train(); //训练
    Tensor predict(Tensor& input); //预测
    //init default
    Network();
    bool end_read=false;
    Tensor train_out;
};
//加入一个tensor裁剪的函数（直接在原tensor上修改）
void clip(Tensor& input,float min,float max);
//加入一个产生gaussian分布随机数的函数
float gaussian(float mean,float sigma);

//激活函数及其导数
float relu(float x);
float relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);
float tanh(float x);
float tanh_derivative(float x);
float softmax(float x);
float softmax_derivative(float x);

//损失函数:一些简化
float mse(float x, float y);
float cross_entropy(float x, float y);

//层:继承自基类
class LinearLayer:public Layer
{
public:
    LinearLayer(Tensor input,Tensor output); //初始化
    Tensor forward(Tensor& input);
    Tensor backward(Tensor& loss);
};
//卷积层：暂未实现
//偏置层
class BiasLayer:public Layer
{public:
    BiasLayer(Tensor& input); //初始化
    Tensor forward(Tensor& input);
    Tensor backward(Tensor& loss);
};
//激活层
typedef float (*activation_func)(float); //激活函数类型
struct activ_info //激活函数信息
{
    string name;
    activation_func func;
    activation_func dfunc;
};
class ActivationLayer:public Layer
{public:
    activ_info info0;
    ActivationLayer(activ_info info); //初始化
    Tensor forward(Tensor& input);
    Tensor backward(Tensor& loss);
};
//dropout层
// class DropoutLayer:public Layer
// {public:
//     DropoutLayer(float rate); //初始化
//     ~DropoutLayer(); //析构
// };

//优化器：由于传参和权限问题，必须使用继承
//在这里声明几个定义好的类
class NoneOptimizer:public Optimizer
{public:
    NoneOptimizer();
    void update(Tensor& loss_end,Layer* tail);
};
class SGD:public Optimizer
{public:
    SGD();
};
class Momentum:public Optimizer
{public:
    Momentum();
    void update(Tensor& loss_end,Layer* tail);
};
class Nesterov:public Optimizer
{public:
    Nesterov();
};
class AdaGrad:public Optimizer
{public:
    AdaGrad();
};
class RMSProp:public Optimizer
{public:
    RMSProp();
};
class Adam:public Optimizer
{public:
    Adam();
    void update(Tensor& loss_end,Layer* tail);
};

//损失函数类声明
class MSE:public LossFunction
{public:
    MSE();
};
class CrossEntropy:public LossFunction
{public:
    CrossEntropy();
};
class Linearloss:public LossFunction
{public:
    Linearloss();
    Tensor loss(Tensor& output, Tensor& target);
};

}//namespace qwqnn
#endif
