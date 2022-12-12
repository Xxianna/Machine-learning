#include "libqwqnn.hpp"
//was included:iostream,fstream,memory,omp,libqwqtensor

using namespace std;
using namespace qwqnn;

//define the layers in libqwqnn.hpp

//backward不止要计算给前一层的梯度，还要计算自己的梯度，用于更新权重

//全连接
LinearLayer::LinearLayer(Tensor input,Tensor output){
    //根据input和output初始化基类
    this->INPUT=input;
    this->OUTPUT=output;
    this->name_type="LinearLayer";
    //初始化权重和梯度
    //全连接中间过程其实没有形状，直接输入输出size相乘
    int sz[2]={input.size(),output.size()},atpos[1]={0};
    this->WEIGHT=Tensor(2,sz);
    this->GRAD=Tensor(2,sz);
    //初始化其数值
    int i;float sigma=sqrt(2.0/(input.size()));
    #pragma omp parallel for private(i)
    for(i=0;i<WEIGHT.size();i++){
        *(WEIGHT.at(1,atpos)+i)=gaussian(0,sigma);
        *(GRAD.at(1,atpos)+i)=0;
    }
}
Tensor LinearLayer::forward(Tensor& input){
    //将input reshape到一条的二维，然后乘以权重，再reshape回去
    int sz[2]={1,input.size()};
    input.reshape(2,sz);
    INPUT=input;
    Tensor output=input.mul(WEIGHT);
    output.reshape(OUTPUT.dim,OUTPUT.shape);
    input.reshape(INPUT.dim,INPUT.shape);
    return output;
}
Tensor LinearLayer::backward(Tensor& grad){
    //输入grad的形状是output的形状，要reshape成一条的二维
    int sz[2]={1,OUTPUT.size()},szr[2]={1,INPUT.size()},sz1[2]={szr[1],szr[0]};
    grad.reshape(2,sz);
    Tensor result(2,szr);
    //计算上一层的梯度
    Tensor WEIGHT_T=WEIGHT.transpose();
    result=grad.mul(WEIGHT_T);
    // //计算自己的梯度
    Tensor input_T=INPUT;
    input_T.reshape(2,sz1);
    Tensor grad_w=input_T.mul(grad);
    //暂时不用以上方法，采用累加的方法
    // Tensor grad_w=GRAD*0;
    // int i,j,atpos[1]={0};
    // for(i=0;i<szr[0];i++){
    //     for(j=0;j<szr[1];j++){
    //         *(grad_w.at(1,atpos)+i*szr[1]+j)=*(INPUT.at(1,atpos)+i*szr[1]+j)*(*(grad.at(1,atpos)+i*szr[1]+j));
    //     }
    // }
    //对grad_w进行裁剪
    float max=1.0/sqrt(WEIGHT.size());
    clip(grad_w,-max,max);
    GRAD=GRAD+grad_w;
    //还原形状
    grad.reshape(OUTPUT.dim,OUTPUT.shape);
    result.reshape(INPUT.dim,INPUT.shape);
    return result;
}

// class BiasLayer:public Layer
// {public:
//     BiasLayer(Tensor& input); //初始化
//     Tensor forward(Tensor& input);
//     Tensor backward(Tensor& loss);
// };
//偏置
BiasLayer::BiasLayer(Tensor& input){
    //根据input初始化基类
    INPUT=input;
    OUTPUT=input;
    this->name_type="Bias";
    //初始化偏置和梯度,可以直接用input的shape
    this->WEIGHT=Tensor(input.dim,input.shape);
    this->GRAD=Tensor(input.dim,input.shape);
    //初始化其数值
    int i,atpos[1]={0};
    #pragma omp parallel for private(i)
    for(i=0;i<WEIGHT.size();i++){
        *(WEIGHT.at(1,atpos)+i)=rand()%1000/1000.0-0.5;
        *(GRAD.at(1,atpos)+i)=0;
    }
}
Tensor BiasLayer::forward(Tensor& input){
    return input+WEIGHT;
}
Tensor BiasLayer::backward(Tensor& grad){
    //需要求出偏置的梯度，但是偏置的梯度就是grad
    GRAD=GRAD+grad;
    return grad;
}

//激活层
// class ActivationLayer:public Layer
// {public:
//     activation_func func; //激活函数
//     ActivationLayer(activ_info info); //初始化
//     Tensor forward(Tensor& input);
//     Tensor backward(Tensor& loss);
// };
ActivationLayer::ActivationLayer(activ_info info){
    //根据input初始化基类
    name_type="Activ";
    //激活层不需要任何参数
    name=info.name;
    info0=info;
}
Tensor ActivationLayer::forward(Tensor& input){
    //input需要保存，因为反向传播需要
    this->INPUT=input;
    //对input的每一项进行激活函数
    Tensor output=input;
    int i,atpos[1]={0};
    #pragma omp parallel for private(i)
    for(i=0;i<input.size();i++){
        *(output.at(1,atpos)+i)=info0.func(*(output.at(1,atpos)+i));
    }
    return output;
}
Tensor ActivationLayer::backward(Tensor& grad){
    //根据激活函数的导数，对grad进行修改
    Tensor output=grad;
    int i,atpos[1]={0};
    #pragma omp parallel for private(i)
    for(i=0;i<grad.size();i++){
        *(output.at(1,atpos)+i)=info0.dfunc(*(INPUT.at(1,atpos)+i))*(*(output.at(1,atpos)+i));
    }
    return output;
}


