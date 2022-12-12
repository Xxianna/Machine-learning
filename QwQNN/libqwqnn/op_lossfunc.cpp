#include "libqwqnn.hpp"
//was included:iostream,fstream,memory,omp,libqwqtensor

//using namespace std;
using namespace qwqnn;

//define the optimizers and loss functions in libqwqnn.hpp

NoneOptimizer::NoneOptimizer(){
    name_type="NoneOp";
    //empty
}
void NoneOptimizer::update(Tensor& loss,Layer* tail){
    //更新所有权重，不经过任何优化器
    //每次调用把lr乘以decay
    //追溯所有层，用每层的梯度直接更新权重。首先要判断这一层有没有权重，避免是激活层
    Layer* p=tail;
    while(p!=NULL){
        if(p->WEIGHT.dim!=0){
            p->WEIGHT=p->WEIGHT+p->GRAD*learning_rate;
            //梯度清零
            p->GRAD=p->GRAD*0;
        }
        p=p->prev;
    }
    learning_rate*=decay;
}
// class Adam:public Optimizer
// {public:
//     Adam();
//     void update(Tensor& loss_end,Layer* tail);
// };
// class Momentum:public Optimizer
// {public:
//     Momentum();
//     void update(Tensor& loss_end,Layer* tail);
// };
Momentum::Momentum(){
    name_type="Momentum";
    //empty
}
void Momentum::update(Tensor& loss,Layer* tail){
    //动量相关的（即之前的梯度）要存储在层的recent_grad中
    //如果是第一次调用（层的recent_grad为空），则创建一个和层的梯度一样大小的recent_grad
    //每次调用把lr乘以decay
    //追溯所有层，用每层的梯度直接更新权重。首先要判断这一层有没有权重，避免是激活层
    Layer* p=tail;
    while(p!=NULL){
        if(p->WEIGHT.dim!=0){
            if(p->Recent_Grad==nullptr){
                p->Recent_Grad=new Tensor(p->GRAD*1);
            }else *(p->Recent_Grad)=*(p->Recent_Grad)*momentum+p->GRAD*(1-momentum);
            p->WEIGHT=p->WEIGHT+*(p->Recent_Grad)*learning_rate;
            //梯度清零
            p->GRAD=p->GRAD*0;
        }
        p=p->prev;
    }
    learning_rate*=decay;
}

Linearloss::Linearloss(){
    name_type="direct";
    //empty
}
Tensor Linearloss::loss(Tensor& output,Tensor& target){
    //直接返回target-output
    return target-output;
    //return output-target;
}

float qwqnn::relu(float x){
    return x>0?x:0.1*x;
}
float qwqnn::relu_derivative(float x){
    return x>0?1:0.1;
}
float qwqnn::sigmoid(float x){
    return 1.0/(1.0+exp(-x))+0.0001*x;
}
float qwqnn::sigmoid_derivative(float x){
    return sigmoid(x)*(1-sigmoid(x))+0.0001;
}
float qwqnn::tanh(float x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))+0.0001*x;
}
float qwqnn::tanh_derivative(float x){
    return 1-tanh(x)*tanh(x)+0.0001;
}
float qwqnn::softmax(float x){
    return exp(x);
}
float qwqnn::softmax_derivative(float x){
    return softmax(x);
}
float qwqnn::mse(float x, float y){
    return (x-y)*(x-y);
}
float qwqnn::cross_entropy(float x, float y){
    return -y*log(x)-(1-y)*log(1-x);
}