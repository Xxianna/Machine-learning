#include "libqwqnn.hpp"
//was included:iostream,fstream,memory,omp,libqwqtensor

using namespace std;


namespace qwqnn{
//define the functions of classes in libqwqnn.hpp

//class layer
Layer::Layer(){
    //empty
}
Layer& Layer::set_name(string name){
    this->name=name;
    return *this;
}
Layer& Layer::set_prev(Layer* prev){
    this->prev=prev;
    return *this;
}
Layer& Layer::set_next(Layer* next){
    this->next=next;
    return *this;
}
string Layer::print(){
    string s;
    s+=name;
    for(int i=0;i<10-s.length();i++){
        s+=" ";
    }
    s=s+"|"+"unknown";
    return s;
}
Optimizer& Optimizer::set_learning_rate(double learning_rate){
    this->learning_rate=learning_rate;
    return *this;
}
Optimizer& Optimizer::set_momentum(float momentum){
    this->momentum=momentum;
    return *this;
}
Optimizer& Optimizer::set_decay(double decay){
    this->decay=decay;
    return *this;
}
Network::Network(){
    //empty
}
Network& Network::add_layer(Layer* layer){
    if(layers.size()==0){
        layer->set_prev(NULL);
        layer->set_next(NULL);
    }
    else{
        layer->set_prev(layers.back());
        layer->set_next(NULL);
        layers.back()->set_next(layer);
    }
    layers.push_back(layer);
    return *this;
}
Network& Network::set_optimizer(Optimizer* optimizer){
    this->optimizer=optimizer;
    return *this;
}
Network& Network::set_lossfunc(LossFunction *lossfunc){
    this->lossfunc=lossfunc;
    return *this;
}
Network& Network::set_batch_size(int batch_size){
    this->batch_size=batch_size;
    return *this;
}
Network& Network::set_epoch(int epoch){
    this->epoch=epoch;
    return *this;
}
Network& Network::set_input(Tensor& input){
    this->INPUT=input;
    return *this;
}
Network& Network::set_output(Tensor& output){
    this->OUTPUT=output;
    this->train_out=output;
    return *this;
}
Tensor Network::predict(Tensor& input){
    Tensor output,tmp;
    output=input;
    for(int i=0;i<layers.size();i++){
        tmp=layers[i]->forward(output);
        output=tmp;
    }
    return output;
}
Network& Network::set_dataloader(DataLoader* loader){
    this->loader=loader;
    return *this;
}
Network& Network::forward(){
    Tensor output,tmp;
    output=INPUT;
    for(int i=0;i<layers.size();i++){
        tmp=layers[i]->forward(output);
        output=tmp;
    }
    train_out=output;
    return *this;
}
Network& Network::backward(){
    Tensor output,tmp;
    output=lossfunc->loss(train_out,OUTPUT);
    for(int i=layers.size()-1;i>=0;i--){
        tmp=layers[i]->backward(output);
        output=tmp;
    }
    train_out=output;
    return *this;
}
void Network::update(){
    optimizer->update(train_out,layers[layers.size()-1]);
}
Network& Network::set_show_loss(show_loss_struct show_loss){
    this->show_loss_func=show_loss;
    return *this;
}
void Network::train(){
    for(int i=0;i<epoch;i++){
        //lossfunc->loss(train_out,OUTPUT).print();
        while(end_read==false){
            for(int j=0;j<batch_size;j++){
                end_read=loader->get(&INPUT,&OUTPUT);
                if(end_read==true) break;
                forward();
                backward();
            }
        }
        //如果达到interval，就调用show_loss_func
        if(i%show_loss_func.interval==0){
            printf("epoch:%d ",i);
            printf("loss: %12.8f\n",show_loss_func.func());
        }        
        update();
        end_read=false;
    }
}
void clip(Tensor& input,float min,float max){
    int i;
    #pragma omp parallel for private(i)
    for(i=0;i<input.size();i++){
        if(input.data[i]<min) input.data[i]=min;
        if(input.data[i]>max) input.data[i]=max;
    }  
}
float gaussian(float mean,float sigma){
    float x1,x2,w;
    do{
        x1=2.0*rand()/RAND_MAX-1.0;
        x2=2.0*rand()/RAND_MAX-1.0;
        w=x1*x1+x2*x2;
    }while(w>=1.0);
    w=sqrt((-2.0*log(w))/w);
    return x1*w*sigma+mean;
}


}