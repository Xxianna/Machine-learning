#include <iostream>
#include <fstream>
#include ".\libqwqnn\libqwqnn.hpp"
#include ".\libqwqnn\qwqnn_info_list.hpp"
using namespace std;
using namespace qwqnn;

//#define test(x)  ((2*(1-x+2*x*x)*exp(-x*x/2))/5)
//#define HW2_1(x)    ((2*(1-x+2*x*x)*exp(-x*x/2))/5)
//注意c++对sinx/x这种不会在0处取极限的
#define HW2_f4(x,y) ((x==0?1:(sin(x)/x))*(y==0?1:(sin(y)/y)))
//#define HW2_f5(x,y) (100*(y-x*x)*(y-x*x)+(1-x)*(1-x))

#define sample 121
float train_set[11*11][2],test_set[21*21][2];
float train_out[11*11],test_out[21*21];

class getdatatest:public qwqnn::DataLoader{
public:
    int i=0,atpos[1]={0};
    bool get(Tensor* input,Tensor* output){
        if(i>=sample){
            i=0;
            return true;
        }
        *(input->at(1,atpos))=train_set[i][0];
        *(input->at(1,atpos)+1)=train_set[i][1];
        *(output->at(1,atpos))=train_out[i];
        i++;
        return false;        
    }
};
Network *net0;
double HW1_showloss();
void dataset_init();
int main(){
    //try下面所有语句，所有报错接收后直接输出其报错信息
    try{
        dataset_init();
        show_loss_struct HW1_loss;//显示MSE用
        HW1_loss.interval=20;
        HW1_loss.func=HW1_showloss;
        int sz1[2]={1,2},sz2[2]={1,4},sz3[2]={1,1};
        Network net;net0=&net;
        Tensor t1(2,sz1),t2(2,sz2),t3(2,sz3);
        BiasLayer b1(t2);
        BiasLayer b2(t3);
        LinearLayer l1(t1,t2),l2(t2,t3);
        ActivationLayer a1(tanh_info);
        net.add_layer(&l1).add_layer(&b1).add_layer(&a1).add_layer(&l2).add_layer(&b2);
        Linearloss loss;//直接求差作为损失函数
        Momentum opt;//加入动量的梯度下降
        opt.set_decay(0.9996).set_learning_rate(0.001).set_momentum(0.6);
        net.set_batch_size(5).set_input(t1).set_output(t3);
        net.set_lossfunc(&loss);
        net.set_optimizer(&opt);
        net.set_epoch(800).set_show_loss(HW1_loss);
        getdatatest datatest;
        net.set_dataloader(&datatest);
        net.train();
        t1.data[0]=1;
        //net.predict(t1).print();
        //产生预测结果保存到csv
        ofstream fout("result.csv");
        fout<<"x,y,Predict,Actual,loss^2"<<endl;
        float tmp1,tmp2;
        for(int i=0;i<100;i++){
            t1.data[0]=i*0.2-10;
            for(int j=0;j<100;j++){
                t1.data[1]=j*0.2-10;
                fout<<t1.data[0]<<","<<t1.data[1]<<",";
                tmp1=net.predict(t1).data[0];
                tmp2=HW2_f4(t1.data[0],t1.data[1]);
                fout<<tmp1<<","<<tmp2<<","<<(tmp1-tmp2)*(tmp1-tmp2)<<endl;
            }
        }
        fout.close();
    }
    catch(const char* msg){
        cout<<"\nERROR: "<<msg<<endl;
    }
    catch(...){
        cout<<"\nERROR: "<<"unknown error"<<endl;
    }
}

void dataset_init(){
    //初始化全局变量数据集，在[-10,10]上均匀取点
    for(int i=0;i<11;i++){
        for(int j=0;j<11;j++){
            train_set[i*11+j][0]=i*2-10;
            train_set[i*11+j][1]=j*2-10;
            train_out[i*11+j]=HW2_f4(train_set[i*11+j][0],train_set[i*11+j][1]);
        }
    }
    //将train_set中的数据打乱
    // int tmp1,tmp2;
    // float tmp3,tmp4;
    // for(int i=0;i<100;i++){
    //     tmp1=rand()%121;
    //     tmp2=rand()%121;
    //     tmp3=train_set[tmp1][0];
    //     tmp4=train_set[tmp1][1];
    //     train_set[tmp1][0]=train_set[tmp2][0];
    //     train_set[tmp1][1]=train_set[tmp2][1];
    //     train_set[tmp2][0]=tmp3;
    //     train_set[tmp2][1]=tmp4;
    //     tmp3=train_out[tmp1];
    //     train_out[tmp1]=train_out[tmp2];
    //     train_out[tmp2]=tmp3;
    // }
    for(int i=0;i<21;i++){
        for(int j=0;j<21;j++){
            test_set[i*21+j][0]=i*1-10;
            test_set[i*21+j][1]=j*1-10;
            test_out[i*21+j]=HW2_f4(test_set[i*21+j][0],test_set[i*21+j][1]);
        }
    }
}

double HW1_showloss(){
    //均匀取点计算loss
    int sz1[2]={1,2};
    Tensor t1(2,sz1);
    double loss=0,tmp;
    for(int i=0;i<21*21;i++){
        t1.data[0]=test_set[i][0];
        t1.data[1]=test_set[i][1];
        tmp=net0->predict(t1).data[0]-test_out[i];
        loss+=tmp*tmp;
    }
    return loss/(21*21);   
}
