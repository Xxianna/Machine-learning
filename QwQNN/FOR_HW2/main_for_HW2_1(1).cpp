#include <iostream>
#include <fstream>
#include ".\libqwqnn\libqwqnn.hpp"
#include ".\libqwqnn\qwqnn_info_list.hpp"
using namespace std;
using namespace qwqnn;

#define test(x)  ((2*(1-x+2*x*x)*exp(-x*x/2))/5)
//#define HW2_1(x)    ((2*(1-x+2*x*x)*exp(-x*x/2))/5)
//注意c++对sinx/x这种不会在0处取极限的
// #define HW2_f4(x,y) ((x==0?1:(sin(x)/x))*(y==0?1:(sin(y)/y)))
// #define HW2_f5(x,y) (100*(y-x*x)*(y-x*x)+(1-x)*(1-x))

#define sample 40
class getdatatest:public qwqnn::DataLoader{
public:
    float iner=8.0/(sample+1);
    float yy[sample]={0};char flag[sample]={0};
    int i=0,atpos[1]={0};
    bool get(Tensor* input,Tensor* output){
        if(i>=sample){
            i=0;
            return true;
        }
        float x=i*iner-4;
        if(flag[i]==0){
            yy[i]=test(x)+gaussian(0,0.02);
            flag[i]=1;
        }
        *(input->at(1,atpos))=x;
        *(output->at(1,atpos))=yy[i];
        i++;
        return false;        
    }
};
Network *net0;
double HW1_showloss();
int main(){
    //try下面所有语句，所有报错接收后直接输出其报错信息
    try{
        show_loss_struct HW1_loss;//显示MSE用
        HW1_loss.interval=20;
        HW1_loss.func=HW1_showloss;
        int sz1[2]={1,1},sz2[2]={1,16};
        Network net;net0=&net;
        Tensor t1(2,sz1),t2(2,sz2);
        BiasLayer b1(t2);
        BiasLayer b2(t1);
        LinearLayer l1(t1,t2),l2(t2,t1);
        ActivationLayer a1(sigmoid_info);
        net.add_layer(&l1).add_layer(&b1).add_layer(&a1).add_layer(&l2).add_layer(&b2);
        Linearloss loss;//直接求差作为损失函数
        Momentum opt;//加入动量的梯度下降
        opt.set_decay(0.9996).set_learning_rate(0.08).set_momentum(0.6);
        net.set_batch_size(1).set_input(t1).set_output(t1);
        net.set_lossfunc(&loss);
        net.set_optimizer(&opt);
        net.set_epoch(3000).set_show_loss(HW1_loss);//最终mse为0.00107
        getdatatest datatest;
        net.set_dataloader(&datatest);
        net.train();
        //b1.WEIGHT.print();
        // b2.WEIGHT.print();
        t1.data[0]=1;
        net.predict(t1).print();
        //产生预测结果保存到csv
        ofstream fout("result.csv");
        fout<<"x,Predict,Actual"<<endl;
        for(int i=0;i<400;i++){
            t1.data[0]=i*0.02-4;
            fout<<t1.data[0]<<","<<net.predict(t1).data[0]<<","<<test(t1.data[0])<<endl;
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

double HW1_showloss(){
    //均匀取点计算loss
    int sz1[2]={1,1};
    Tensor t1(2,sz1);
    double loss=0,tmp;
    for(int i=0;i<160;i++){
        t1.data[0]=i*0.05-4;
        tmp=net0->predict(t1).data[0]-test(t1.data[0]);
        loss+=tmp*tmp;
    }
    return loss/160;    
}
