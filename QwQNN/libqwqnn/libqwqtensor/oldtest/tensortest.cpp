//测试accelerator.h中的所有函数
//测试tensor类中的所有函数

#include <iostream>

//#define __SSE4_2__
//经过测试，并不需要定义__SSE4_2__，因为“-msse4”已经定义了
#include "accelerator.h"

#include "libqwqtensor.hpp"
//#pragma comment(lib, "libqwqtensor.lib")
//上面这是MSVC的写法，gcc的写法是只能在MAKEFILE中写

void test_accelerator();
void test_tensor();

int main(){
    // printf("test_accelerator:\n");
    // test_accelerator();
    printf("test_tensor:\n");
    test_tensor();
    return 0;
}

void test_tensor(){
    //首先测试构造
    Tensor a;
    int shape[3] = {2,3,4},sp[2]={6,4};
    Tensor b(3, shape);
    for(int i=0;i<b.size();i++){
        b.data[i] = i;
    }
    a=b*0.5+1;
    printf("a:\n");
    for(int i=0;i<a.size();i++){
        printf("%f ", a.data[i]);
    }
    a+=b;
    printf("\na+b:\n");
    for(int i=0;i<a.size();i++){
        printf("%f ", a.data[i]);
    }
    a-=-a;
    printf("\n2*a:\n");
    for(int i=0;i<a.size();i++){
        printf("%f ", a.data[i]);
    }
    printf("\na.dim = %d\n", a.dim);
    printf("a.shape = (%d, %d, %d)\n", a.shape[0], a.shape[1], a.shape[2]);
    a.reshape(2, sp);
    printf("\na.dim = %d\n", a.dim);
    printf("a.shape = (%d, %d)\n", a.shape[0], a.shape[1]);
    a.unsqueeze(1);
    printf("\na.dim = %d\n", a.dim);
    printf("a.shape = (%d, %d, %d)\n", a.shape[0], a.shape[1], a.shape[2]);
    a.squeeze(1);

    Tensor c=a.transpose();
    printf("\nc.dim = %d\n", c.dim);
    printf("c.shape = (%d, %d)\n", c.shape[0], c.shape[1]);
    //按照6*4的顺序输出
    printf("\nc:\n");
    for(int i=0;i<c.size(0);i++){
        for(int j=0;j<c.size(1);j++){
            printf("%f ", c.data[i*c.size(1)+j]);
        }
        printf("\n");
    }
    auto d=a.mul(c);
    auto e=c.mul(a);
    printf("\nd:\n");
    for(int i=0;i<d.size(0);i++){
        for(int j=0;j<d.size(1);j++){
            printf("%f ", d.data[i*d.size(1)+j]);
        }
        printf("\n");
    }
    printf("\ne:\n");
    for(int i=0;i<e.size(0);i++){
        for(int j=0;j<e.size(1);j++){
            printf("%f ", e.data[i*e.size(1)+j]);
        }
        printf("\n");
    }
    e=e.divide();
    printf("\ne:\n");
    for(int i=0;i<e.size(0);i++){
        for(int j=0;j<e.size(1);j++){
            printf("%f ", e.data[i*e.size(1)+j]);
        }
        printf("\n");
    }
    printf("\ne.sum = %f",e.sum());
    printf("\ne.mean = %f",e.mean());

    Tensor gete=e[1];
    printf("\ngete dim = %d", gete.dim);
    printf("\ngete:\n");
    for(int i=0;i<gete.size(0);i++){
        printf("%f ", gete.data[i]);
    }
    Tensor getget=e[1][1];
    printf("\ngetget: %f", getget.data[0]);
    int pos[2]={1,1};
    float *atget=e.at(2,pos);
    printf("\natget: %f %f", atget[0], atget[1]);
    atget[0]=100;
    gete=e[1];
    printf("\ngete:\n");
    for(int i=0;i<gete.size(0);i++){
        printf("%f ", gete.data[i]);
    }
    //截至目前，只有print未测试
    //测试print
    e.print();
    d.print();
    b.print();
    a[1].print();
    e[1][1].print();
}

void test_accelerator(){
#ifdef __SSE4_2__
    printf("__SSE4_2__ defined\n");
    float a[4] = {1,2,3,4}, b[4] = {5,6,7,8}, c[4];
    float4mul(a, b, c);
    printf("c[0] = %f, c[1] = %f, c[2] = %f, c[3] = %f\n", c[0], c[1], c[2], c[3]);

    float d[7]={1.1,1.2,1.3,1.4,1.5,1.6,1.7}, e[7]={2.1,2.2,2.3,2.4,2.5,2.6,2.7}, f[7];
    floatsmul(d, e, f, 6);// 6 just for test
    printf("f[0] = %f, f[1] = %f, f[2] = %f, f[3] = %f, f[4] = %f, f[5] = %f, f[6] = %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6]);

    floatsmul(d, e, f, 7);
    float fa=floatsum_long(f, 7);
    float fb=floatsdot(d, e, 7);
    printf("floatsum_long(f, 7) = %f, floatsdot(d, e, 7) = %f\n", fa, fb);

    float fc=0;
    for(int i=0; i<7; i++){
        fc+=d[i]*e[i];
    }
    printf("dot(d, e, 7) = %f\n", fc);
    
    int tail=6;
    printf("floatsum_long(f, %d) = %f\n", tail,floatsum_long(f, tail));
    printf("floatsdot(d, e, %d) = %f\n",tail, floatsdot(d, e, tail));
    fc=0;
    for(int i=0; i<tail; i++){
        fc+=d[i]*e[i];
    }
    printf("dot(d, e, %d) = %f\n",tail, fc);

    printf("Congratulations! All functions in accelerator.h are tested!\n");
    printf("I have check all the output, and they are all correct!\n\n");
#else
    printf("__SSE4_2__ not defined\n");
    float a[4] = {1,2,3,4}, b[4] = {5,6,7,8}, c[4];
    float4mul(a, b, c);
    printf("c[0] = %f, c[1] = %f, c[2] = %f, c[3] = %f\n", c[0], c[1], c[2], c[3]);

    float d[7]={1.1,1.2,1.3,1.4,1.5,1.6,1.7}, e[7]={2.1,2.2,2.3,2.4,2.5,2.6,2.7}, f[7];
    floatsmul(d, e, f, 6);// 6 just for test
    printf("f[0] = %f, f[1] = %f, f[2] = %f, f[3] = %f, f[4] = %f, f[5] = %f, f[6] = %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6]);

    floatsmul(d, e, f, 7);
    float fa=floatsum_long(f, 7);
    float fb=floatsdot(d, e, 7);
    printf("floatsum_long(f, 7) = %f, floatsdot(d, e, 7) = %f\n", fa, fb);

    float fc=0;
    for(int i=0; i<7; i++){
        fc+=d[i]*e[i];
    }
    printf("dot(d, e, 7) = %f\n", fc);
    
    int tail=6;
    printf("floatsum_long(f, %d) = %f\n", tail,floatsum_long(f, tail));
    printf("floatsdot(d, e, %d) = %f\n",tail, floatsdot(d, e, tail));
    fc=0;
    for(int i=0; i<tail; i++){
        fc+=d[i]*e[i];
    }
    printf("dot(d, e, %d) = %f\n",tail, fc);

    printf("Congratulations! All functions in accelerator.h are tested!\n");
    printf("I have check all the output, and they are all correct!\n\n");

#endif
}