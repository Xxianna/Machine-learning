//this file is the main header file of the library qwqtensor
//______________________________________________________
//-------------THE ONLY AUTHOR ：Xxianna-----------------
//______________________________________________________

//本库设计了tensor类，用于存储多维数组，支持多种运算（SSE4.2）

#ifndef LIBQWQTENSOR_HPP
#define LIBQWQTENSOR_HPP

#include <iostream>
#include <memory>
//打开openmp支持
#include <omp.h>

namespace qwqnn
{


class Tensor
{
public:
    Tensor();//没啥用，但是可以用作被赋值的对象
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);//移动构造函数，可以直接对将亡值赋值，C++11新特性
    Tensor(int dim, int* shape);
    Tensor(int dim, int* shape, float* data);
    ~Tensor();

    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other);//解决将亡值问题
    bool operator==(const Tensor& other) const;//此函数要求全等。一般使用要求近似相等，请自行编写
    bool operator!=(const Tensor& other) const;

    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);

    Tensor operator+(float other);
    Tensor operator-(float other);
    Tensor operator*(float other);
    Tensor operator/(float other);

    Tensor& operator+=(float other);
    Tensor& operator-=(float other);
    Tensor& operator*=(float other);
    Tensor& operator/=(float other);

    Tensor operator-();

    Tensor operator[](int index);//注意！这种方法会返回一个新的tensor，而不是原tensor的一个引用
    //要获得原tensor的引用，请使用at函数（不安全）
    Tensor& unsqueeze(int dim);//在dim维度上增加一个维度（shape需要是1）
    Tensor& squeeze(int dim);//在dim维度上去除一个维度（shape需要是1）
    void reshape(int dim, int* shape);
    int size(int dim=-1) const;//返回维度dim的大小，dim=-1时返回总大小

    void print();
    float* at(int pos_len,int* pos);//直接支出数组指针。不安全但易用的修改窗口

    //目前是public的，但其实可以改成private的，请不要直接访问
    int dim;
    int* shape;
    float* data;

    //其他
    Tensor mul(const Tensor& other);//矩阵叉乘
    Tensor transpose() const;//转置
    double sum();//求和
    double mean();//求均值
    Tensor divide();//求倒数
};

}
#endif // LIBQWQTENSOR_HPP