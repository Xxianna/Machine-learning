//实现tensor类

#include "libqwqtensor.hpp"
#define __windows__
#include "cmdsheet.h"
//#define __SSE4_2__
//不需要这句。链接自动的
#include "accelerator.h"

namespace qwqnn
{
Tensor::Tensor()
{
    dim = 0;
    shape = nullptr;
    data = nullptr;
}

Tensor::Tensor(const Tensor& other)
{
    dim = other.dim;
    shape = new int[dim];
    memcpy(shape, other.shape, sizeof(int) * dim);
    data = new float[other.size()];
    memcpy(data, other.data, sizeof(float) * other.size());
}

Tensor::Tensor(Tensor&& other)
{
    dim = other.dim;
    shape = other.shape;
    data = other.data;
    other.dim = 0;
    other.shape = nullptr;
    other.data = nullptr;
}

Tensor::Tensor(int dim, int* shape)
{
    this->dim = dim;
    this->shape = new int[dim];
    memcpy(this->shape, shape, sizeof(int) * dim);
    data = new float[size()];
}

Tensor::Tensor(int dim, int* shape, float* data)
{
    this->dim = dim;
    this->shape = new int[dim];
    memcpy(this->shape, shape, sizeof(int) * dim);
    this->data = new float[size()];
    memcpy(this->data, data, sizeof(float) * size());
}

Tensor::~Tensor()
{
    if (shape != nullptr)
    {
        delete[] shape;
        shape = nullptr;
    }
    if (data != nullptr)
    {
        delete[] data;
        data = nullptr;
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (shape != nullptr)
    {
        delete[] shape;
        shape = nullptr;
    }
    if (data != nullptr)
    {
        delete[] data;
        data = nullptr;
    }
    dim = other.dim;
    shape = new int[dim];
    memcpy(shape, other.shape, sizeof(int) * dim);
    data = new float[other.size()];
    memcpy(data, other.data, sizeof(float) * other.size());
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if (shape != nullptr)
    {
        delete[] shape;
        shape = nullptr;
    }
    if (data != nullptr)
    {
        delete[] data;
        data = nullptr;
    }
    dim = other.dim;
    shape = other.shape;
    data = other.data;
    other.dim = 0;
    other.shape = nullptr;
    other.data = nullptr;
    return *this;
}

bool Tensor::operator==(const Tensor& other) const
{
    if (dim != other.dim)
    {
        return false;
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            return false;
        }
    }
    for (int i = 0; i < size(); i++)
    {
        if (data[i] != other.data[i])
        {
            return false;
        }
    }
    return true;
}

bool Tensor::operator!=(const Tensor& other) const
{
    return !(*this == other);
}

Tensor Tensor::operator+(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator+: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator+: shape not equal";
        }
    }
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i=0; i < size(); i++)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator-: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator-: shape not equal";
        }
    }
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator*: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator*: shape not equal";
        }
    }
    Tensor result(dim, shape);
    //使用accelerator加速
    floatsmul(data, other.data, result.data, size());
    return result;
}

Tensor& Tensor::operator+=(float other){
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] += other;
    }
    return *this;
}

Tensor& Tensor::operator-=(float other){
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] -= other;
    }
    return *this;
}

Tensor& Tensor::operator*=(float other){
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] *= other;
    }
    return *this;
}

Tensor& Tensor::operator/=(float other){
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] /= other;
    }
    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator+=: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator+=: shape not equal";
        }
    }
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator-=: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator-=: shape not equal";
        }
    }
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other)
{
    if (dim != other.dim)
    {
        throw "Tensor::operator*=: dim not equal";
    }
    for (int i = 0; i < dim; i++)
    {
        if (shape[i] != other.shape[i])
        {
            throw "Tensor::operator*=: shape not equal";
        }
    }
    //使用accelerator加速
    float *temp = new float[size()];
    floatsmul(data, other.data, temp, size());
    delete[] data;
    data = temp;
    return *this;
}

Tensor Tensor::operator+(float other)
{
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = data[i] + other;
    }
    return result;
}

Tensor Tensor::operator-(float other)
{
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = data[i] - other;
    }
    return result;
}

Tensor Tensor::operator*(float other)
{
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = data[i] * other;
    }
    return result;
}

Tensor Tensor::operator/(float other)
{
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = data[i] / other;
    }
    return result;
}

Tensor Tensor::operator-()
{
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = -data[i];
    }
    return result;
}

Tensor Tensor::operator[](int index)
{
    if (dim == 0)
    {
        throw "Tensor::operator[]: dim not match";
    }
    if (index >= shape[0])
    {
        throw "Tensor::operator[]: index not match";
    }
    //如果是一维数组，直接返回单个元素的tensor
    if (dim == 1)
    {
        int shape[1] = {1};
        Tensor result(1, shape);
        result.data[0] = data[index];
        return result;
    }
    int new_dim = dim - 1;
    int *new_shape = new int[new_dim];
    for (int i = 0; i < new_dim; i++)
    {
        new_shape[i] = shape[i + 1];
    }
    Tensor result(new_dim, new_shape);
    delete[] new_shape;
    memcpy(result.data, data + index * result.size(), result.size() * sizeof(float));
    return result;
}

void Tensor::reshape(int new_dim, int *new_shape)
{
    if (new_dim == 0)
    {
        throw "Tensor::reshape: new_dim not match";
    }
    int new_size = 1;
    for (int i = 0; i < new_dim; i++)
    {
        if (new_shape[i] <= 0)
        {
            throw "Tensor::reshape: new_shape not match";
        }
        new_size *= new_shape[i];
    }
    if (new_size != size())
    {
        throw "Tensor::reshape: size not match";
    }
    dim = new_dim;
    delete[] shape;
    shape = new int[dim];
    memcpy(shape, new_shape, dim * sizeof(int));
}

Tensor Tensor::transpose() const
{
    //转置只能是二维
    if (dim != 2)
    {
        throw "Tensor::transpose: dim not match";
    }
    int new_shape[2] = {shape[1], shape[0]};
    Tensor result(2, new_shape);
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < shape[0]; i++)
    {
        for (j = 0; j < shape[1]; j++)
        {
            result.data[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }
    return result;
}

Tensor Tensor::divide(){
    Tensor result(dim, shape);
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size(); i++)
    {
        result.data[i] = 1 / data[i];
    }
    return result;
}

double Tensor::sum(){
    return floatsum_long(data, size());
}

double Tensor::mean(){
    return sum() / size();
}

int Tensor::size(int wdim) const
{
    if(wdim==-1){
        //返回总大小
        int result = 1;
        for (int i = 0; i < dim; i++)
        {
            result *= shape[i];
        }
        return result;
    }else return shape[wdim];
}

Tensor Tensor::mul(const Tensor& other)
{
    if (dim != 2 || other.dim != 2)
    {
        throw "Tensor::dot: dim not match: dim must be 2";
    }
    if (shape[1] != other.shape[0])
    {
        printf("usr try to dot %d*%d and %d*%d\n", shape[0], shape[1], other.shape[0], other.shape[1]);
        throw "Tensor::dot: shape not match";
    }
    int new_shape[2] = {shape[0], other.shape[1]};
    Tensor result(2, new_shape);
    Tensor other_t = other.transpose();
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < shape[0]; i++)
    {
        for (j = 0; j < other.shape[1]; j++)
        {
            result.data[i * other.shape[1] + j] = floatsdot(data + i * shape[1], other_t.data + j * other_t.shape[1], shape[1]);
        }
    }
    return result;
}

Tensor& Tensor::squeeze(int wdim){
    //如果shape[wdim]==1，那么就把这个维度去掉
    if(shape[wdim]!=1) throw "Tensor::squeeze: shape not 1";
    int new_dim=dim-1;
    int *new_shape=new int[new_dim];
    memcpy(new_shape, shape, wdim*sizeof(int));
    memcpy(new_shape+wdim, shape+wdim+1, (dim-wdim-1)*sizeof(int));
    delete[] shape;
    shape=new_shape;
    dim=new_dim;
    return *this;
}

Tensor& Tensor::unsqueeze(int wdim){
    //在wdim维度上增加一个维度，这个维度的大小为1
    int new_dim=dim+1;
    int *new_shape=new int[new_dim];
    memcpy(new_shape, shape, wdim*sizeof(int));
    new_shape[wdim]=1;
    memcpy(new_shape+wdim+1, shape+wdim, (dim-wdim)*sizeof(int));
    delete[] shape;
    shape=new_shape;
    dim=new_dim;
    return *this;
}

void Tensor::print(){
    printf("\nTensor dim=%d, shape=[", dim);
    for(int i=0;i<dim;i++) printf("%d ", shape[i]);
    printf("]\n");
    //主要针对1 2 3维的tensor
    if(dim==1){
        printf("[");
        for(int i=0;i<shape[0];i++) printf("%10.4f\t", data[i]);
        printf("\b]\n");
    }else if(dim==2){
        printf("[");
        for(int i=0;i<shape[0];i++){
            printf("[");
            for(int j=0;j<shape[1];j++) printf("%10.4f\t", data[i*shape[1]+j]);
            if(i<shape[0]-1)printf("\b],\n ");
            else printf("\b]");
        }
        printf("]\n");
    }else if(dim==3){
        printf("[");
        for(int i=0;i<shape[0];i++){
            printf("[");
            for(int j=0;j<shape[1];j++){
                printf("[");
                for(int k=0;k<shape[2];k++) printf("%10.4f\t", data[i*shape[1]*shape[2]+j*shape[2]+k]);
                if(j<shape[1]-1)printf("\b],\n  ");
                else printf("\b]");
            }
            if(i<shape[0]-1)printf("],\n ");
            else printf("]");
        }
        printf("]\n");
    }else{
        printf("\n[A tensor with dim>3 here]\n");return;
    }
}

float* Tensor::at(int pos_len,int* pos){
    if(pos_len>dim) throw "Tensor::at: pos_len dim not match";
    int offset=0;
    for(int i=0;i<pos_len;i++){
        if(pos[i]>=shape[i]) throw "Tensor::at: pos shape not match";
        offset+=pos[i]*(i+1==dim?1:shape[i+1]);
    }
    return data+offset;
}



} // namespace tensor