//硬件运算加速库
#ifndef ACCELERATOR_H
#define ACCELERATOR_H

//#define __SSE4_2__
#ifdef __SSE4_2__
#include <immintrin.h> //否则会破坏__m128的定义
#endif
#include <omp.h>

//定义__m128的加法，否则无法openmp加速
#ifdef __SSE4_2__
//根据#pragma omp declare reduction(MyAdd: MyClass: omp_out += omp_in) initializer(omp_priv=MyClass(0)) 语句，定义__m128的加法
#pragma omp declare reduction(MyAdd: __m128: omp_out = _mm_add_ps(omp_out, omp_in)) initializer(omp_priv=_mm_setzero_ps())
#endif

//需要加速的主要是乘法。sse4只能给出4个float的乘法
#ifdef __SSE4_2__
inline void sse4_mul(float* a, float* b, float* c)
{
    __m128 a0 = _mm_loadu_ps(a);
    __m128 b0 = _mm_loadu_ps(b);
    __m128 c0 = _mm_mul_ps(a0, b0);
    _mm_storeu_ps(c, c0);
}
#endif

inline void float4mul(float* a, float* b, float* c)
{
#ifndef __SSE4_2__
    c[0] = a[0] * b[0];
    c[1] = a[1] * b[1];
    c[2] = a[2] * b[2];
    c[3] = a[3] * b[3];
#else
    sse4_mul(a, b, c);
#endif
}

//对两个向量进行点乘
inline void floatsmul(float* a, float* b, float* c,int n)
{
    //每四个一组。openmp并行
    int i;
    #pragma omp parallel for private(i)
    for (i=0; i <= n-4; i += 4)
    {
        float4mul(a + i, b + i, c + i);
    }
    //剩余的
    for (i=n-n%4; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

//此外，点乘可以进行优化，因为sse4有对应的指令
inline float floatsdot(float* a, float* b, int n)
{
#ifndef __SSE4_2__
    float sum = 0;
    //每四个一组。openmp并行
    int i;float tool[4];
    #pragma omp parallel for private(i) reduction(+ : sum)
    for (i=0; i <= n-4; i += 4)
    {
        float4mul(a + i, b + i, tool);
        sum += tool[0] + tool[1] + tool[2] + tool[3];
        //这里直接用累加即可。reduction记录的是每次的操作，而不是每次的结果
    }
    //剩余的
    for (i=n-n%4; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
#else
    __m128 sum = _mm_setzero_ps();
    //每四个一组。openmp并行
    int i;
    #pragma omp parallel for reduction(MyAdd : sum) private(i)
    for (i = 0; i <= n - 4; i += 4)
    {
        __m128 a0 = _mm_loadu_ps(a + i);
        __m128 b0 = _mm_loadu_ps(b + i);
        __m128 c0 = _mm_mul_ps(a0, b0);
        sum = _mm_add_ps(sum, c0);
    }
    //导出
    float* sumf = (float*)&sum;
    float sum0 = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    //剩余的
    for (i=n-n%4; i < n; i++)
    {
        sum0 += a[i] * b[i];
    }
    return sum0;
#endif
}

//对长单线向量进行求和
inline float floatsum_long(float* a, int n)
{
#ifndef __SSE4_2__
    int i;float sum = 0;
    #pragma omp parallel for private(i) reduction(+ : sum)
    for (i=0; i <= n - 4; i += 4)
    {
        sum += a[i];
        sum += a[i + 1];
        sum += a[i + 2];
        sum += a[i + 3];
    }
    //剩余的
    for (i=n-n%4; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
#else
    __m128 sum = _mm_setzero_ps();
    //每四个一组。openmp并行
    int i;
    #pragma omp parallel for private(i) reduction(MyAdd : sum)
    for (i=0; i <= n - 4; i += 4)
    {
        __m128 a0 = _mm_loadu_ps(a + i);
        sum = _mm_add_ps(sum, a0);
    }
    //导出
    float* sumf = (float*)&sum;
    float sum0 = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    //剩余的
    for (i=n-n%4; i < n; i++)
    {
        sum0 += a[i];
    }
    return sum0;
#endif
}


#endif
