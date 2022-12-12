#ifndef QWQNN_INFO_LIST
#define QWQNN_INFO_LIST
//本文件不能include多次
#include "libqwqnn.hpp"

namespace qwqnn{
    activ_info relu_info={"relu",relu,relu_derivative};
    activ_info sigmoid_info={"sigmoid",sigmoid,sigmoid_derivative};
    activ_info tanh_info={"tanh",tanh,tanh_derivative};
    activ_info softmax_info={"softmax",softmax,softmax_derivative};

}
#else
#error "just include this file once"
#endif