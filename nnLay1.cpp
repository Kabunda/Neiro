#include "nnLay1.h"

#define learnRate 0.1
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))

int nnLay::getInCount() 
{
    return in;
}
int nnLay::getOutCount() 
{
    return out; 
}
float** nnLay::getMatrix() 
{
    return matrix; 
}
void nnLay::updMatrix(float* enteredVal)
{
    for (int ou = 0; ou < out; ou++)
    {

        for (int hid = 0; hid < in; hid++)
        {
            matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
        }
        matrix[in][ou] += (learnRate * errors[ou]);
    }
}
void nnLay::setIO(int inputs, int outputs)
{
    in = inputs;        //входные связи
    out = outputs;      //выходные связи

    hidden = new float[out];        //массив выходов
    matrix = new float*[in + 1];    //массив весов 2D

    errors = new float[out];        //массив ошибок

    for (int inp = 0; inp < in + 1; inp++)
    {
        matrix[inp] = new float[out];
    }
    for (int inp = 0; inp < in + 1; inp++)
    {
        for (int outp = 0; outp < out; outp++)
        {
            matrix[inp][outp] = randWeight; //заполняем случайно
        }
    }
}
nnLay::~nnLay() {
    std::cout << "nnLay_Delete\n";
    for (int inp = 0; inp < in + 1; inp++)
    {
        delete matrix[inp];
    }
    delete[] matrix;
    delete[] hidden;  
}
void nnLay::makeHidden(float* inputs)
{
    for (int hid = 0; hid < out; hid++)
    {
        float tmpS = 0.0;
        for (int inp = 0; inp < in; inp++)
        {
            tmpS += inputs[inp] * matrix[inp][hid];
        }
        tmpS += matrix[in][hid];        //нейрон смещения (bias)
        hidden[hid] = sigmoida(tmpS);
    }
}
float* nnLay::getHidden()
{
    return hidden;
}
void nnLay::calcOutError(float* targets)
{
    for (int ou = 0; ou < out; ou++)
    {
        errors[ou] = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);
    }
}
void nnLay::calcHidError(float* targets, float** outWeights, int inS, int outS)
{
    for (int hid = 0; hid < inS; hid++)
    {
        errors[hid] = 0.0;
        for (int ou = 0; ou < outS; ou++)
        {
            errors[hid] += targets[ou] * outWeights[hid][ou];
        }
        errors[hid] *= sigmoidasDerivate(hidden[hid]);
    }
}
float* nnLay::getErrors()
{
    return errors;
}
float nnLay::sigmoida(float val)
{
    return (1.0 / (1.0 + exp(-val)));
}
float nnLay::sigmoidasDerivate(float val)
{
    return (val * (1.0 - val));
}