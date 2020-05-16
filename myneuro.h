#ifndef MYNEURO_H
#define MYNEURO_H
#include <iostream>
#include <math.h>
#include <Windows.h>  // для вывода на консоль с координатами
#include "nnLay1.h"
//#define learnRate 0.1
//#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))
class myNeuro
{
public:
    myNeuro();
    ~myNeuro();
    void feedForwarding(bool ok);
    void backPropagate();
    void train(float* in, float* targ);
    void query(float* in);
    void printArray(float* arr, int s);
    void printLay(int number);
private:
    nnLay* list;
    int inputNeurons;
    int outputNeurons;
    int nlCount;

    float* inputs;
    float* targets;
};

#endif // MYNEURO_H