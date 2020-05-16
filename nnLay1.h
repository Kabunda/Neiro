#pragma once
#include <iostream>
class nnLay
{
public:
    ~nnLay();
    int in;
    int out;
    float** matrix;
    float* hidden;
    float* errors;

    int getInCount();
    int getOutCount();
    float** getMatrix();
    void updMatrix(float* enteredVal);
    void setIO(int inputs, int outputs);
    void makeHidden(float* inputs);
    float* getHidden();
    void calcOutError(float* targets);
    void calcHidError(float* targets, float** outWeights, int inS, int outS);
    float* getErrors();
    float sigmoida(float val);
    float sigmoidasDerivate(float val);
};