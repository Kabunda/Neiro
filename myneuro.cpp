#include "myneuro.h"

myNeuro::myNeuro()
{
    /*
    //--------многослойный
    inputNeurons = 100;
    outputNeurons = 2;
    nlCount = 4;
    list = (nnLay*)malloc((nlCount) * sizeof(nnLay));

    inputs = (float*)malloc((inputNeurons) * sizeof(float));
    targets = (float*)malloc((outputNeurons) * sizeof(float));

    list[0].setIO(100, 20);
    list[1].setIO(20, 6);
    list[2].setIO(6, 3);
    list[3].setIO(3, 2);
    */
    //--------однослойный---------
    inputNeurons = 2;
    outputNeurons = 1;
    nlCount = 2;
    /*
    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));
    inputs = (float*) malloc((inputNeurons)*sizeof(float));
    targets = (float*) malloc((outputNeurons)*sizeof(float));
    */
    list = new nnLay[nlCount];
    inputs = new float[inputNeurons];
    targets = new float[outputNeurons];

    list[0].setIO(2,2);
    list[1].setIO(2,1);

}
myNeuro::~myNeuro()
{
    //delete targets;
    //delete inputs;
    //delete list;
    std::cout << "Neuro_Delete\n";
}

void myNeuro::feedForwarding(bool ok)
{
    list[0].makeHidden(inputs);
    for (int i = 1; i < nlCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());

    if (!ok)
    {
        std::cout << "\t~\t";
        for (int out = 0; out < outputNeurons; out++)
        {
            std::cout << list[nlCount - 1].hidden[out] << '\t';
        }
        std::cout << std::endl;
        return;
    }
    else
    {
        //printArray(list[0].getErrors(),list[0].getOutCount());
        //printArray(list[1].getErrors(), list[1].getOutCount());
        backPropagate();
    }
}

void myNeuro::backPropagate()
{
    //-------------------------------ERRORS-----CALC---------
    list[nlCount - 1].calcOutError(targets);
    for (int i = nlCount - 2; i >= 0; i--)
        list[i].calcHidError(list[i + 1].getErrors(), list[i + 1].getMatrix(),
            list[i + 1].getInCount(), list[i + 1].getOutCount());

    //-------------------------------UPD-----WEIGHT---------
    for (int i = nlCount - 1; i > 0; i--)
        list[i].updMatrix(list[i - 1].getHidden());
    list[0].updMatrix(inputs);
}

void myNeuro::train(float* in, float* targ)
{
    inputs = in;
    targets = targ;
    feedForwarding(true);
}

void myNeuro::query(float* in)
{
    inputs = in;
    feedForwarding(false);
}

void myNeuro::printArray(float* arr, int s)
{
    std::cout << "_\n";
    for (int inp = 0; inp < s; inp++)
    {
        std::cout << arr[inp]<<"\t";
    }
}