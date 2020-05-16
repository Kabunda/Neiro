#include "myneuro.h"

int main(int argc, char* argv[])
{
    //myNeuro* bb = new myNeuro();
    myNeuro bb;
    //----------------------------------INPUTS----GENERATOR-------------
    //qsrand((QTime::currentTime().second()));
    /*float* abc = new float[100];
    for (int i = 0; i < 100; i++)
    {
        abc[i] = (rand() % 98) * 0.01 + 0.01;
    }

    float* cba = new float[100];
    for (int i = 0; i < 100; i++)
    {
        cba[i] = (rand() % 98) * 0.01 + 0.01;
    }*/
    float input1[2] = { 0,0 };
    float input2[2] = { 0,1 };
    float input3[2] = { 1,0 };
    float input4[2] = { 1,1 };
    //---------------------------------TARGETS----GENERATOR-------------
    /*float* tar1 = new float[2];
    tar1[0] = 0.22;
    tar1[1] = 0.88;
    float* tar2 = new float[2];
    tar2[0] = 0.77;
    tar2[1] = 0.15;*/
    float output0[1] = { 0 };
    float output1[1] = { 1 };

    //bb.printInfo();

    int i = 0;
    while (i < 100000)
    {
        bb.train(input1, output0);
        bb.train(input2, output1);
        bb.train(input3, output1);
        bb.train(input4, output0);
        i++;
    }

    bb.printInfo();
    std::cout << "_RESULT_XOR_" << std::endl;
    std::cout << "_0_0_";
    bb.query(input1);
    std::cout << "_0_1_";
    bb.query(input2);
    std::cout << "_1_0_";
    bb.query(input3);
    std::cout << "_1_1_";
    bb.query(input4);

    return 0;
}