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

    list[0].setIO(2,3);
    list[1].setIO(3,1);

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
    //printLay(list, 0);
    for (int i = 1; i < nlCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());

    if (!ok)
    {
        std::cout << "\t~\t";
        for (int out = 0; out < outputNeurons; out++)
        {
            printf("%06.4f\t", list[nlCount - 1].hidden[out]);
            //std::cout << list[nlCount - 1].hidden[out] << '\t';
        }
        std::cout << std::endl;
        return;
    }
    else
    {
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

void myNeuro::printLay(int number) 
{
    /*COORD position;										// Объявление необходимой структуры
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);	// Получение дескриптора устройства стандартного вывода
    position.X = 0;									// Установка координаты X
    position.Y = number*5;									// Установка координаты Y
    SetConsoleCursorPosition(hConsole, position);		// Перемещение каретки по заданным координатам
    */
    std::cout << "Matrix Lay\t" << number << std::endl;
    for (int i = 0; i < list[number].getInCount(); i++) {
        for (int ou = 0; ou < list[number].getOutCount(); ou++) {
            printf("%07.4f\t", list[number].getMatrix()[i][ou]);
        }
        std::cout << std::endl;
    }
}
void myNeuro::printInfo()
{
    std::cout << "NeuroNumLayers:\t" << nlCount << std::endl;
    std::cout << "InputNeurons:\t" << inputNeurons << std::endl;
    for (int i = 0; i < nlCount-1; i++)
        std::cout << "HiddenNeurons:\t" << list[i].getOutCount() << std::endl;
    std::cout << "OutputNeurons:\t" << outputNeurons << std::endl;
    std::cout << "-\t-\t-\t-\t-" << std::endl;
    for (int i = 0; i < nlCount; i++)
    {
        printLay(i);
    }
    std::cout << "-\t-\t-\t-\t-" << std::endl;
}