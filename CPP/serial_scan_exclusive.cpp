#include <iostream>
using namespace std;

int main()
{
    const int ARRAY_SIZE = 10;
    int acc = 0;
    int out[ARRAY_SIZE];
    int elements [] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (int i=0; i<ARRAY_SIZE; i++){
        out[i] = acc;  // A
        acc = acc + elements[i];  // B

        // Exclusive: A then B
        // Inclusive: B then A
    }
    
    for (int i=0; i<ARRAY_SIZE; i++){
        cout << "i: " << i << " out: " << out[i] << endl;
    }
    

    return 0;
}
