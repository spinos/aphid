#include <iostream>
using namespace std;

#define ETypeMask ~0x4
#define ELeafOffsetMask ~ETypeMask


inline const char *byte_to_binary(int x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

bool isLeaf(unsigned combined)
{
	return ((combined & ~ETypeMask) > 0); 
}

void setLeaf(bool is, unsigned& combined)
{
	combined = (is) ? (combined | ~ETypeMask):(combined & ETypeMask);
}

void setOffset(unsigned offset, unsigned& combined)
{
    combined = (offset << 3 ) | ELeafOffsetMask;
}

unsigned getOffset(unsigned & combined)
{
    return (combined & ~ELeafOffsetMask) >> 3;
}


int main()
{
    cout<<"byte operation test\n";
    
    cout<<"  ETypeMask "<<byte_to_binary(ETypeMask)<<"\n";
    cout<<"          6 "<<byte_to_binary(6)<<"\n";
    cout<<"  masked  6 "<<byte_to_binary(6 & ~ETypeMask)<<"\n";
    
    unsigned n = 4561;
    cout<<"       4561 "<<byte_to_binary(n)<<"\n";
    n = n<<3;
    cout<<"    4561<<3 "<<byte_to_binary(n)<<"\n";
    n = n | ~ETypeMask;
    cout<<"       n| ~m"<<byte_to_binary(n)<<"\n";
    n = n & ETypeMask;
    cout<<"       n & m"<<byte_to_binary(n)<<"\n";
    n = n >> 3;
    cout<<" n>>3       "<<byte_to_binary(n)<<"\n";
    unsigned t = 6;
    if(isLeaf(t)) cout<<" "<<t<<" is leaf\n";
    
    setOffset(45691, t);
    
    cout<<" set offset "<<t<<"\n";
    
    unsigned off = getOffset(t);
    
    cout<<" get offset "<<off<<"\n";
    
    setLeaf(1, t);
    if(isLeaf(t)) cout<<" "<<t<<" is leaf\n";
    else cout<<" "<<t<<" is no leaf\n";
    
    off = getOffset(t);
    
    cout<<" get offset "<<off<<"\n";
    
    
    return 0;
}

