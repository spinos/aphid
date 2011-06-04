// pointers to base class
#include <iostream>
using namespace std;

class A {
  public:
	A() {cout<<"initialize A\n";}
	virtual ~A() {cout<<"destroy A\n";}
	static int pref;
  };
  
int  A::pref;

class B: public A {
  public:
	B() {cout<<"initialize B\n";}
	virtual ~B() {cout<<"destroy B\n";}
  };
  
class C: public B {
  public:
	C() {cout<<"initialize C\n";}
	virtual ~C() {cout<<"destroy C\n";}
  };
  
void g(const char *note, A &data)
{
    cout<<" "<<note<<data.pref<<endl;
}

int main () {
    
  A::pref = 97;
  
  cout<<"A::pref "<<A::pref<<endl;
  cout<<"B::pref "<<B::pref<<endl;
  cout<<"C::pref "<<C::pref<<endl;
  
  A ppoly0;
  B ppoly1;
  C ppoly2;
  
  ppoly1.pref = 83; 
  
  g("ppoly0.pref ", ppoly0);
  g("ppoly1.pref ", ppoly1);
  g("ppoly2.pref ", ppoly2);
  
  C::pref = 71; 
  
  cout<<"A::pref "<<A::pref<<endl;
  cout<<"B::pref "<<B::pref<<endl;
  cout<<"C::pref "<<C::pref<<endl;
  
  B::pref = -59;
  
  g("ppoly0.pref ", ppoly0);
  g("ppoly1.pref ", ppoly1);
  g("ppoly2.pref ", ppoly2);
  
  return 0;
  
}
