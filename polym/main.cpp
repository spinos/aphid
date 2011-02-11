// pointers to base class
#include <iostream>
using namespace std;

class CPolygon {
  protected:
    int width, height;
  public:
	CPolygon() {cout<<"Polygon\n";}
	virtual ~CPolygon() {cout<<"~Polygon\n";}
    void set_values (int a, int b)
      { width=a; height=b; }
      virtual int area ()
      { return (0); }
  };

class CRectangle: public CPolygon {
  public:
  CRectangle() {cout<<"Polygon::Rectangle\n";}
	~CRectangle() {cout<<"~Polygon::Rectangle\n";}
    int area ()
      { return (width * height); }
  };
  
  class CSquare: public CRectangle {
  public:
  CSquare() {cout<<"Polygon::Rectangle::Rectangle\n";}
	~CSquare() {cout<<"~Polygon::Rectangle::Square\n";}
    int area ()
      { return (width * height); }
  };

class CTriangle: public CPolygon {
  public:
	CTriangle() {cout<<"Polygon::Triangle\n";}
	~CTriangle() {cout<<"~Polygon::Triangle\n";}
    int area ()
      { return (width * height / 2); }
  };
  
void g(CPolygon* poly)
{
	cout << poly->area() << endl;
}

int main () {

  CPolygon * ppoly1 = new CRectangle;
  CPolygon * ppoly2 = new CTriangle;
  CPolygon * ppoly3 = new CSquare;

  
  ppoly1->set_values (4,6);
  ppoly2->set_values (4,6);
  ppoly3->set_values (4,6);
  
  g(ppoly1);
  g(ppoly2);
  g(ppoly3);
  delete ppoly1;
  delete ppoly2;
  delete ppoly3;
  return 0;
  
}
