#ifndef CUSTOMER_H
#define CUSTOMER_H

#include <QtGui>

class HWidget : public QWidget
{
public:
	HWidget();
	virtual ~HWidget();
	virtual void setAttribute(const char *attributeName, int value);
};

class Hoatzin
{
public:
	Hoatzin();
	
	int set_attribute_int(const char *widgetName, const char *attributeName, int value);
	
private:
	
};
#endif        //  #ifndef CUSTOMER_H

