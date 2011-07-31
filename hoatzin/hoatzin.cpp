#include "Hoatzin.h"
#include <QtCore>
#include <QtGui>
#include <QtDebug>
#include <iostream>
using namespace std;

HWidget::HWidget() {}

HWidget::~HWidget() {}

void HWidget::setAttribute(const char *attributeName, int value)
{
	qDebug()<<"beep";
}

Hoatzin::Hoatzin()
{
}

int Hoatzin::set_attribute_int(const char *widgetName, const char *attributeName, int value)
{
	foreach (QWidget *widget, QApplication::allWidgets())
    {
            if(widget->objectName() == QString(widgetName))
            {
                HWidget *h = (HWidget *)widget;
				h->setAttribute(attributeName, value);
				return 1;		
			}
            
    }
	return 0;
}

