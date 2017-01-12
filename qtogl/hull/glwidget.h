#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>

class HullContainer;

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

public slots:
	
signals:

protected:
    virtual void clientInit();
    virtual void clientDraw();
		
private slots:
    
private:	
	HullContainer* _dynamics;

};

#endif
