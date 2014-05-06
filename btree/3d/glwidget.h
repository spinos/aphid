#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>

//! [0]
class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
public slots:
	
signals:

protected:
    
private:

private slots:
    

};
//! [3]

#endif
