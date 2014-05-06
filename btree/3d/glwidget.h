#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <C3Tree.h>
using namespace sdb;

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
    C3Tree * m_tree;
    V3 * m_pool;
private slots:
    

};
//! [3]

#endif
