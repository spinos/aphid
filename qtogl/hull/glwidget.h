#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <qt/Base3DView.h>

namespace aphid {
class ATriangleMesh;
}

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
	aphid::ATriangleMesh * m_tri;
	
};

#endif
