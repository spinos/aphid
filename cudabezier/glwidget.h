#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <BezierProgram.h>
#include <BaseCurve.h>

class CUDABuffer;

class GLWidget : public Base3DView
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
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	
private:
	
private:
	CUDABuffer * m_vertexBuffer;
	CUDABuffer * m_cvs;
	BezierProgram * m_program;
	BaseCurve * m_curve;
private slots:
    

};
//! [3]

#endif
