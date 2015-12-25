#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <Sculptor.h>

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
    virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	
private:
	void drawPoints(C3Tree * tree);
	void drawPoints(List<VertexP> * ps);
	void drawPoints(const Sculptor::ActiveGroup & grp);

private:
	Sculptor * m_sculptor;
    Vector3F * m_pos;
	Vector3F * m_nor;
	Vector3F * m_ref;
	PNPref * m_pool;
	
private slots:
    

};
//! [3]

#endif
