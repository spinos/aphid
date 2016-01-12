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
	void drawPoints(WorldGrid<Array<int, VertexP>, VertexP > * tree);
	void drawPoints(Array<int, VertexP> * ps);
	void drawPoints(const ActiveGroup & grp);

private:
	Sculptor * m_sculptor;
    PNPrefW * m_pool;
	
private slots:
    

};
//! [3]

#endif
