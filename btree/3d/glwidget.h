#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <C3Tree.h>
#include <RayMarch.h>
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
	virtual void keyPressEvent(QKeyEvent *event);
private:
	void drawPoints(const List<VertexP> * ps);
	bool intersect(const List<VertexP> * ps, const Ray & ray, const float & threshold, List<VertexP> & dst);
private:
	RayMarch m_march;
    C3Tree * m_tree;
    V3 * m_pool;
	Vector3F m_rayBegin, m_rayEnd;
	
private slots:
    

};
//! [3]

#endif
