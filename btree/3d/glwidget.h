#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <C3Tree.h>
#include <RayMarch.h>
#include <Ordered.h>
using namespace sdb;

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
	struct ActiveGroup {
		ActiveGroup() { vertices = new Ordered<int, VertexP>; reset(); }
		
		void reset() {
			depthMin = 10e8;
			depthMax = -10e8;
			vertices->clear();
		}
		
		float depthRange() {
			return depthMax - depthMin;
		}
		
		Ordered<int, VertexP> * vertices;
		float depthMin, depthMax, gridSize, threshold;
	};
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
public slots:
	
signals:

protected:
    virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
private:
	void selectPoints(const Ray * incident);
	void deselectPoints();
	void drawPoints(const List<VertexP> * ps);
	void drawPoints(const ActiveGroup & grp);
	bool intersect(List<VertexP> * ps, const Ray & ray, ActiveGroup & dst);
private:
	RayMarch m_march;
	ActiveGroup m_active;
	C3Tree * m_tree;
    V3 * m_pool;
	
private slots:
    

};
//! [3]

#endif
