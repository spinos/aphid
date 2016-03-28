#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include <KdEngine.h>
#include <ConvexShape.h>
#include <IntersectionContext.h>
#include <VoxelGrid.h>

using namespace aphid;

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    
    GLWidget(const std::string & filename, QWidget *parent = 0);
    ~GLWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resizeEvent(QResizeEvent * event);

private:
    void drawBoxes() const;
    void drawTree();
    void drawIntersect();
	KdNTree<cvx::Cube, KdNode4 > * tree();
	bool readTree(const std::string & filename);
	void testTree();
	void testGrid();
	void testIntersect(const Ray * incident);
	void drawActiveSource(const unsigned & iLeaf);
	void drawGrid();
	
private slots:
	
private:
	IntersectionContext m_intersectCtx;
	sdb::VectorArray<cvx::Cube> * m_source;
	KdNTree<cvx::Cube, KdNode4 > * m_tree;
	VoxelGrid<KdNTree<cvx::Cube, KdNode4 >, cvx::Cube > * m_grid;
	int m_treeletColI;
	int m_maxDrawTreeLevel;
};
//! [3]

#endif
