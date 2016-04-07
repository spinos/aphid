#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include <KdEngine.h>
#include <ConvexShape.h>
#include <IntersectionContext.h>
#include <VoxelGrid.h>

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
	aphid::KdNTree<aphid::cvx::Cube, aphid::KdNode4 > * tree();
	bool readTree(const std::string & filename);
	void testTree();
	void testGrid();
	void testIntersect(const aphid::Ray * incident);
	void drawActiveSource(const unsigned & iLeaf);
	void drawGrid();
	aphid::BoundingBox getFrameBox();
	
private slots:
	
private:
	aphid::IntersectionContext m_intersectCtx;
	aphid::sdb::VectorArray<aphid::cvx::Cube> * m_source;
	aphid::KdNTree<aphid::cvx::Cube, aphid::KdNode4 > * m_tree;
	aphid::VoxelGrid<aphid::cvx::Cube, aphid::KdNode4 > * m_grid;
	int m_treeletColI;
	int m_maxDrawTreeLevel;
};
//! [3]

#endif
