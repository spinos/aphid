#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include <KdNTree.h>

class TestBox : public BoundingBox
{
public:
    TestBox() {}
    virtual ~TestBox() {}
    BoundingBox calculateBBox() const
    { return * this; }
    BoundingBox bbox() const
    { return * this; }
};

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    
//! [3]
private:
    void drawBoxes() const;
    void drawTree();
    void drawNode(KdNode4 * nodes, int idx, const BoundingBox & box);
    void drawSplitPlane(KdTreeNode * node, const BoundingBox & box);
    
private slots:
    void simulate();
    
private:
    KdNTree<TestBox, KdNode4 > * m_tree;
    SahSplit<TestBox> * m_boxes;
};
//! [3]

#endif
