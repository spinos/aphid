#ifndef WldWidget_H
#define WldWidget_H

#include <QGLWidget>
#include <Base3DView.h>
#include <KdEngine.h>
#include <ConvexShape.h>
#include <IntersectionContext.h>
#include <HWorldGrid.h>
#include <HAssetGrid.h>
#include <NTreeIO.h>

class WldWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    
    WldWidget(const std::string & filename, QWidget *parent = 0);
    ~WldWidget();
	
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
	void drawVoxel();
	bool readTree(const std::string & filename);
	void testTree();
	void testIntersect(const aphid::Ray * incident);
	void drawActiveSource(const unsigned & iLeaf);
	aphid::BoundingBox getFrameBox();
	
private slots:
	
private:
typedef aphid::sdb::HAssetGrid<aphid::HTriangleAsset, aphid::cvx::Triangle > InnerGridT;
typedef aphid::sdb::HWorldGrid<InnerGridT, aphid::cvx::Triangle > WorldGridT;
	aphid::IntersectionContext m_intersectCtx;
	WorldGridT * m_grid;
	aphid::KdNTree<aphid::Voxel, aphid::KdNode4 > * m_voxelTree;
	aphid::NTreeIO m_hio;
};
//! [3]

#endif
