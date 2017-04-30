#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include <Sculptor.h>

class GLWidget : public aphid::Base3DView
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
	void drawPoints(aphid::sdb::WorldGrid<aphid::sdb::Array<int, aphid::sdb::VertexP>, aphid::sdb::VertexP > * tree);
	void drawPoints(aphid::sdb::Array<int, aphid::sdb::VertexP> * ps);
	void drawPoints(const aphid::sdb::ActiveGroup & grp);

private:
	aphid::sdb::Sculptor * m_sculptor;
    aphid::sdb::PNPrefW * m_pool;
	
private slots:
    

};
//! [3]

#endif
