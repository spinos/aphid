#pragma once
#include <Base3DView.h>
#include <FeatherExample.h>

class BarbView : public Base3DView, public FeatherExample {
Q_OBJECT

public:
    BarbView(QWidget *parent = 0);
    ~BarbView();
	
	virtual void clientDraw();
	virtual void clientSelect();
	virtual void clientMouseInput();
public slots:
	void receiveShapeChanged();
private:

};
