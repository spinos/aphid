#ifndef APH_Q_ICON_FRAME_H
#define APH_Q_ICON_FRAME_H

#include <QLabel>

QT_BEGIN_NAMESPACE
class QPixmap;
QT_END_NAMESPACE

namespace aphid {

class QIconFrame : public QLabel
{
    Q_OBJECT

public:
    QIconFrame(QWidget *parent = 0);
	
	void addIconFile(const QString & fileName);
	void setIconIndex(int index);
	int getIconIndex() const;
	
	char useNextIcon();
	
signals:

public slots:

protected:
    virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);

private:
    QList<QPixmap *> icons;
	int currentIconIndex;
};

}

#endif
