#ifndef QIconFrame_H
#define QIconFrame_H

#include <QColor>
#include <QImage>
#include <QPoint>
#include <QWidget>

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
