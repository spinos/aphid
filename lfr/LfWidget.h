#include <QPixmap>
#include <QWidget>

namespace lfr {

class LfWorld;
class LfThread;
class LfWidget : public QWidget
{
    Q_OBJECT

public:
    LfWidget(LfWorld * world, QWidget *parent = 0);

protected:
    void paintEvent(QPaintEvent *event);
    void resizeEvent(QResizeEvent *event);

private slots:
    void updatePixmap(const QImage &image);
	void recvDictionary(const QImage &image);
	void recvSparsity(const QImage &image);
	void simulate(); 

private:
	LfWorld * m_world;
    LfThread * m_thread;
    QPixmap m_pixmap;
	QPixmap m_sparsityPix;
};

}