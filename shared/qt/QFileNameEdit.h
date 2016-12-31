#pragma once
#include "QModelEdit.h"

namespace aphid {

class QFileNameEdit : public QModelEdit
{
    Q_OBJECT
public:
    QFileNameEdit(const QModelIndex & idx, QWidget * parent = 0);
    void setValue(const std::string & x);
    std::string value();
    std::string pickFile();
public slots:
    
signals:
    
protected:
    
private:
    std::string m_fileName;
};

}

