#include "LfMachine.h"

namespace lfr {

LfMachine::LfMachine(LfParameter * param) 
{ m_param = param; }

LfMachine::~LfMachine() {}

const LfParameter * LfMachine::param() const
{ return m_param; }

LfParameter * LfMachine::param1()
{ return m_param; }

void LfMachine::initDictionary() {}

void LfMachine::dictionaryAsImage(unsigned * imageBits, int imageW, int imageH) {}

void LfMachine::fillPatch(unsigned * dst, float * color, int s, int imageW, int rank)
{
    const int stride = s * s;
	int crgb[3];
	int i, j, k;
	unsigned * line = dst;
	for(j=0;j<s; j++) {
		for(i=0; i<s; i++) {
			unsigned v = 255<<24;
			for(k=0;k<rank;k++) {				
				crgb[k] = 8 + 500 * color[(j * s + i) + k * stride];
				crgb[k] = std::min<int>(crgb[k], 255);
				crgb[k] = std::max<int>(crgb[k], 0);
			}
			v = v | ( crgb[0] << 16 );
			v = v | ( crgb[1] << 8 );
			v = v | ( crgb[2] );
			line[i] = v;
		}
		line += imageW;
	}
}

void LfMachine::cleanDictionary()
{
    // m_machine->cleanDictionary(m_param);
}

void LfMachine::preLearn() {}

void LfMachine::learn(const ExrImage * image, int iPatch0, int iPatch1) {}

void LfMachine::updateDictionary(const ExrImage * image, int t) {}

void LfMachine::fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor) {}

float LfMachine::computePSNR(const ExrImage * image, int iImage)
{ return 0;
    //const int m = m_param->imageNumPatches(iImage);
    //const int s = m_param->atomSize();
	//return m_machine->computePSNR(image, m, s);
}

void LfMachine::recycleData() {}

}
