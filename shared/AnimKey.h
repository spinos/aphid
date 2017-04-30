#ifndef ANIMKEY_H
#define ANIMKEY_H

namespace aphid {
struct AnimKey
{
	double key;
	double value;
	double inAngle;
	double inWeight;
	double outAngle;
	double outWeight;
	short inTangent;
	short outTangent;
};

struct AnimCurveData
{
	std::string curveType;
	std::string name;
	std::string unitless;
	std::string weighted;
	std::string unitName;
	std::string preInfinity;
	std::string postInfinity;
};
}
#endif        //  #ifndef ANIMKEY_H

