#pragma once

namespace sdb {

class Dropoff {
public:
    enum DistanceFunction {
        kLinear = 0,
        kQuadratic = 1,
        kCubic = 2,
        kCosineCurve = 3,
        kExponential = 4
    };
	virtual ~Dropoff() {}
    virtual float f(const float & x, const float & scaling) {
        return 1.f;
    }
    static float linear(const float & x, const float & scaling);
    static float quadratic(const float & x, const float & scaling);
    static float cubic(const float & x, const float & scaling);
    static float cosineCurve(const float & x, const float & scaling);
    static float exponentialCurve(const float & x, const float & scaling);
private:
    
};

template<typename T>
class DropoffFunction : public Dropoff {
public:
    DropoffFunction() {}
    virtual float f(const float & x, const float & scaling) {
        return static_cast<T *>(this)->func(x, scaling);
    }
};

class DropoffLinear : public DropoffFunction<DropoffLinear> {
public:
    float func(const float & x, const float & scaling) {
        return linear(x, scaling);
    }
};

class DropoffQuadratic : public DropoffFunction<DropoffQuadratic> {
public:
    float func(const float & x, const float & scaling) {
        return quadratic(x, scaling);
    }
};

class DropoffCubic : public DropoffFunction<DropoffCubic> {
public:
    float func(const float & x, const float & scaling) {
        return cubic(x, scaling);
    }
};

class DropoffCosineCurve : public DropoffFunction<DropoffCosineCurve> {
public:
    float func(const float & x, const float & scaling) {
        return cosineCurve(x, scaling);
    }
};

class DropoffExponential : public DropoffFunction<DropoffExponential> {
public:
    float func(const float & x, const float & scaling) {
        return exponentialCurve(x, scaling);
    }
};

}
