#pragma once

namespace sdb {

class Dropoff {
public:
    enum DistanceFunction {
        Linear = 0,
        Quadratic = 1,
        Cubic = 2
    };
    virtual float f(const float & x, const float & scaling) {
        return 1.f;
    }
    static float linear(const float & x, const float & scaling);
    static float quadratic(const float & x, const float & scaling);
    static float cubic(const float & x, const float & scaling);
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

}
