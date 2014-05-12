#include "Dropoff.h"

namespace sdb {
float Dropoff::linear(const float & x, const float & scaling)
{
    float y = 1.f - x / scaling;
    if(y < 0.f) y = 0.f;
    return y;
}

float Dropoff::quadratic(const float & x, const float & scaling)
{
    float y = 1.f - x / scaling;
    if(y < 0.f) y = 0.f;
    y *= y;
    return y;
}

float Dropoff::cubic(const float & x, const float & scaling)
{
    float y = 1.f - x / scaling;
    if(y < 0.f) y = 0.f;
    y *= y * y;
    return y;
}
}
