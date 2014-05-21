#include "Dropoff.h"
#include <cmath>
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

float Dropoff::cosineCurve(const float & x, const float & scaling)
{
    return (cos(x/scaling * 3.14159269) + 1.f) * .5f;
}

float Dropoff::exponentialCurve(const float & x, const float & scaling)
{
    return exp(x/scaling * -6.);
}
}
