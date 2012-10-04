__device__ 
inline float dot(float3 v0, float3 v1)
{
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

__device__ 
inline float3 scale(float3 v0, float scalar)
{
    float3 res = v0;
    res.x *= scalar;
    res.y *= scalar;
    res.z *= scalar;
    return res;
}

__device__ 
inline float3 divide(float3 v0, float scalar)
{
    float3 res = v0;
    res.x /= scalar;
    res.y /= scalar;
    res.z /= scalar;
    return res;
}

__device__
inline float3 add(float3 v0, float3 v1)
{
    float3 res = v0;
    res.x += v1.x;
    res.y += v1.y;
    res.z += v1.z;
    return res;
}

__device__ 
inline float3 minus(float3 v0, float3 v1)
{
    float3 res = v0;
    res.x -= v1.x;
    res.y -= v1.y;
    res.z -= v1.z;
    return res;
}

__device__
inline float3 normalize(float3 v0)
{
    float3 res = v0;
    float len = sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z);
    if(len > 10e-6)
    {
        res.x /= len;
        res.y /= len;
        res.z /= len;
    }
    else
    {
        res.x = res.y = res.z = 0.57735f;
    }
    
    return res;
}

__device__
inline float sqr(float x)
{
    return x * x;
}

__device__ 
inline float3 reflect(float3 I, float3 N)
{
    return  minus(scale(N, 2.f * dot(I,N)), I);
}

__device__
inline float3 calculateL(float3 * p, unsigned int width, unsigned int x, unsigned int y)
{
    return normalize(p[y * width + x]);   
}

__device__ 
inline float beckmann(float m, float t)
{
    float M = m*m;
    float T = t*t;
    return exp((T-1)/(M*T)) / (M*T*T);
}

__device__ 
inline float fresnel(float f0, float u)
{
    return f0 + (1.f - f0) * pow(1.f - u, 5);
}
