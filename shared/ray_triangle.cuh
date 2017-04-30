#ifndef RAY_TRIANGLE_CUH
#define RAY_TRIANGLE_CUH
/*
 * reference
 * http://geomalgorithms.com/a06-_intersect-2.html
 * http://www.hugi.scene.org/online/hugi25/hugi%2025%20-%20coding%20corner%20graphics,%20sound%20&%20synchronization%20ken%20ray-triangle%20intersection%20tests%20for%20dummies.htm
 */
#include "ray_intersection.cuh"

__device__ int ray_triangle_MollerTrumbore(const Ray & iray,
                                float3 v1,
                                float3 v2,
                                float3 v3,
                                float & u,
                                float & v,
                                float & t,
                                int & frontfacing)
{ 
    float3 e2 = float3_difference(v3,v1);       // second edge
    float3 e1 = float3_difference(v2,v1);       // first edge
    float3 r = float3_cross<float4, float3>(iray.d, e2);  // (d X e2) is used two times in the formula
                                        // so we store it in an appropriate vector
    float3 s = float3_difference<float4, float3>(iray.o, v1);       // translated ray origin
    float a = float3_dot(e1, r);    // a=(d X e2)*e1
    float f = 1.f / a;           // slow division*
    float3 q= float3_cross(s, e1);
    u= float3_dot(s,r);
    frontfacing=1;
    if(a > 1e-6f)            // eps is the machine fpu epsilon (precision), 
                        // or a very small number :)
    { // Front facing triangle...
        if (u<0||u>a) return 0;
        v = float3_dot<float4, float3>(iray.d, q);
        if (v<0||u+v>a) return 0;
    }
    else if (a<-1e-6f)
    { // Back facing triangle...
        frontfacing=0;
        if (u>0||u<a) return 0;
        v= float3_dot<float4, float3>(iray.d, q);
        if (v>0||u+v<a) return 0;
    } 
    else return 0; // Ray parallel to triangle plane
    t= f * float3_dot(e2,q);
    u=u*f; v=v*f;
    return 1;
}

__device__ void triangle_normal(float3 & N,
                                float3 v1,
                                float3 v2,
                                float3 v3)
{
    float3 e2 = float3_difference(v3,v1);       // second edge
    float3 e1 = float3_difference(v2,v1);       // first edge
    float3_cross1(N, e1, e2);
}
#endif        //  #ifndef RAY_TRIANGLE_CUH

