#include <cmath>
#include <algorithm>
#include "Joints.h"
#include <emmintrin.h>
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstddef>
#include <map>

constexpr float eps = 1e-32;

template<int width> struct alignas(width * sizeof(float)) Vec
{
    float data[width];

    Vec()
    {
    }

    Vec(float val)
    {
        for(int i = 0; i < width; i++)data[i] = val;
    }

    Vec operator - () const
    {
        Vec res;
        for(int i = 0; i < width; i++)res.data[i] = -data[i];
        return res;
    }

    Vec &operator += (const Vec &v)
    {
        for(int i = 0; i < width; i++)data[i] += v[i];  return *this;
    }

    Vec &operator -= (const Vec &v)
    {
        for(int i = 0; i < width; i++)data[i] -= v[i];  return *this;
    }

    Vec &operator *= (const Vec &v)
    {
        for(int i = 0; i < width; i++)data[i] *= v[i];  return *this;
    }

    Vec &operator /= (const Vec &v)
    {
        for(int i = 0; i < width; i++)data[i] /= v[i];  return *this;
    }

    const float &operator [] (int index) const
    {
        return data[index];
    }

    float &operator [] (int index)
    {
        return data[index];
    }
};

template<> struct Vec<4>
{
    __m128 data;

    Vec()
    {
    }

    Vec(float val) : data(_mm_set1_ps(val))
    {
    }

    Vec operator - () const
    {
        Vec res(0);  _mm_sub_ps(res.data, data);  return res;
    }

    Vec &operator += (const Vec &v)
    {
        data = _mm_add_ps(data, v.data);  return *this;
    }

    Vec &operator -= (const Vec &v)
    {
        data = _mm_sub_ps(data, v.data);  return *this;
    }

    Vec &operator *= (const Vec &v)
    {
        data = _mm_mul_ps(data, v.data);  return *this;
    }

    Vec &operator /= (const Vec &v)
    {
        data = _mm_div_ps(data, v.data);  return *this;
    }

    const float &operator [] (int index) const
    {
        return reinterpret_cast<const float *>(&data)[index];
    }

    float &operator [] (int index)
    {
        return reinterpret_cast<float *>(&data)[index];
    }
};

template<int width> Vec<width> operator + (const Vec<width> &a, const Vec<width> &b)
{
    Vec<width> res = a;  res += b;  return res;
}

template<int width> Vec<width> operator - (const Vec<width> &a, const Vec<width> &b)
{
    Vec<width> res = a;  res -= b;  return res;
}

template<int width> Vec<width> operator * (const Vec<width> &a, const Vec<width> &b)
{
    Vec<width> res = a;  res *= b;  return res;
}

template<int width> Vec<width> operator / (const Vec<width> &a, const Vec<width> &b)
{
    Vec<width> res = a;  res /= b;  return res;
}

template<int width> Vec<width> max0(const Vec<width> &v)
{
    Vec<width> res;
    for(int i = 0; i < width; i++)
        res.data[i] = v.data[i] > 0 ? v.data[i] : 0;
    return res;
}

template<> Vec<4> max0(const Vec<4> &v)
{
    Vec<4> res;
    res.data = _mm_max_ps(_mm_set1_ps(0), v.data);
    return res;
}

template<int width> Vec<width> limit(const Vec<width> &v, const Vec<width> &l)
{
    Vec<width> res;
    for(int i = 0; i < width; i++)
    {
        float vabs = v.data[i] < 0 ? -v.data[i] : v.data[i];
        vabs = vabs < l.data[i] ? vabs : l.data[i];
        res.data[i] = v.data[i] < 0 ? -vabs : vabs;
    }
    return res;
}

template<> Vec<4> limit(const Vec<4> &v, const Vec<4> &l)
{
    Vec<4> res;
    __m128 sign = _mm_and_ps(v.data, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));
    __m128 vabs = _mm_and_ps(v.data, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
    res.data = _mm_or_ps(sign, _mm_min_ps(vabs, l.data));
    return res;
}


template<int degreeCount, int vectorWidth> struct JointDataFriction
{
    Vec<vectorWidth> projMainPre[2 * degreeCount], projMainPost[2 * degreeCount];
    Vec<vectorWidth> projFrictPre[2 * degreeCount], projFrictPost[2 * degreeCount];
    Vec<vectorWidth> dstVel;

    Vec<vectorWidth> impulseMain;
    Vec<vectorWidth> impulseFrict;
    Vec<vectorWidth> bodyData[2 * degreeCount];
    float *next[2][vectorWidth];

    float *BodyPtr(int pair, int index)
    {
        return &bodyData[pair * degreeCount][index];
    }

    JointDataFriction()
    {
        std::memset(this, 0, sizeof(*this) - sizeof(next));
        for(int i = 0; i < vectorWidth; i++)next[0][i] = BodyPtr(0, i);
        for(int i = 0; i < vectorWidth; i++)next[1][i] = BodyPtr(1, i);
    }

    void Solve() __attribute__((noinline))
    {
        constexpr float frictCoeff = 0.3f;

        Vec<vectorWidth> deltaMain = impulseMain;  impulseMain += dstVel;
        for(int j = 0; j < 2 * degreeCount; j++)impulseMain -= projMainPre[j] * bodyData[j];
        impulseMain = max0(impulseMain);  deltaMain = impulseMain - deltaMain;
        for(int j = 0; j < 2 * degreeCount; j++)bodyData[j] += projMainPost[j] * deltaMain;

        Vec<vectorWidth> deltaFrict = impulseFrict, maxFrict = Vec<vectorWidth>(frictCoeff) * impulseMain;
        for(int j = 0; j < 2 * degreeCount; j++)impulseFrict -= projFrictPre[j] * bodyData[j];
        impulseFrict = limit(impulseFrict, maxFrict);  deltaFrict = impulseFrict - deltaFrict;
        for(int j = 0; j < 2 * degreeCount; j++)bodyData[j] += projFrictPost[j] * deltaFrict;

        for(int i = 0; i < vectorWidth; i++)
        {
            for(int j = 0; j < degreeCount; j++)
                next[0][i][j * vectorWidth] = bodyData[j + 0 * degreeCount][i];
            for(int j = 0; j < degreeCount; j++)
                next[1][i][j * vectorWidth] = bodyData[j + 1 * degreeCount][i];
        }
    }
};

template<int degreeCount, int vectorWidth> struct JointDataSimple
{
    Vec<vectorWidth> projPre[2 * degreeCount], projPost[2 * degreeCount];
    Vec<vectorWidth> dstVel;

    Vec<vectorWidth> impulse;
    Vec<vectorWidth> bodyData[2 * degreeCount];
    float *next[2][vectorWidth];

    float *BodyPtr(int pair, int index)
    {
        return &bodyData[pair * degreeCount][index];
    }

    JointDataSimple()
    {
        std::memset(this, 0, sizeof(*this) - sizeof(next));
        for(int i = 0; i < vectorWidth; i++)next[0][i] = BodyPtr(0, i);
        for(int i = 0; i < vectorWidth; i++)next[1][i] = BodyPtr(1, i);
    }

    void Solve() __attribute__((noinline))
    {
        Vec<vectorWidth> delta = impulse;  impulse += dstVel;
        for(int j = 0; j < 2 * degreeCount; j++)impulse -= projPre[j] * bodyData[j];
        impulse = max0(impulse);  delta = impulse - delta;
        for(int j = 0; j < 2 * degreeCount; j++)bodyData[j] += projPost[j] * delta;

        for(int i = 0; i < vectorWidth; i++)
        {
            for(int j = 0; j < degreeCount; j++)
                next[0][i][j * vectorWidth] = bodyData[j + 0 * degreeCount][i];
            for(int j = 0; j < degreeCount; j++)
                next[1][i][j * vectorWidth] = bodyData[j + 1 * degreeCount][i];
        }
    }
};

template<int vectorWidth> void FillJointData(const ContactJoint &joint,
    JointDataFriction<3, vectorWidth> &data, int index, float *next0, float *next1)
{
    typedef JointDataFriction<3, vectorWidth> Data;  assert(unsigned(index) < vectorWidth);
    ptrdiff_t offs0 = (reinterpret_cast<char *>(next0) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t offs1 = (reinterpret_cast<char *>(next1) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t beg = offsetof(Data, bodyData), end = beg + sizeof(data.bodyData) - 2 * vectorWidth * sizeof(float);
    assert(offs0 >= beg && offs0 < end);
    assert(offs1 >= beg && offs1 < end);

    data.projMainPre[0][index] = joint.normalLimiter.normalProjector1.x * joint.normalLimiter.compInvMass;
    data.projMainPre[1][index] = joint.normalLimiter.normalProjector1.y * joint.normalLimiter.compInvMass;
    data.projMainPre[2][index] = joint.normalLimiter.angularProjector1 * joint.normalLimiter.compInvMass;
    data.projMainPre[3][index] = joint.normalLimiter.normalProjector2.x * joint.normalLimiter.compInvMass;
    data.projMainPre[4][index] = joint.normalLimiter.normalProjector2.y * joint.normalLimiter.compInvMass;
    data.projMainPre[5][index] = joint.normalLimiter.angularProjector2 * joint.normalLimiter.compInvMass;
    data.projMainPost[0][index] = joint.normalLimiter.compMass1_linear.x;
    data.projMainPost[1][index] = joint.normalLimiter.compMass1_linear.y;
    data.projMainPost[2][index] = joint.normalLimiter.compMass1_angular;
    data.projMainPost[3][index] = joint.normalLimiter.compMass2_linear.x;
    data.projMainPost[4][index] = joint.normalLimiter.compMass2_linear.y;
    data.projMainPost[5][index] = joint.normalLimiter.compMass2_angular;

    data.projFrictPre[0][index] = joint.frictionLimiter.normalProjector1.x * joint.frictionLimiter.compInvMass;
    data.projFrictPre[1][index] = joint.frictionLimiter.normalProjector1.y * joint.frictionLimiter.compInvMass;
    data.projFrictPre[2][index] = joint.frictionLimiter.angularProjector1 * joint.frictionLimiter.compInvMass;
    data.projFrictPre[3][index] = joint.frictionLimiter.normalProjector2.x * joint.frictionLimiter.compInvMass;
    data.projFrictPre[4][index] = joint.frictionLimiter.normalProjector2.y * joint.frictionLimiter.compInvMass;
    data.projFrictPre[5][index] = joint.frictionLimiter.angularProjector2 * joint.frictionLimiter.compInvMass;
    data.projFrictPost[0][index] = joint.frictionLimiter.compMass1_linear.x;
    data.projFrictPost[1][index] = joint.frictionLimiter.compMass1_linear.y;
    data.projFrictPost[2][index] = joint.frictionLimiter.compMass1_angular;
    data.projFrictPost[3][index] = joint.frictionLimiter.compMass2_linear.x;
    data.projFrictPost[4][index] = joint.frictionLimiter.compMass2_linear.y;
    data.projFrictPost[5][index] = joint.frictionLimiter.compMass2_angular;

    data.dstVel[index] = joint.normalLimiter.dstVelocity * joint.normalLimiter.compInvMass;

    data.impulseMain[index] = joint.normalLimiter.accumulatedImpulse;
    data.impulseFrict[index] = joint.frictionLimiter.accumulatedImpulse;

    data.bodyData[0][index] = joint.body1->velocity.x;
    data.bodyData[1][index] = joint.body1->velocity.y;
    data.bodyData[2][index] = joint.body1->angularVelocity;
    data.bodyData[3][index] = joint.body2->velocity.x;
    data.bodyData[4][index] = joint.body2->velocity.y;
    data.bodyData[5][index] = joint.body2->angularVelocity;

    data.next[0][index] = next0;
    data.next[1][index] = next1;
}

template<int vectorWidth> void ApplyJointData(ContactJoint &joint, const JointDataFriction<3, vectorWidth> &data, int index)
{
    joint.normalLimiter.accumulatedImpulse = data.impulseMain[index];
    joint.frictionLimiter.accumulatedImpulse = data.impulseFrict[index];
}

template<int vectorWidth> void ApplyBodyData(RigidBody *body, const float *data, const JointDataFriction<3, vectorWidth> *unused)
{
    body->velocity.x = data[0 * vectorWidth];
    body->velocity.y = data[1 * vectorWidth];
    body->angularVelocity = data[2 * vectorWidth];
}

template<int vectorWidth> void FillJointData(const ContactJoint &joint,
    JointDataSimple<3, vectorWidth> &data, int index, float *next0, float *next1)
{
    typedef JointDataSimple<3, vectorWidth> Data;  assert(unsigned(index) < vectorWidth);
    ptrdiff_t offs0 = (reinterpret_cast<char *>(next0) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t offs1 = (reinterpret_cast<char *>(next1) - reinterpret_cast<char *>(&data) + 65536 * sizeof(data)) % sizeof(data);
    ptrdiff_t beg = offsetof(Data, bodyData), end = beg + sizeof(data.bodyData) - 2 * vectorWidth * sizeof(float);
    assert(offs0 >= beg && offs0 < end);
    assert(offs1 >= beg && offs1 < end);

    data.projPre[0][index] = joint.normalLimiter.normalProjector1.x * joint.normalLimiter.compInvMass;
    data.projPre[1][index] = joint.normalLimiter.normalProjector1.y * joint.normalLimiter.compInvMass;
    data.projPre[2][index] = joint.normalLimiter.angularProjector1 * joint.normalLimiter.compInvMass;
    data.projPre[3][index] = joint.normalLimiter.normalProjector2.x * joint.normalLimiter.compInvMass;
    data.projPre[4][index] = joint.normalLimiter.normalProjector2.y * joint.normalLimiter.compInvMass;
    data.projPre[5][index] = joint.normalLimiter.angularProjector2 * joint.normalLimiter.compInvMass;
    data.projPost[0][index] = joint.normalLimiter.compMass1_linear.x;
    data.projPost[1][index] = joint.normalLimiter.compMass1_linear.y;
    data.projPost[2][index] = joint.normalLimiter.compMass1_angular;
    data.projPost[3][index] = joint.normalLimiter.compMass2_linear.x;
    data.projPost[4][index] = joint.normalLimiter.compMass2_linear.y;
    data.projPost[5][index] = joint.normalLimiter.compMass2_angular;

    data.dstVel[index] = joint.normalLimiter.dstDisplacingVelocity * joint.normalLimiter.compInvMass;
    data.impulse[index] = joint.normalLimiter.accumulatedDisplacingImpulse;

    data.bodyData[0][index] = joint.body1->displacingVelocity.x;
    data.bodyData[1][index] = joint.body1->displacingVelocity.y;
    data.bodyData[2][index] = joint.body1->displacingAngularVelocity;
    data.bodyData[3][index] = joint.body2->displacingVelocity.x;
    data.bodyData[4][index] = joint.body2->displacingVelocity.y;
    data.bodyData[5][index] = joint.body2->displacingAngularVelocity;

    data.next[0][index] = next0;
    data.next[1][index] = next1;
}

template<int vectorWidth> void ApplyJointData(ContactJoint &joint, const JointDataSimple<3, vectorWidth> &data, int index)
{
    joint.normalLimiter.accumulatedDisplacingImpulse = data.impulse[index];
}

template<int vectorWidth> void ApplyBodyData(RigidBody *body, const float *data, const JointDataSimple<3, vectorWidth> *unused)
{
    body->displacingVelocity.x = data[0 * vectorWidth];
    body->displacingVelocity.y = data[1 * vectorWidth];
    body->displacingAngularVelocity = data[2 * vectorWidth];
}


float CheckSolveImpulse(ContactJoint &joint)
{
    constexpr int index = 2;
    JointDataFriction<3, 4> data;
    FillJointData(joint, data, index, &data.bodyData[0][index], &data.bodyData[3][index]);
    joint.SolveImpulse();
    data.Solve();

    float err = 0;
    err += std::abs(joint.body1->velocity.x - data.bodyData[0][index]);
    err += std::abs(joint.body1->velocity.y - data.bodyData[1][index]);
    err += std::abs(joint.body1->angularVelocity - data.bodyData[2][index]);
    err += std::abs(joint.body2->velocity.x - data.bodyData[3][index]);
    err += std::abs(joint.body2->velocity.y - data.bodyData[4][index]);
    err += std::abs(joint.body2->angularVelocity - data.bodyData[5][index]);

    err += std::abs(joint.normalLimiter.accumulatedImpulse - data.impulseMain[index]);
    err += std::abs(joint.frictionLimiter.accumulatedImpulse - data.impulseFrict[index]);

    if(!(err < 1e-3))std::cout << "Error " << err << std::endl;
    return err;
}

float CheckSolveDisplacingImpulse(ContactJoint &joint)
{
    constexpr int index = 1;
    JointDataSimple<3, 4> data;
    FillJointData(joint, data, index, &data.bodyData[0][index], &data.bodyData[3][index]);
    joint.SolveDisplacement();
    data.Solve();

    float err = 0;
    err += std::abs(joint.body1->displacingVelocity.x - data.bodyData[0][index]);
    err += std::abs(joint.body1->displacingVelocity.y - data.bodyData[1][index]);
    err += std::abs(joint.body1->displacingAngularVelocity - data.bodyData[2][index]);
    err += std::abs(joint.body2->displacingVelocity.x - data.bodyData[3][index]);
    err += std::abs(joint.body2->displacingVelocity.y - data.bodyData[4][index]);
    err += std::abs(joint.body2->displacingAngularVelocity - data.bodyData[5][index]);

    err += std::abs(joint.normalLimiter.accumulatedDisplacingImpulse - data.impulse[index]);

    if(!(err < 1e-3))std::cout << "Error " << err << std::endl;
    return err;
}


struct JointData
{
    int index, next[2];

    JointData() : index(-1), next{-1, -1}
    {
    }
};

struct BodyData
{
    int first, last, *prev;

    BodyData() : first(-1), last(-1), prev(&first)
    {
    }
};

template<int log2width> struct JointSorter
{
    static constexpr int width = 1 << log2width, mask = width - 1;
    static constexpr int mask2 = (2 << log2width) - 1;

    ContactJoint *joints;
    std::vector<JointData> jointData;
    std::map<RigidBody *, BodyData> bodyData;
    std::vector<int> queue;
    int nWaiting, current;

    bool Append(const ContactJoint &joint, int index)
    {
        BodyData &body1 = bodyData[joint.body1];
        BodyData &body2 = bodyData[joint.body2];
        if(!((body1.last ^ current) & ~mask2) || !((body2.last ^ current) & ~mask2))
        {
            if(nWaiting >= queue.size())queue.resize(nWaiting + 1);
            queue[nWaiting++] = index;  return false;
        }
        *body1.prev = body1.last = current | 0;
        body1.prev = &jointData[index].next[0];
        *body2.prev = body2.last = current | 1;
        body2.prev = &jointData[index].next[1];
        jointData[index].index = current >> 1;
        return !((current += 2) & mask2);
    }

    void ProcessQueue()
    {
        int n = nWaiting, i = 0;  nWaiting = 0;
        while(i < n)
            if(Append(joints[queue[i]], queue[i]))
            {
                while(i < n)queue[nWaiting++] = queue[i++];
                n = nWaiting;  i = nWaiting = 0;
            }
            else i++;
    }

    JointSorter(ContactJoint *joints_, int nJoints) : joints(joints_), jointData(nJoints), nWaiting(0), current(0)
    {
        for(int i = 0; i < nJoints; i++)
        {
            if(Append(joints[i], i))ProcessQueue();
        }
        for(;;)
        {
            for(; current & mask2; current += 2);  // TODO: mark for clearing
            if(!nWaiting)break;  ProcessQueue();
        }
        for(auto &body : bodyData)*body.second.prev = body.second.first;
        assert(!(current & mask2));  current >>= 1;
    }

    int Size() const
    {
        return current >> log2width;
    }

    template<typename T> static float *DataPointer(T *data, int ref)
    {
        return data[ref >> (log2width + 1)].BodyPtr(ref & 1, (ref >> 1) & mask);
    }

    template<typename T> void Fill(T *data) const
    {
        for(int i = 0; i < jointData.size(); i++)
            FillJointData<width>(joints[i], data[jointData[i].index >> log2width], jointData[i].index & mask,
                DataPointer(data, jointData[i].next[0]), DataPointer(data, jointData[i].next[1]));
    }

    template<typename T> void Apply(T *data) const
    {
        for(int i = 0; i < jointData.size(); i++)
            ApplyJointData<width>(joints[i], data[jointData[i].index >> log2width], jointData[i].index & mask);
        for(auto &body : bodyData)
            ApplyBodyData<width>(body.first, DataPointer(data, body.second.first), data);
    }

    template<template<int, int> class T> void Solve(int nIter) const
    {
        std::vector<T<3, width>> data(Size());
        Fill(data.data());
        for(int i = 0; i < nIter; i++)
            for(auto &block : data)block.Solve();
        Apply(data.data());
    }
};

void FastSolveJoints(ContactJoint *joints, int nJoints, int nIter, int nDisp)
{
    JointSorter<2> sorter(joints, nJoints);
    sorter.Solve<JointDataFriction>(nIter);
    sorter.Solve<JointDataSimple>(nDisp);
}
