/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/
#include "intersector_lds.h"

#include "calc.h"
#include "executable.h"
#include "../primitive/mesh.h"
#include "../primitive/instance.h"
#include "../translator/q_bvh_translator.h"
#include "../world/world.h"

#include <iostream>
#include <fstream>
#include <math.h>       /* fma, FP_FAST_FMA */'
#include <queue>


#include <chrono>
using namespace std::chrono;

#define SINGLE_TRIANGLE 0
#define SINGLE_BOX 0
#define PRINT_TREE 0
#define SERIALIZE_RAYS 1
#define PROFILE_TRAVERSAL 1


struct Stats
{
    uint32_t rayTris;
    uint32_t rayAABBs;
    float epoSum;
    float epoSumTotal;
    float surfaceAreaSumInt;
    float surfaceAreaSumLeaf;
    float pad0;
    float pad1;
};

namespace Math
{
    struct Range
    {
        float min;
        float max;
    };

    struct Vec3
    {
        float x;
        float y;
        float z;

        Vec3 operator-(const Vec3 &rhs) const
        {
            return { x - rhs.x, y - rhs.y, z - rhs.z };
        }

        Vec3 operator*(const float &rhs) const
        {
            return { x * rhs, y * rhs, z * rhs };
        }

        Vec3 operator*(const Vec3 &rhs) const
        {
            return { x * rhs.x, y * rhs.y, z * rhs.z };
        }

        Vec3 operator+(const Vec3 &rhs) const
        {
            return { x + rhs.x, y + rhs.y, z + rhs.z };
        }

        float dot(const Vec3 &rhs) const
        {
            return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
        }

        Vec3 Cross(const Vec3 &rhs) const
        {
            return {
                (y * rhs.z) - (z * rhs.y),
                (z * rhs.x) - (x * rhs.z),
                (x * rhs.y) - (y * rhs.x) };
        }

        float Magnitude() const
        {
            return sqrt((x * x) + (y * y) + (z * z));
        }

        Vec3 Normalize() const
        {
            float mag = Magnitude();
            return { x / mag, y / mag, z / mag };
        }

        Vec3 Inverse() const
        {
            const float eps = 1e-6;

            return { 1.0f / (fabs(x) > eps ? x : copysign(eps, x)),
                     1.0f / (fabs(y) > eps ? y : copysign(eps, y)),
                     1.0f / (fabs(z) > eps ? z : copysign(eps, z)) };
        }

        Vec3 Min(const Vec3 rhs) const
        {
            return { std::min(x, rhs.x), std::min(y, rhs.y), std::min(z, rhs.z) };
        }

        Vec3 Max(const Vec3 rhs) const
        {
            return { std::max(x, rhs.x), std::max(y, rhs.y), std::max(z, rhs.z) };
        }
    };

    struct Plane
    {
        Vec3 point;
        Vec3 normal;
    };

    struct AABB
    {
        Vec3 min;
        Vec3 max;
    };

    struct Triangle
    {
        Vec3 v0;
        Vec3 v1;
        Vec3 v2;
    };

    struct Ray
    {
        Vec3 origin;
        Vec3 dir;
        float t_min;
        float t_max;

        Ray(Vec3 p0, Vec3 p1) :
            origin(p0),
            dir(p1 - p0),
            t_min(0),
            t_max(dir.Magnitude())
        {
            dir = dir.Normalize();
        }

        Vec3 At(float t)
        {
            return origin + (dir * t);
        }

        float Intersect(const Plane& plane) const
        {
            const float eps = 1e-6;

            float denom = plane.normal.dot(dir);

            if (abs(denom) > eps) {
                return (plane.point - origin).dot(plane.normal) / denom;
            }

            return -FLT_MAX;
        }

        Range Intersect(const AABB& aabb) const
        {
            Vec3 invdir = dir.Inverse();
            Vec3 oxinvdir = origin * invdir;

            const Vec3 f = (aabb.max * invdir) + oxinvdir;
            const Vec3 n = (aabb.min * invdir) + oxinvdir;
            const Vec3 tmax = f.Max(n);
            const Vec3 tmin = f.Min(n);

            return { std::max(std::max(std::max(tmin.x, tmin.y), tmin.z), t_min),
                     std::min(std::min(std::min(tmax.x, tmax.y), tmax.z), t_max) };
        }
    };
}

static bool AabbAabbIntersect(
    const Math::AABB& aabb0,
    const Math::AABB& aabb1)
{
    return (aabb0.min.x <= aabb1.max.x && aabb0.max.x >= aabb1.min.x) &&
           (aabb0.min.y <= aabb1.max.y && aabb0.max.y >= aabb1.min.y) &&
           (aabb0.min.z <= aabb1.max.z && aabb0.max.z >= aabb1.min.z);
}

static float AabbTriIntersectArea(
    const Math::AABB& aabb,
    const Math::Triangle& tri_orig)
{
    std::queue<Math::Triangle> tris;
    tris.push(tri_orig);
    size_t triCount = 0;

    float eps = 1e-6;

    std::vector<Math::Plane> aabbPlanes = {
        { {aabb.min.x - eps, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f} },
        { {aabb.max.x + eps, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f} },
        { {0.0f, aabb.min.y - eps, 0.0f}, {0.0f, -1.0f, 0.0f} },
        { {0.0f, aabb.max.y + eps, 0.0f}, {0.0f, 1.0f, 0.0f} },
        { {0.0f, 0.0f, aabb.min.z - eps}, {0.0f, 0.0f, -1.0f} },
        { {0.0f, 0.0f, aabb.max.z + eps}, {0.0f, 0.0f, 1.0f} } };

    for (auto& plane : aabbPlanes)
    {
        triCount = tris.size();

        for (uint32_t i = 0; i < triCount; ++i)
        {
            Math::Triangle& tri = tris.front();

            std::vector<Math::Vec3> posHalf;
            std::vector<Math::Vec3> negHalf;

            if ((tri.v0 - plane.point).dot(plane.normal) < 0.0f)
            {
                posHalf.push_back(tri.v0);
            }
            else
            {
                negHalf.push_back(tri.v0);
            }

            if ((tri.v1 - plane.point).dot(plane.normal) < 0.0f)
            {
                posHalf.push_back(tri.v1);
            }
            else
            {
                negHalf.push_back(tri.v1);
            }

            if ((tri.v2 - plane.point).dot(plane.normal) < 0.0f)
            {
                posHalf.push_back(tri.v2);
            }
            else
            {
                negHalf.push_back(tri.v2);
            }

            switch (posHalf.size())
            {
            case 0:
                break;
            case 1:
            {
                Math::Ray ray0 = Math::Ray(negHalf[0], posHalf[0]);
                Math::Ray ray1 = Math::Ray(negHalf[1], posHalf[0]);

                float t0 = ray0.Intersect(plane);
                float t1 = ray1.Intersect(plane);

                if (t0 != -FLT_MAX && t1 != -FLT_MAX)
                {
                    Math::Vec3 p0 = ray0.At(t0);
                    Math::Vec3 p1 = ray1.At(t1);

                    tris.push({ posHalf[0], p0, p1 });
                }
                break;
            }
            case 2:
            {
                Math::Ray ray0 = Math::Ray(negHalf[0], posHalf[0]);
                Math::Ray ray1 = Math::Ray(negHalf[0], posHalf[1]);

                float t0 = ray0.Intersect(plane);
                float t1 = ray1.Intersect(plane);

                if (t0 != -FLT_MAX && t1 != -FLT_MAX)
                {
                    Math::Vec3 p0 = ray0.At(t0);
                    Math::Vec3 p1 = ray1.At(t1);

                    tris.push({ posHalf[0], p0, p1 });
                    tris.push({ posHalf[0], p1, posHalf[1] });
                }
                break;
            }
            case 3:
                tris.push(tri);
                break;
            default:
                std::cout << "Should never get here!!! Invalid value posHalf.size() = " << posHalf.size() << std::endl;
                break;
            };

            tris.pop();
        }
    }

    float area = 0;

    while (!tris.empty())
    {
        auto& tri = tris.front();

        area += (tri.v0 - tri.v1).Cross((tri.v2 - tri.v1)).Magnitude() / 2.0f;

        tris.pop();
    }

    return area;
}

namespace RadeonRays
{
    // Preferred work group size for Radeon devices
    static int const kMaxStackSize  = 48;
    static int const kWorkGroupSize = 64;

    struct IntersectorLDS::GpuData
    {
        struct Program
        {
            Program(Calc::Device *device)
                : device(device)
                , executable(nullptr)
                , isect_func(nullptr)
                , occlude_func(nullptr)
            {
            }

            ~Program()
            {
                if (executable)
                {
                    executable->DeleteFunction(isect_func);
                    executable->DeleteFunction(occlude_func);
                    device->DeleteExecutable(executable);
                }
            }

            Calc::Device *device;

            Calc::Executable *executable;
            Calc::Function *isect_func;
            Calc::Function *occlude_func;
        };

        // Device
        Calc::Device *device;
        // BVH nodes
        Calc::Buffer *bvh;
        // Traversal stack
        Calc::Buffer *stack;

        Program *prog;
        Program bvh_prog;
        Program qbvh_prog;

#if SERIALIZE_RAYS
        Program serial_prog;
#if PROFILE_TRAVERSAL
        Program profiling_prog;
#endif
#endif

        GpuData(Calc::Device *device)
            : device(device)
            , bvh(nullptr)
            , stack(nullptr)
            , prog(nullptr)
            , bvh_prog(device)
            , qbvh_prog(device)
#if SERIALIZE_RAYS
            , serial_prog(device)
#if PROFILE_TRAVERSAL
            , profiling_prog(device)
#endif
#endif
        {
        }

        ~GpuData()
        {
            device->DeleteBuffer(bvh);
            device->DeleteBuffer(stack);
        }
    };

    IntersectorLDS::IntersectorLDS(Calc::Device *device)
        : Intersector(device)
        , m_gpudata(new GpuData(device))
        , m_epoBuffer(nullptr)
        , m_surfAreaBuffer(nullptr)
        , m_totalArea(0.0f)
        , m_rootSA(0.0f)
    {
        std::string buildopts;
#ifdef RR_RAY_MASK
        buildopts.append("-D RR_RAY_MASK ");
#endif

#ifdef RR_BACKFACE_CULL
        buildopts.append("-D RR_BACKFACE_CULL ");
#endif // RR_BACKFACE_CULL

#ifdef USE_SAFE_MATH
        buildopts.append("-D USE_SAFE_MATH ");
#endif

        Calc::DeviceSpec spec;
        m_device->GetSpec(spec);

#ifndef RR_EMBED_KERNELS
        if (device->GetPlatform() == Calc::Platform::kOpenCL)
        {
            const char *headers[] = { "../RadeonRays/src/kernels/CL/common.cl" };

            int numheaders = sizeof(headers) / sizeof(const char *);

            m_gpudata->bvh_prog.executable = m_device->CompileExecutable("../RadeonRays/src/kernels/CL/intersect_bvh2_lds.cl", headers, numheaders, buildopts.c_str());
            if (spec.has_fp16)
                m_gpudata->qbvh_prog.executable = m_device->CompileExecutable("../RadeonRays/src/kernels/CL/intersect_bvh2_lds_fp16.cl", headers, numheaders, buildopts.c_str());
        }
        else
        {
            assert(device->GetPlatform() == Calc::Platform::kVulkan);
            m_gpudata->bvh_prog.executable = m_device->CompileExecutable("../RadeonRays/src/kernels/GLSL/bvh2.comp", nullptr, 0, buildopts.c_str());
            if (spec.has_fp16)
                m_gpudata->qbvh_prog.executable = m_device->CompileExecutable("../RadeonRays/src/kernels/GLSL/bvh2_fp16.comp", nullptr, 0, buildopts.c_str());
        }
#else
#if USE_OPENCL
        if (device->GetPlatform() == Calc::Platform::kOpenCL)
        {
#if SERIALIZE_RAYS
#if BVH3
            m_gpudata->serial_prog.executable = m_device->CompileExecutable(g_intersect_bvh3_lds_serial_opencl, std::strlen(g_intersect_bvh3_lds_serial_opencl), buildopts.c_str());
#else
            m_gpudata->serial_prog.executable = m_device->CompileExecutable(g_intersect_bvh2_lds_serial_opencl, std::strlen(g_intersect_bvh2_lds_serial_opencl), buildopts.c_str());
#endif
#if PROFILE_TRAVERSAL
#if BVH3
            m_gpudata->profiling_prog.executable = m_device->CompileExecutable(g_intersect_bvh3_lds_profiling_opencl, std::strlen(g_intersect_bvh3_lds_profiling_opencl), buildopts.c_str());
#else
            m_gpudata->profiling_prog.executable = m_device->CompileExecutable(g_intersect_bvh2_lds_profiling_opencl, std::strlen(g_intersect_bvh2_lds_profiling_opencl), buildopts.c_str());
#endif
#endif
#endif
            m_gpudata->bvh_prog.executable = m_device->CompileExecutable(
#if SINGLE_TRIANGLE
                g_intersect_bvh2_lds_opencl_single_triangle, std::strlen(g_intersect_bvh2_lds_opencl_single_triangle),
#elif SINGLE_BOX
                g_intersect_bvh2_lds_opencl_single_box, std::strlen(g_intersect_bvh2_lds_opencl_single_box),
#else
#if BVH3
                g_intersect_bvh3_lds_opencl, std::strlen(g_intersect_bvh3_lds_opencl),
#else
                g_intersect_bvh2_lds_opencl, std::strlen(g_intersect_bvh2_lds_opencl),
#endif
#endif
                buildopts.c_str());
            if (spec.has_fp16)
                m_gpudata->qbvh_prog.executable = m_device->CompileExecutable(g_intersect_bvh2_lds_fp16_opencl, std::strlen(g_intersect_bvh2_lds_fp16_opencl), buildopts.c_str());
        }
#endif
#if USE_VULKAN
        if (device->GetPlatform() == Calc::Platform::kVulkan)
        {
            if (m_gpudata->bvh_prog.executable == nullptr)
                m_gpudata->bvh_prog.executable = m_device->CompileExecutable(g_bvh2_vulkan, std::strlen(g_bvh2_vulkan), buildopts.c_str());
            if (m_gpudata->qbvh_prog.executable == nullptr && spec.has_fp16)
                m_gpudata->qbvh_prog.executable = m_device->CompileExecutable(g_bvh2_fp16_vulkan, std::strlen(g_bvh2_fp16_vulkan), buildopts.c_str());
        }
#endif
#endif

#if SERIALIZE_RAYS
        m_gpudata->serial_prog.isect_func = m_gpudata->serial_prog.executable->CreateFunction("intersect_main");
        m_gpudata->serial_prog.occlude_func = m_gpudata->serial_prog.executable->CreateFunction("occluded_main");
#if PROFILE_TRAVERSAL
        m_gpudata->profiling_prog.isect_func = m_gpudata->profiling_prog.executable->CreateFunction("intersect_main");
        m_gpudata->profiling_prog.occlude_func = m_gpudata->profiling_prog.executable->CreateFunction("occluded_main");
#endif
#endif
        m_gpudata->bvh_prog.isect_func = m_gpudata->bvh_prog.executable->CreateFunction("intersect_main");
        m_gpudata->bvh_prog.occlude_func = m_gpudata->bvh_prog.executable->CreateFunction("occluded_main");

        if (m_gpudata->qbvh_prog.executable)
        {
            m_gpudata->qbvh_prog.isect_func = m_gpudata->qbvh_prog.executable->CreateFunction("intersect_main");
            m_gpudata->qbvh_prog.occlude_func = m_gpudata->qbvh_prog.executable->CreateFunction("occluded_main");
        }
    }

    void IntersectorLDS::GetEPO(float* epo, uint32_t comp, uint32_t current, BvhX* tree, bool totalSum) const
    {
        BvhX::Node& currentNode = tree->m_nodes[current];

        Math::AABB currentBounds = { {tree->m_epoDatas[current].aabb_min[0], tree->m_epoDatas[current].aabb_min[1], tree->m_epoDatas[current].aabb_min[2]},
                                     {tree->m_epoDatas[current].aabb_max[0], tree->m_epoDatas[current].aabb_max[1], tree->m_epoDatas[current].aabb_max[2]} };
        Math::AABB compBounds = { {tree->m_epoDatas[comp].aabb_min[0], tree->m_epoDatas[comp].aabb_min[1], tree->m_epoDatas[comp].aabb_min[2]},
                                  {tree->m_epoDatas[comp].aabb_max[0], tree->m_epoDatas[comp].aabb_max[1], tree->m_epoDatas[comp].aabb_max[2]} };

        if ((totalSum || current != comp) && AabbAabbIntersect(currentBounds, compBounds))
        {
            if (!BvhX::IsInternal(currentNode))
            {
                Math::Triangle tri = { {currentNode.aabb_left_min_or_v0[0], currentNode.aabb_left_min_or_v0[1], currentNode.aabb_left_min_or_v0[2]},
                                       {currentNode.aabb_left_max_or_v1[0], currentNode.aabb_left_max_or_v1[1], currentNode.aabb_left_max_or_v1[2]},
                                       {currentNode.aabb_right_min_or_v2[0], currentNode.aabb_right_min_or_v2[1], currentNode.aabb_right_min_or_v2[2]} };

                *epo += AabbTriIntersectArea(compBounds, tri);

#if BVH3
                if (currentNode.prim_id2 != BvhX::Constants::kInvalidId)
                {
                    tri = { {currentNode.aabb_right_max_or_v3[0], currentNode.aabb_right_max_or_v3[1], currentNode.aabb_right_max_or_v3[2]},
                            {currentNode.aabb_mid_min_or_v4[0], currentNode.aabb_mid_min_or_v4[1], currentNode.aabb_mid_min_or_v4[2]},
                            {currentNode.aabb_mid_max_or_v5[0], currentNode.aabb_mid_max_or_v5[1], currentNode.aabb_mid_max_or_v5[2]} };

                    *epo += AabbTriIntersectArea(compBounds, tri);
                }
#endif

                if (isnan(abs(*epo)))
                {
                    std::cout << "Found NaN at idx " << current << std::endl;
                }
            }
            else
            {
                GetEPO(epo, comp, currentNode.addr_left, tree, totalSum);
#if BVH3
                GetEPO(epo, comp, currentNode.addr_mid_mesh_id2, tree, totalSum);
#endif
                GetEPO(epo, comp, currentNode.addr_right, tree, totalSum);
            }
        }
    }

    inline
        float mm_select(__m128 v, std::uint32_t index)
    {
        _MM_ALIGN16 float temp[4];
        _mm_store_ps(temp, v);
        return temp[index];
    }

    inline
        __m128 aabb_surface_area(__m128 pmin, __m128 pmax)
    {
        auto ext = _mm_sub_ps(pmax, pmin);
        auto xxy = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 1, 0, 0));
        auto yzz = _mm_shuffle_ps(ext, ext, _MM_SHUFFLE(3, 2, 2, 1));
        return _mm_mul_ps(_mm_dp_ps(xxy, yzz, 0xff), _mm_set_ps(2.f, 2.f, 2.f, 2.f));
    }

    void IntersectorLDS::Process(const World &world)
    {
        // If something has been changed we need to rebuild BVH
        if (!m_gpudata->bvh || world.has_changed() || world.GetStateChange() != ShapeImpl::kStateChangeNone)
        {
            // Free previous data
            if (m_gpudata->bvh)
            {
                m_device->DeleteBuffer(m_gpudata->bvh);
            }

            if (m_epoBuffer)
            {
                m_device->DeleteBuffer(m_epoBuffer);
                m_epoBuffer = nullptr;
            }

            // Look up build options for world
            auto type = world.options_.GetOption("bvh.type");
            auto builder = world.options_.GetOption("bvh.builder");
            auto nbins = world.options_.GetOption("bvh.sah.num_bins");
            auto tcost = world.options_.GetOption("bvh.sah.traversal_cost");

            bool use_qbvh = false, use_sah = false;
            int num_bins = (nbins ? static_cast<int>(nbins->AsFloat()) : 64);
            float traversal_cost = (tcost ? tcost->AsFloat() : 10.0f);

#if 0
            if (type && type->AsString() == "qbvh")
            {
                use_qbvh = (m_gpudata->qbvh_prog.executable != nullptr);
            }
#endif

            if (builder && builder->AsString() == "sah")
            {
                use_sah = true;
            }

            // Create the bvh
            BvhX bvh(traversal_cost, num_bins, use_sah);

            auto beginTime = high_resolution_clock::now();
            bvh.Build(world.shapes_.begin(), world.shapes_.end());
            std::cout << duration_cast<microseconds>(high_resolution_clock::now() - beginTime).count() << std::endl;

            // Upload BVH data to GPU memory
            if (!use_qbvh)
            {
                auto bvh_size_in_bytes = bvh.GetSizeInBytes();
                m_gpudata->bvh = m_device->CreateBuffer(bvh_size_in_bytes, Calc::BufferType::kRead);

                // Get the pointer to mapped data
                Calc::Event *e = nullptr;
                BvhX::Node *bvhdata = nullptr;

                m_device->MapBuffer(m_gpudata->bvh, 0, 0, bvh_size_in_bytes, Calc::MapType::kMapWrite, (void **)&bvhdata, &e);

                e->Wait();
                m_device->DeleteEvent(e);

#if PROFILE_TRAVERSAL
                m_epoBuffer = m_device->CreateBuffer(bvh.m_nodecount * sizeof(float) * 2, Calc::BufferType::kRead);

                float *epodata = nullptr;
                m_device->MapBuffer(m_epoBuffer, 0, 0, bvh.m_nodecount * sizeof(float) * 2, Calc::MapType::kMapWrite, (void **)&epodata, &e);
                e->Wait();
                m_device->DeleteEvent(e);

                memset(epodata, 0, bvh.m_nodecount * sizeof(float) * 2);

                m_totalArea = 0.0f;

                m_surfAreaBuffer = m_device->CreateBuffer(bvh.m_nodecount * sizeof(float), Calc::BufferType::kRead);

                float *sadata = nullptr;
                m_device->MapBuffer(m_surfAreaBuffer, 0, 0, bvh.m_nodecount * sizeof(float), Calc::MapType::kMapWrite, (void **)&sadata, &e);
                e->Wait();
                m_device->DeleteEvent(e);
#endif

#if PRINT_TREE
#if BVH3
                printf("===============BVH3==============\n");
#else
                printf("===============BVH2==============\n");
#endif
#endif

                // Copy BVH data
                for (std::size_t i = 0; i < bvh.m_nodecount; ++i)
                {
                    RadeonRays::BvhX::Node& node = bvh.m_nodes[i];
#if BVH3
                    if (Bvh3::IsValid(node))
                    {
#endif
#if PRINT_TREE
#if BVH3
                        printf("#%ull: \tlevel - n/a, \tleft_addr - %u, \tmid_addr - %u, right_addr - %u, \tprim_id - %u, \tprim_id2%u\n",
                            i, /*myData.level,*/ node.addr_left, node.addr_mid_mesh_id2, node.addr_right, node.prim_id, node.prim_id2);
                        printf(" aabb_min_left: x - %f y - %f z - %f \n",
                            node.aabb_left_min_or_v0[0], node.aabb_left_min_or_v0[1], node.aabb_left_min_or_v0[2]);
                        printf(" aabb_max_left: x - %f y - %f z - %f \n",
                            node.aabb_left_max_or_v1[0], node.aabb_left_max_or_v1[1], node.aabb_left_max_or_v1[2]);
                        printf(" aabb_min_min: x - %f y - %f z - %f \n",
                            node.aabb_mid_min_or_v4[0], node.aabb_mid_min_or_v4[1], node.aabb_mid_min_or_v4[2]);
                        printf(" aabb_max_min: x - %f y - %f z - %f \n",
                            node.aabb_mid_max_or_v5[0], node.aabb_mid_max_or_v5[1], node.aabb_mid_max_or_v5[2]);
                        printf(" aabb_min_right: x - %f y - %f z - %f \n",
                            node.aabb_right_min_or_v2[0], node.aabb_right_min_or_v2[1], node.aabb_right_min_or_v2[2]);
                        printf(" aabb_max_right: x - %f y - %f z - %f \n\n",
                            node.aabb_right_max_or_v3[0], node.aabb_right_max_or_v3[1], node.aabb_right_max_or_v3[2]);
#else
                        printf("#%u: \tlevel - %u, \tleft_addr - %u, \tright_addr - %u, \tprim_id - %u\n",
                            i, myData.level, node.addr_left, node.addr_right, node.prim_id);
#endif
#endif

                        bvhdata[i] = bvh.m_nodes[i];

#if PROFILE_TRAVERSAL
                        GetEPO(&epodata[2 * i], i, 0, &bvh);
                        GetEPO(&epodata[(2 * i) + 1], i, 0, &bvh, true);

                        float length = bvh.m_epoDatas[i].aabb_max[0] - bvh.m_epoDatas[i].aabb_min[0];
                        float width = bvh.m_epoDatas[i].aabb_max[1] - bvh.m_epoDatas[i].aabb_min[1];
                        float height = bvh.m_epoDatas[i].aabb_max[2] - bvh.m_epoDatas[i].aabb_min[2];

                        sadata[i] = (2 * (length * width)) + (2 * (length * height)) + (2 * (height * width));

                        if (!BvhX::IsInternal(bvh.m_nodes[i]))
                        {
                            Math::Triangle tri = { {bvh.m_nodes[i].aabb_left_min_or_v0[0], bvh.m_nodes[i].aabb_left_min_or_v0[1], bvh.m_nodes[i].aabb_left_min_or_v0[2]},
                                                   {bvh.m_nodes[i].aabb_left_max_or_v1[0], bvh.m_nodes[i].aabb_left_max_or_v1[1], bvh.m_nodes[i].aabb_left_max_or_v1[2]},
                                                   {bvh.m_nodes[i].aabb_right_min_or_v2[0], bvh.m_nodes[i].aabb_right_min_or_v2[1], bvh.m_nodes[i].aabb_right_min_or_v2[2]} };

                            m_totalArea += (tri.v0 - tri.v1).Cross((tri.v2 - tri.v1)).Magnitude() / 2.0f;

#if BVH3
                            if (bvh.m_nodes[i].prim_id2 != BvhX::Constants::kInvalidId)
                            {
                                tri = { {bvh.m_nodes[i].aabb_right_max_or_v3[0], bvh.m_nodes[i].aabb_right_max_or_v3[1], bvh.m_nodes[i].aabb_right_max_or_v3[2]},
                                        {bvh.m_nodes[i].aabb_mid_min_or_v4[0], bvh.m_nodes[i].aabb_mid_min_or_v4[1], bvh.m_nodes[i].aabb_mid_min_or_v4[2]},
                                        {bvh.m_nodes[i].aabb_mid_max_or_v5[0], bvh.m_nodes[i].aabb_mid_max_or_v5[1], bvh.m_nodes[i].aabb_mid_max_or_v5[2]} };

                                m_totalArea += (tri.v0 - tri.v1).Cross((tri.v2 - tri.v1)).Magnitude() / 2.0f;
                            }
#endif
                        }
#endif
#if BVH3
                    }
#endif
                }

                // Unmap gpu data
                m_device->UnmapBuffer(m_gpudata->bvh, 0, bvhdata, &e);

                e->Wait();
                m_device->DeleteEvent(e);

#if PROFILE_TRAVERSAL
                char filename[1024];
                memset(filename, 0, 1024);
                strcat(filename, "C:\\git\\Baikal_Mine\\build\\");
                strcat(filename, getenv("BAIKAL_MODEL_NAME"));

                char* comma = strstr(filename, ",");
                if (comma != nullptr) { strncpy(comma, "\0", 1); }

                strcat(filename, "_epo.csv");

                std::ofstream file;
                file.open(filename);
                if (file.is_open())
                {
                    file << "index,epo,epoTotal,\n";
                    for (uint32_t i = 0; i < bvh.m_nodecount; i++)
                    {
                        file << i << "," << epodata[2 * i] << "," << epodata[(2 * i) + 1] << ",\n";
                    }
                }
                file.close();

                m_device->UnmapBuffer(m_epoBuffer, 0, epodata, &e);

                e->Wait();
                m_device->DeleteEvent(e);

                m_rootSA = sadata[0];

                m_device->UnmapBuffer(m_surfAreaBuffer, 0, sadata, &e);

                e->Wait();
                m_device->DeleteEvent(e);
#endif
                std::cout << "level count: " << bvh.m_levelcount << std::endl;

                // Select intersection program
                m_gpudata->prog = &m_gpudata->bvh_prog;
            }
            else
            {
#if !BVH3
                QBvhTranslator translator;
                translator.Process(bvh);

                // Update GPU data
                auto bvh_size_in_bytes = translator.GetSizeInBytes();
                m_gpudata->bvh = m_device->CreateBuffer(bvh_size_in_bytes, Calc::BufferType::kRead);

                // Get the pointer to mapped data
                Calc::Event *e = nullptr;
                QBvhTranslator::Node *bvhdata = nullptr;

                m_device->MapBuffer(m_gpudata->bvh, 0, 0, bvh_size_in_bytes, Calc::MapType::kMapWrite, (void **)&bvhdata, &e);

                e->Wait();
                m_device->DeleteEvent(e);

                // Copy BVH data
                std::size_t i = 0;
                for (auto & node : translator.nodes_)
                    bvhdata[i++] = node;

                // Unmap gpu data
                m_device->UnmapBuffer(m_gpudata->bvh, 0, bvhdata, &e);

                e->Wait();
                m_device->DeleteEvent(e);

                // Select intersection program
                m_gpudata->prog = &m_gpudata->qbvh_prog;
#endif
            }

            // Make sure everything is committed
            m_device->Finish(0);
        }
    }

    void IntersectorLDS::Intersect(std::uint32_t queue_idx, const Calc::Buffer *rays, const Calc::Buffer *num_rays,
        std::uint32_t max_rays, Calc::Buffer *hits,
        const Calc::Event *wait_event, Calc::Event **event) const
    {
        std::size_t stack_size = 4 * max_rays * kMaxStackSize;

        // Check if we need to reallocate memory
        if (!m_gpudata->stack || stack_size > m_gpudata->stack->GetSize())
        {
            m_device->DeleteBuffer(m_gpudata->stack);
            m_gpudata->stack = m_device->CreateBuffer(stack_size, Calc::BufferType::kWrite);
        }

        assert(m_gpudata->prog);
        auto &func = m_gpudata->prog->isect_func;

        // Set args
        int arg = 0;

#if SERIALIZE_RAYS
        Calc::Buffer* index = m_device->CreateBuffer(sizeof(uint32_t), Calc::BufferType::kRead);
#if PROFILE_TRAVERSAL
        Calc::Buffer* stats = m_device->CreateBuffer(max_rays * sizeof(Stats), Calc::BufferType::kWrite);
#endif
#endif

        func->SetArg(arg++, m_gpudata->bvh);
        func->SetArg(arg++, rays);
        func->SetArg(arg++, num_rays);
        func->SetArg(arg++, m_gpudata->stack);
        func->SetArg(arg++, hits);
#if SERIALIZE_RAYS
        auto &serial_func = m_gpudata->serial_prog.isect_func;
        int serial_arg = 0;
        serial_func->SetArg(serial_arg++, m_gpudata->bvh);
        serial_func->SetArg(serial_arg++, rays);
        serial_func->SetArg(serial_arg++, num_rays);
        serial_func->SetArg(serial_arg++, m_gpudata->stack);
        serial_func->SetArg(serial_arg++, hits);
        serial_func->SetArg(serial_arg++, index);
#if PROFILE_TRAVERSAL
        auto &profiling_func = m_gpudata->profiling_prog.isect_func;
        int profiling_arg = 0;
        profiling_func->SetArg(profiling_arg++, m_gpudata->bvh);
        profiling_func->SetArg(profiling_arg++, rays);
        profiling_func->SetArg(profiling_arg++, num_rays);
        profiling_func->SetArg(profiling_arg++, m_gpudata->stack);
        profiling_func->SetArg(profiling_arg++, hits);
        profiling_func->SetArg(profiling_arg++, index);
        profiling_func->SetArg(profiling_arg++, stats);
        profiling_func->SetArg(profiling_arg++, m_epoBuffer);
        profiling_func->SetArg(profiling_arg++, m_surfAreaBuffer);
#endif
#endif

        std::size_t localsize = kWorkGroupSize;
        std::size_t globalsize = ((max_rays + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;

        int32_t num_rays_data = 0;
        m_device->ReadBuffer(num_rays, queue_idx, 0, sizeof(int32_t), &num_rays_data, event);

#if SERIALIZE_RAYS
        printf("Starting serialized draw...");

        float* timing_data = new float[num_rays_data];

        for (uint32_t i = 0; i < num_rays_data; i++)
        {
            m_device->WriteBuffer(index, 0, 0, sizeof(uint32_t), &i, event);
            timing_data[i] = m_device->Execute(serial_func, queue_idx, 1, event);
#if PROFILE_TRAVERSAL
            m_device->Execute(profiling_func, queue_idx, 1, event);
#endif
        }

#if PROFILE_TRAVERSAL
        Stats* stats_data = new Stats[max_rays];
        memset(stats_data, 0, sizeof(Stats) * max_rays);
        m_device->ReadBuffer(stats, queue_idx, 0, max_rays * sizeof(Stats), stats_data, event);
#endif
        std::ofstream file;

        char filename[1024];
        memset(filename, 0, 1024);
        strcat(filename, "C:\\git\\Baikal_Mine\\build\\");
        strcat(filename, getenv("BAIKAL_MODEL_NAME"));

        char* comma = strstr(filename, ",");
        if (comma != nullptr) { strncpy(comma, "\0", 1); }

        strcat(filename, ".csv");

        file.open(filename);
        if (file.is_open())
        {
#if PROFILE_TRAVERSAL
            file << "index,time,rayAABBs,rayTris,epoSum,epoTotalSum,totalArea,surfaceAreaSumInt,surfaceAreaSumLeaf,rootSA\n";
            for (uint32_t i = 0; i < num_rays_data; i++)
            {
                file << i << "," << timing_data[i] << "," << stats_data[i].rayAABBs << "," << stats_data[i].rayTris << "," << stats_data[i].epoSum << "," << stats_data[i].epoSumTotal << "," << m_totalArea << "," << stats_data[i].surfaceAreaSumInt << "," << stats_data[i].surfaceAreaSumLeaf << "," << m_rootSA << ",\n";
            }
#else
            file << "index,time,\n";
            for (uint32_t i = 0; i < num_rays_data; i++)
            {
                file << i << "," << timing_data[i] << ",\n";
            }
#endif
        }
        file.close();

#if PROFILE_TRAVERSAL
        delete stats_data;
#endif
        delete timing_data;

        exit(0);
#else
        m_device->Execute(func, queue_idx, globalsize, localsize, event);
#endif
    }

    void IntersectorLDS::Occluded(std::uint32_t queue_idx, const Calc::Buffer *rays, const Calc::Buffer *num_rays,
        std::uint32_t max_rays, Calc::Buffer *hits,
        const Calc::Event *wait_event, Calc::Event **event) const
    {
        std::size_t stack_size = 4 * max_rays * kMaxStackSize;

        // Check if we need to reallocate memory
        if (!m_gpudata->stack || stack_size > m_gpudata->stack->GetSize())
        {
            m_device->DeleteBuffer(m_gpudata->stack);
            m_gpudata->stack = m_device->CreateBuffer(stack_size, Calc::BufferType::kWrite);
        }

        assert(m_gpudata->prog);
        auto &func = m_gpudata->prog->occlude_func;

        // Set args
        int arg = 0;

        func->SetArg(arg++, m_gpudata->bvh);
        func->SetArg(arg++, rays);
        func->SetArg(arg++, num_rays);
        func->SetArg(arg++, m_gpudata->stack);
        func->SetArg(arg++, hits);

        std::size_t localsize = kWorkGroupSize;
        std::size_t globalsize = ((max_rays + kWorkGroupSize - 1) / kWorkGroupSize) * kWorkGroupSize;

        m_device->Execute(func, queue_idx, globalsize, localsize, event);
    }
}
