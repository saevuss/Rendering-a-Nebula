// step2_main_nvdb.cpp
//
// Crab Nebula ray marcher — NanoVDB (.nvdb) grid version.
// Compile (macOS, g++-15 + OpenMP):
/*      g++-15 -fopenmp -std=c++17 \
        -I/opt/homebrew/Cellar/openvdb/13.0.0_1/include \
         step2_main_nvdb.cpp -o step2_main_nvdb        
 g++-15 -fopenmp -std=c++17 -O3 -march=native \
-isysroot $(xcrun --show-sdk-path) \
-I/opt/homebrew/Cellar/openvdb/13.0.0_1/include \
step2_main_nvdb.cpp -o step2_main_nvdb        
         
         
         */ 
//
// Usage:
//   ./step2_main_nvdb                                      -- renders 10 frames (default)
//   ./step2_main_nvdb 24 or ./step2_main_nvdb --frames 24        -- renders 24 frames
//   ./step2_main_nvdb -f 1        a                         -- single frame, no GIF produced
//   ./step2_main_nvdb --nvdb-dir output/nvdb_512                  -- to specify where to take file .bin and .nvdb
//
//
// Required files at runtime:
//   density.nvdb  nii_ha.nvdb  sii_ha.nvdb  sii_sii.nvdb  vel.nvdb  syn.nvdb  gaia_stars.csv

#include "step2_common.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>

// ─────────────────────────────────────────────────────────────────────────────
//  integrate  (NanoVDB version)
//
//  Same rendering equation as the binary version (see main_bin.cpp) but using
//  NanoVDB's built-in sampler for trilinear interpolation and its sparse-tree
//  structure for fast empty-space skipping.
// ─────────────────────────────────────────────────────────────────────────────
static void integrate(const Ray& ray,
                      vec3& L, float& T,
                      const nanovdb::FloatGrid* densityGrid,
                      const nanovdb::FloatGrid* niiGrid,
                      const nanovdb::FloatGrid* siiGrid,
                      const nanovdb::FloatGrid* siiSiiGrid,
                      const nanovdb::FloatGrid* velGrid,
                      const nanovdb::FloatGrid* synGrid,
                      std::default_random_engine& rng,
                      std::uniform_real_distribution<float>& dist)
{
    const float sigma_t    = 0.1f;
    const float emissivity = 5.5f;
    const vec3  synchColor {0.15f, 0.55f, 0.85f};

    T = 1.f;
    L = {0.f, 0.f, 0.f};

    // ── Build NanoVDB ray in world space, then convert to index space ─────────
    nanovdb::Vec3f nvOrig(ray.orig.x, ray.orig.y, ray.orig.z);
    nanovdb::Vec3f nvDir (ray.dir.x,  ray.dir.y,  ray.dir.z);
    nanovdb::math::Ray<float> nvRay(nvOrig, nvDir, 0.f, 1000.f);
    nanovdb::math::Ray<float> idxRay = nvRay.worldToIndexF(*densityGrid);

    // Clip march to the populated index-space bounding box
    float t0 = idxRay.t0();
    float t1 = idxRay.t1();
    if (!idxRay.intersects(densityGrid->indexBBox(), t0, t1))
        return;

    // ── Samplers (trilinear, one per grid) ───────────────────────────────────
    using SamplerT = nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
    SamplerT densitySampler (densityGrid->tree());
    SamplerT niiSampler     (niiGrid->tree());
    SamplerT siiSampler     (siiGrid->tree());
    SamplerT siiSiiSampler  (siiSiiGrid->tree());
    SamplerT velSampler     (velGrid->tree());
    SamplerT synSampler     (synGrid->tree());

    // ── March in index space (1 voxel per step) ───────────────────────────────
    const float stepSize  = 1.0f;                                       // voxels
    const size_t numSteps = std::max((size_t)1,
                                     (size_t)std::ceil((t1-t0) / stepSize));
    const float stride    = (t1 - t0) / (float)numSteps;

    // Physical step length for Beer-Lambert 
    const float worldStride = stride * densityGrid->voxelSize()[0];

    for (size_t n = 0; n < numSteps; ++n)
    {
        float t = t0 + stride * ((float)n + dist(rng));
        t = std::min(t, t1);

        const nanovdb::Vec3f idxPos = idxRay(t);

        const float dens = densitySampler(idxPos);

        // Skip empty voxels — no emission, only update transmittance
        if (dens > 0.01f)
        {
            const float nii_ha  = niiSampler   (idxPos);
            const float sii_ha  = siiSampler   (idxPos);
            const float sii_sii = siiSiiSampler(idxPos);
            const float vel     = velSampler   (idxPos);

            vec3 emColor = nebulaColor(nii_ha, sii_ha, sii_sii, dens, vel);

            T *= exp(-dens * worldStride * sigma_t);
            if (T < 1e-4f) return;

            L += emColor * dens * emissivity * T * worldStride;
        }

        // Synchrotron (PWN) — sampled even in low-density voxels
        float syn = synSampler(idxPos);
        syn = pow(syn, 1.5f);
        // if (syn > 0.02f)
        //     L += synchColor * syn * emissivity * 0.08f * T * worldStride;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  render
// ─────────────────────────────────────────────────────────────────────────────
static void render(int numFrames, const std::string& nvdbDir)
{
    // ── Load NanoVDB grids ────────────────────────────────────────────────────
    //  One handle per file: each .nvdb contains a single FloatGrid at index 0.
    //  Separate handles ensure each grid's memory lifetime is independent.
    fprintf(stderr, "Loading .nvdb grids...\n");
 
    nanovdb::GridHandle densityHandle, niiHandle, siiHandle,
                        siiSiiHandle,  velHandle, synHandle;
 
    const nanovdb::FloatGrid* densityGrid = nullptr;
    const nanovdb::FloatGrid* niiGrid     = nullptr;
    const nanovdb::FloatGrid* siiGrid     = nullptr;
    const nanovdb::FloatGrid* siiSiiGrid  = nullptr;
    const nanovdb::FloatGrid* velGrid     = nullptr;
    const nanovdb::FloatGrid* synGrid     = nullptr;
 
    try {
        densityHandle = nanovdb::io::readGrid(nvdbDir + "/density.nvdb");
        niiHandle     = nanovdb::io::readGrid(nvdbDir + "/nii_ha.nvdb");
        siiHandle     = nanovdb::io::readGrid(nvdbDir + "/sii_ha.nvdb");
        siiSiiHandle  = nanovdb::io::readGrid(nvdbDir + "/sii_sii.nvdb");
        velHandle     = nanovdb::io::readGrid(nvdbDir + "/vel.nvdb");
        synHandle     = nanovdb::io::readGrid(nvdbDir + "/syn.nvdb");
 
        densityGrid = densityHandle.grid<float>(0);
        niiGrid     = niiHandle    .grid<float>(0);
        siiGrid     = siiHandle    .grid<float>(0);
        siiSiiGrid  = siiSiiHandle .grid<float>(0);
        velGrid     = velHandle    .grid<float>(0);
        synGrid     = synHandle    .grid<float>(0);
 
        if (!densityGrid || !niiGrid || !siiGrid ||
            !siiSiiGrid  || !velGrid || !synGrid)
        {
            fprintf(stderr, "Error: one or more grids are null after loading.\n");
            return;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Exception loading .nvdb: %s\n", e.what());
        return;
    }
 
    // ── Diagnostic: grid geometry ─────────────────────────────────────────────
    {
        auto vs    = densityGrid->voxelSize();
        auto ibbox = densityGrid->indexBBox();
        auto wbbox = densityGrid->worldBBox();
        fprintf(stderr, "voxelSize : %.4f %.4f %.4f\n", vs[0], vs[1], vs[2]);
        fprintf(stderr, "indexBBox : (%d,%d,%d) -> (%d,%d,%d)\n",
            ibbox.min()[0], ibbox.min()[1], ibbox.min()[2],
            ibbox.max()[0], ibbox.max()[1], ibbox.max()[2]);
        fprintf(stderr, "worldBBox : (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f)\n",
            wbbox.min()[0], wbbox.min()[1], wbbox.min()[2],
            wbbox.max()[0], wbbox.max()[1], wbbox.max()[2]);
 
        auto sb = synGrid->worldBBox();
        fprintf(stderr, "syn world : (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f)\n",
            sb.min()[0], sb.min()[1], sb.min()[2],
            sb.max()[0], sb.max()[1], sb.max()[2]);
    }
 
    // ── Star catalog ──────────────────────────────────────────────────────────
    std::vector<Star> stars = loadStars("input_data/gaia_stars.csv");
 
    // ── Image buffer ──────────────────────────────────────────────────────────
    constexpr size_t width  = 800;
    constexpr size_t height = 800;
    auto imgbuf = std::make_unique<unsigned char[]>(width * height * 3);
 
    const float frameAspectRatio = (float)width / (float)height;
    const float fov   = 45.f;
    const float focal = tan((float)M_PI / 180.f * fov * 0.5f);
 
    // ── Per-thread RNG ────────────────────────────────────────────────────────
    std::vector<std::default_random_engine> rngs((size_t)omp_get_max_threads());
    std::random_device rd;
    for (int t = 0; t < omp_get_max_threads(); ++t)
        rngs[(size_t)t].seed(rd() + (unsigned)t);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
 
    fprintf(stderr, "Available threads: %d\n", omp_get_max_threads());
    fprintf(stderr, "Rendering %d frame%s\n", numFrames, numFrames > 1 ? "s" : "");
 
    // ── Grid centre (orbit target) ────────────────────────────────────────────
    auto wbbox = densityGrid->worldBBox();
    const vec3 gridCenter{
        (float)(wbbox.min()[0] + wbbox.max()[0]) * 0.5f,
        (float)(wbbox.min()[1] + wbbox.max()[1]) * 0.5f,
        (float)(wbbox.min()[2] + wbbox.max()[2]) * 0.5f
    };
    fprintf(stderr, "Grid centre: (%.2f, %.2f, %.2f)\n",
            gridCenter.x, gridCenter.y, gridCenter.z);
 
    // ── Animation loop ────────────────────────────────────────────────────────
    const std::string outDir = "nebula_nvdb";
    std::filesystem::create_directories(outDir);
 
    for (int frame = 0; frame < numFrames; ++frame)
    {
        fprintf(stderr, "\n=== Frame %d / %d ===\n", frame+1, numFrames);
 
        const float angle  = 2.f * (float)M_PI * frame / numFrames;
        const Matrix cam   = buildOrbitCamera(angle, 100.f, gridCenter);
        const vec3 rayOrig = transformPoint(cam, {0,0,0});
 
        std::atomic<int> rowsDone{0};
 
        #pragma omp parallel for schedule(dynamic, 4)
        for (int j = 0; j < (int)height; ++j)
        {
            auto& rng = rngs[(size_t)omp_get_thread_num()];
 
            for (int i = 0; i < (int)width; ++i)
            {
                vec3 rayDir{
                     (2.f * ((float)i + 0.5f) / width  - 1.f) * focal,
                    -(2.f * ((float)j + 0.5f) / height - 1.f) * focal / frameAspectRatio,
                    -1.f
                };
                rayDir *= cam;
                rayDir.nor();
 
                Ray ray(rayOrig, rayDir);
 
                vec3  L{0,0,0};
                float Tr = 1.f;
 
                integrate(ray, L, Tr,
                          densityGrid, niiGrid, siiGrid, siiSiiGrid, velGrid, synGrid,
                          rng, dist);
 
                const float starBrightness = 40.f;
                L += starContribution(ray, stars, starBrightness) * Tr;
 
                vec3 mapped = reinhard(background_color * Tr + L);
                const size_t off = ((size_t)j * width + (size_t)i) * 3;
                imgbuf[off+0] = (unsigned char)(std::clamp(mapped.x, 0.f, 1.f) * 255.f);
                imgbuf[off+1] = (unsigned char)(std::clamp(mapped.y, 0.f, 1.f) * 255.f);
                imgbuf[off+2] = (unsigned char)(std::clamp(mapped.z, 0.f, 1.f) * 255.f);
            }
 
            #pragma omp critical
            { fprintf(stderr, "\r%.1f%%", 100.f * ++rowsDone / height); }
        }
 
        // ── Save PPM ──────────────────────────────────────────────────────────
        char filename[64];
        snprintf(filename, sizeof(filename), "frame_%04d.ppm", frame);
        std::ofstream ofs(outDir + "/" + filename, std::ios::binary);
        ofs << "P6\n" << width << " " << height << "\n255\n";
        ofs.write(reinterpret_cast<const char*>(imgbuf.get()),
                  (std::streamsize)(width * height * 3));
    }
 
    fprintf(stderr, "\nAll frames done.\n");
 
    // ── GIF assembly (only when more than one frame was rendered) ─────────────
    if (numFrames > 1)
        makeGif(outDir, "frame_%04d.ppm", outDir + "/anim.gif", /*fps=*/6);
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    {
        const float ra  = 84.41f * (float)M_PI / 180.f;
        const float dec = 21.14f * (float)M_PI / 180.f;
        vec3 zetaTau{ cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec) };
        Matrix icrsToWorld = buildICRSToWorldMatrix();
        vec3 v = applyRotation(icrsToWorld, zetaTau);
        v.nor();
        fprintf(stderr, "zeta Tauri world dir: (%+.3f, %+.3f, %+.3f)  (expect Z ~= -1)\n",
                v.x, v.y, v.z);
    }
 
    const int numFrames = parseArgs(argc, argv, /*default=*/10);

    std::string nvdbDir = "output/nvdb_512"; // default
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--nvdb-dir" && i + 1 < argc)
            nvdbDir = argv[++i];
    }

    render(numFrames, nvdbDir);
    return 0;
}