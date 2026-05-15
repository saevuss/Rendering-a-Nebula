// step2_main_bin.cpp
//
// Crab Nebula ray marcher -- binary (.bin) grid version.
//
// Compile (macOS, g++-15 + OpenMP):
//   g++-15 -fopenmp -std=c++17 step2_main_bin.cpp -o step2_main_bin
/* g++-15 -fopenmp -std=c++17 -O3 -march=native \
-isysroot "$(xcrun --show-sdk-path)" \
-I/opt/homebrew/Cellar/openvdb/13.0.0_1/include \
step2_main_bin.cpp -o step2_main_bin

 */
//
// Usage:
//   ./step2_main_bin                                      -- renders 10 frames (default)
//   ./step2_main_bin 24 or ./step2_main_bin --frames 24         -- renders 24 frames
//   ./step2_main_bin -f 1                                 -- single frame, no GIF produced
//   ./main --bin-dir output/bin_512                 -- to specify where to take file .bin and .nvdb
//
// Required files at runtime:
//   density.bin  nii_ha.bin  sii_ha.bin  sii_sii.bin  vel.bin  syn.bin  gaia_stars.csv

#include "step2_common.h"

// ─────────────────────────────────────────────────────────────────────────────
//  loadBinary
//
//  Reads exactly N floats from a raw binary file into a pre-allocated buffer.
//  Throws std::runtime_error if the file cannot be opened or is too short.
// ─────────────────────────────────────────────────────────────────────────────
static void loadBinary(const std::string& path, std::unique_ptr<float[]>& data, size_t N)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    ifs.read(reinterpret_cast<char*>(data.get()), sizeof(float) * N);
    if (!ifs)
        throw std::runtime_error("Error reading file (incomplete or corrupted): " + path);
}

// ─────────────────────────────────────────────────────────────────────────────
//  integrate  (binary grid version, better explained in the report)
//
//  Ray-marches through the volume between tMin and tMax, accumulating: emitted radiance (L) 
//  and transmittance remaining at the end of the ray (T)
//
//  Synchrotron: the pulsar-wind nebula (PWN) inner blue glow is stored in a
//  separate grid and added on top of the line-emission colour.  Its values
//  are raised to the power 1.5 to concentrate the contribution in the densest
//  core and suppress the noisy background.
// ─────────────────────────────────────────────────────────────────────────────
static void integrate(const Ray& ray, float tMin, float tMax,
                      vec3& L, float& T,
                      const Grid& density, const Grid& niiGrid,
                      const Grid& siiGrid, const Grid& siiSiiGrid,
                      const Grid& velGrid, const Grid& synGrid,
                      std::default_random_engine& rng,
                      std::uniform_real_distribution<float>& dist)
{
    const float stepSize  = 0.117f;
    const float sigma_t   = 0.1f;  // extinction = absorption (scattering negligible)
    const float emissivity = 5.5f;
    const vec3  synchColor{0.15f, 0.55f, 0.85f}; // PWN blue

    const size_t numSteps = (size_t)std::ceil((tMax - tMin) / stepSize);
    const float  stride   = (tMax - tMin) / (float)numSteps; // exact step to stay within [tMin, tMax]

    T = 1.f;
    L = {0.f, 0.f, 0.f};

    for (size_t n = 0; n < numSteps; ++n)
    {
        float t = tMin + stride * ((float)n + dist(rng)); // jittered sample position
        t = std::min(t, tMax);
        const vec3 p = ray(t);

        const float dens    = lookup(density,    p);
        const float nii_ha  = lookup(niiGrid,    p);
        const float sii_ha  = lookup(siiGrid,    p);
        const float sii_sii = lookup(siiSiiGrid, p);
        const float vel     = lookup(velGrid,    p);
              float syn     = lookup(synGrid,    p);
        syn = pow(syn, 1.5f); // suppress noisy background, boost dense core

        // Transmittance update (Beer-Lambert)
        T *= exp(-dens * stride * sigma_t);
        if (T < 1e-4f) break; //early exit

        // Emission accumulation
        vec3 emColor = nebulaColor(nii_ha, sii_ha, sii_sii, dens, vel);
        L += emColor * dens * emissivity * T * stride;

        // Synchrotron (PWN) integration
        if (syn > 0.02f)
             L += synchColor * syn * emissivity * 0.08f * T * stride;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  render
// ─────────────────────────────────────────────────────────────────────────────
static void render(int numFrames, size_t N, const std::string& binDir)
{
    // ── Load all six binary grids ─────────────────────────────────────────────
    //  Resolution is 512^3 -> 512*512*512 * 4 bytes ~ 536 MB per file.
    const size_t NVOX = N * N * N; 
 
    auto makeGrid = [&](const std::string& filename) {
        Grid g;
        g.baseResolution = N;
        g.densityData    = std::make_unique<float[]>(NVOX);
        loadBinary(binDir + "/" + filename, g.densityData, NVOX);
        return g;
    };
 
    Grid densityGrid, niiGrid, siiGrid, siiSiiGrid, velGrid, synGrid;
    try {
        fprintf(stderr, "Loading binary grids...\n");
        densityGrid = makeGrid("density.bin");
        niiGrid     = makeGrid("nii_ha.bin");
        siiGrid     = makeGrid("sii_ha.bin");
        siiSiiGrid  = makeGrid("sii_sii.bin");
        velGrid     = makeGrid("vel.bin");
        synGrid     = makeGrid("syn.bin");
        fprintf(stderr, "All grids loaded.\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading grids: %s\n", e.what());
        return;
    }
 
    // ── Star catalog ──────────────────────────────────────────────────────────
    std::vector<Star> stars = loadStars("gaia_stars.csv");
 
    // ── Image buffer ──────────────────────────────────────────────────────────
    constexpr size_t width  = 800;
    constexpr size_t height = 800;
    auto imgbuf = std::make_unique<unsigned char[]>(width * height * 3);
 
    const float frameAspectRatio = (float)width / (float)height;
    const float fov   = 45.f;
    const float focal = tan(M_PI / 180.f * fov * 0.5f);
 
    // ── Per-thread RNG (one generator per OpenMP thread) ─────────────────────
    std::vector<std::default_random_engine> rngs((size_t)omp_get_max_threads());
    std::random_device rd;
    for (int t = 0; t < omp_get_max_threads(); ++t)
        rngs[(size_t)t].seed(rd() + (unsigned)t);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
 
    fprintf(stderr, "Available threads: %d\n", omp_get_max_threads());
    fprintf(stderr, "Rendering %d frame%s\n", numFrames, numFrames > 1 ? "s" : "");
 
    // ── Animation loop ────────────────────────────────────────────────────────
    const std::string outDir = "nebula_bin";
    std::filesystem::create_directories(outDir);
 
    for (int frame = 0; frame < numFrames; ++frame)
    {
        fprintf(stderr, "\n=== Frame %d / %d ===\n", frame+1, numFrames);
 
        // Full 360-degree orbit: angle 0 at frame 0, 2pi at frame numFrames
        const float angle = 2.f * (float)M_PI * frame / numFrames;
 
        const vec3   gridCenter{0.f, 0.f, 0.f}; // bin grids are centred at origin
        const Matrix cam     = buildOrbitCamera(angle, 100.f, gridCenter);
        const vec3   rayOrig = transformPoint(cam, {0,0,0});
 
        std::atomic<int> rowsDone{0};
 
        #pragma omp parallel for schedule(dynamic, 4)
        for (int j = 0; j < (int)height; ++j)
        {
            auto& rng = rngs[(size_t)omp_get_thread_num()];
 
            for (int i = 0; i < (int)width; ++i)
            {
                // Camera-space ray direction
                vec3 rayDir{
                     (2.f * ((float)i + 0.5f) / width  - 1.f) * focal,
                    -(2.f * ((float)j + 0.5f) / height - 1.f) * focal / frameAspectRatio,
                    -1.f
                };
                rayDir *= cam; // rotate to world space
                rayDir.nor();
 
                Ray ray(rayOrig, rayDir);
 
                vec3  L{0,0,0};
                float T = 1.f;
 
                // March only inside the bounding box
                float tmin, tmax;
                if (raybox(ray, densityGrid.bounds, tmin, tmax))
                    integrate(ray, tmin, tmax, L, T,
                              densityGrid, niiGrid, siiGrid, siiSiiGrid, velGrid, synGrid,
                              rng, dist);
 
                // Add star contributions attenuated by nebula transmittance
                const float starBrightness = 40.f;
                L += starContribution(ray, stars, starBrightness) * T;
 
                // Tone map and write to buffer
                vec3 mapped = reinhard(background_color * T + L);
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
        makeGif(outDir, "frame_%04d.ppm", outDir + "/anim.gif", /*fps=*/8);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    // Sanity check: ζ Tauri should appear almost directly in front of the camera
    // (world Z ≈ -1) because it is angularly very close to the Crab Nebula.
    {
        const float ra  = 84.41f * (float)M_PI / 180.f;
        const float dec = 21.14f * (float)M_PI / 180.f;
        vec3 zetaTau{ cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec) };
        Matrix icrsToWorld = buildICRSToWorldMatrix();
        vec3 v = applyRotation(icrsToWorld, zetaTau);
        v.nor();
        fprintf(stderr, "ζ Tauri world dir: (%+.3f, %+.3f, %+.3f)  (expect Z ≈ -1)\n",
                v.x, v.y, v.z);
    }

    const int numFrames = parseArgs(argc, argv, /*default=*/10);
    std::string binDir = "output/bin_512";
    // da argv: ./step2_main_bin --bin-dir output/bin_256
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bin-dir" && i + 1 < argc)
            binDir = argv[++i];
    }
    const size_t N = extractResolution(binDir);
    render(numFrames, N, binDir);
    return 0;
}