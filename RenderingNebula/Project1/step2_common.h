#pragma once //to include file only once in the compilation

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable : 4146)

#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <cassert>
#include <filesystem>
#include <string>
#include <sstream>
#include <atomic>
#include <omp.h>

struct Matrix
{
    const float  operator[](size_t i) const { return m[i]; }
    float&       operator[](size_t i)       { return m[i]; }
    float m[16];
};

struct vec3
{
    float x{0}, y{0}, z{0};

    float length() const { return sqrt(x*x + y*y + z*z); }

    // Normalise in-place; returns *this for chaining.
    vec3& nor()
    {
        float len = x*x + y*y + z*z;
        if (len != 0) len = sqrt(len);
        x /= len; y /= len; z /= len;
        return *this;
    }

    // Dot product
    float operator*(const vec3& v) const { return x*v.x + y*v.y + z*v.z; }

    vec3  operator-(const vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    vec3  operator-()              const { return {-x, -y, -z}; }
    vec3  operator+(const vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }

    vec3& operator+=(const vec3& v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
    vec3& operator*=(float r)       { x*=r;   y*=r;   z*=r;   return *this; }

    vec3  operator*(float r)        const { return {x*r,   y*r,   z*r};   }
    vec3  operator/(const vec3& v)  const { return {x/v.x, y/v.y, z/v.z}; }

    // Scalar on the left: r * vec
    friend vec3 operator*(float r, const vec3& v) { return {v.x*r, v.y*r, v.z*r}; }
    // Component-wise: r / vec  (used to build invDir in Ray)
    friend vec3 operator/(float r, const vec3& v) { return {r/v.x, r/v.y, r/v.z}; }

    friend std::ostream& operator<<(std::ostream& os, const vec3& v)
    { os << v.x << " " << v.y << " " << v.z; return os; }

    vec3 operator*(const Matrix& m) const
    {
        return {
            m[0]*x + m[4]*y + m[8]*z,
            m[1]*x + m[5]*y + m[9]*z,
            m[2]*x + m[6]*y + m[10]*z
        };
    }
    vec3& operator*=(const Matrix& m) { *this = *this * m; return *this; }
};

// Cross product (free function for readability)
inline vec3 cross(const vec3& a, const vec3& b)
{
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}


inline vec3 applyRotation(const Matrix& m, const vec3& v)
{
    return {
        m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
        m.m[3]*v.x + m.m[4]*v.y + m.m[5]*v.z,
        m.m[6]*v.x + m.m[7]*v.y + m.m[8]*v.z
    };
}
// ─────────────────────────────────────────────────────────────────────────────
//  transformPoint
//
//  Applies the full 4×4 column-major matrix to a point (includes translation).
// ─────────────────────────────────────────────────────────────────────────────
inline vec3 transformPoint(const Matrix& m, const vec3& p)
{
    return {
        m[0]*p.x + m[4]*p.y + m[8]*p.z  + m[12],
        m[1]*p.x + m[5]*p.y + m[9]*p.z  + m[13],
        m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14]
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  GRID  (binary voxel volume — used by main_bin only)
//
//  Stores a 3-D density field as a flat float array in Z-major order:
//    index = (z * N + y) * N + x    where N = baseResolution
// ─────────────────────────────────────────────────────────────────────────────
struct Grid
{
    size_t baseResolution = 512;
    std::unique_ptr<float[]> densityData;
    vec3 bounds[2]{ vec3{-30,-30,-30}, vec3{30,30,30} };
 
    float operator()(int xi, int yi, int zi) const
    {
        if (xi < 0 || xi > (int)baseResolution-1 ||
            yi < 0 || yi > (int)baseResolution-1 ||
            zi < 0 || zi > (int)baseResolution-1)
            return 0.f;
        return densityData[(zi * baseResolution + yi) * baseResolution + xi];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  RAY
//
//  Stores origin, direction, its reciprocal (invDir), and the sign of each
//  invDir component.  Pre-computing invDir turns divisions into multiplications
//  in the ray-box intersection loop.
// ─────────────────────────────────────────────────────────────────────────────
struct Ray
{
    Ray(const vec3& orig, const vec3& dir) : orig(orig), dir(dir)
    {
        invDir   = 1.f / dir;               // component-wise reciprocal
        sign[0]  = (invDir.x < 0);          
        sign[1]  = (invDir.y < 0);
        sign[2]  = (invDir.z < 0);
    }

    // Point along the ray at distance t from the origin
    vec3 operator()(float t) const { return orig + dir * t; }

    vec3 orig;
    vec3 dir;
    vec3 invDir;
    bool sign[3];
};

constexpr vec3 background_color{0, 0, 0};

struct Star
{
    vec3  dir;        // unit vector in world space pointing toward the star
    float brightness; // linear flux: mag 0 → 1.0,  mag 5 → 0.01,  mag 10 → 0.0001
    vec3  color;      // RGB derived from Gaia BP–RP colour index (Ballesteros + Helland)
};

// ─────────────────────────────────────────────────────────────────────────────
//  buildICRSToWorldMatrix
//
//  Builds a 3×3 rotation that maps ICRS (Gaia catalog) coordinates to world
//  space so that the Crab Nebula centre points toward -Z world, which is the camera's forward direction at frame 0.
//
//  Construction:
//    camZ  = -crabDir           (we look *toward* the Crab, so forward = -camZ)
//    camX  = northPole × camZ   (right = up_ref × forward, lies in the sky plane)
//    camY  = camZ × camX        (up, orthogonal by construction)
//
// ─────────────────────────────────────────────────────────────────────────────
inline Matrix buildICRSToWorldMatrix()
{
    const float raCrab  = (float)(83.63 * M_PI / 180.0);
    const float decCrab = (float)(22.01 * M_PI / 180.0);

    vec3 crabDir{
        cos(decCrab) * cos(raCrab),
        cos(decCrab) * sin(raCrab),
        sin(decCrab)
    };

    vec3 camZ  = {-crabDir.x, -crabDir.y, -crabDir.z};  // -forward
    vec3 northPole = {0.f, 0.f, 1.f};

    vec3 camX = cross(northPole, camZ); 
    camX.nor();
    vec3 camY = cross(camZ, camX);      
    camY.nor();

    // Store rows: each row is one world axis expressed in ICRS components
    Matrix m;
    m.m[0]=camX.x; m.m[1]=camX.y; m.m[2]=camX.z;
    m.m[3]=camY.x; m.m[4]=camY.y; m.m[5]=camY.z;
    m.m[6]=camZ.x; m.m[7]=camZ.y; m.m[8]=camZ.z;
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
//  buildOrbitCamera
//
//  Places the camera on a circle of given radius around `target` in the XZ
//  plane, rotating by `orbitAngle` radians.  Returns a column-major 4×4
//  camera-to-world matrix.
// ─────────────────────────────────────────────────────────────────────────────
inline Matrix buildOrbitCamera(float orbitAngle, float distance, vec3 target)
{
    vec3 camPos{
        target.x + distance * sin(orbitAngle),
        target.y,
        target.z + distance * cos(orbitAngle)
    };

    vec3 forward{
        (target.x - camPos.x) / distance,
        (target.y - camPos.y) / distance,
        (target.z - camPos.z) / distance
    };

    vec3 worldUp{0.f, 1.f, 0.f};
    vec3 right = cross(forward, worldUp); right.nor();
    vec3 up    = cross(right, forward);   up.nor();
    vec3 camZ  = {-forward.x, -forward.y, -forward.z};

    Matrix m;
    m.m[0]=right.x;  m.m[4]=up.x;  m.m[8] =camZ.x;  m.m[12]=camPos.x;
    m.m[1]=right.y;  m.m[5]=up.y;  m.m[9] =camZ.y;  m.m[13]=camPos.y;
    m.m[2]=right.z;  m.m[6]=up.z;  m.m[10]=camZ.z;  m.m[14]=camPos.z;
    m.m[3]=0.f;      m.m[7]=0.f;   m.m[11]=0.f;      m.m[15]=1.f;
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadStars
//
//  Reads a Gaia CSV with columns: source_id, ra, dec, phot_g_mean_mag, bp_rp
//  Converts each entry to a Star:
//    • direction: spherical ICRS → Cartesian → rotated to world space
//    • brightness: Pogson's law  
//    • color: bp_rp Ballesteros 2012 + RGB (Helland formula)
//
//  Stars fainter than brightness < 1e-6 are skipped (negligible contribution).
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<Star> loadStars(const std::string& csvPath)
{
    std::vector<Star> stars;
    std::ifstream f(csvPath);
    if (!f.is_open()) {
        fprintf(stderr, "Warning: cannot open %s — rendering without stars\n", csvPath.c_str());
        return stars;
    }

    std::string line;
    std::getline(f, line); // skip header row

    while (std::getline(f, line))
    {
        long long sid;
        double ra, dec, mag, bprp;
        if (sscanf(line.c_str(), "%lld,%lf,%lf,%lf, %lf", &sid, &ra, &dec, &mag, &bprp) != 5)
            continue;

        const float raRad  = (float)(ra  * M_PI / 180.0);
        const float decRad = (float)(dec * M_PI / 180.0);

        vec3 dir{
            cos(decRad) * cos(raRad),
            cos(decRad) * sin(raRad),
            sin(decRad)
        };

        // Pogson's law: magnitude → linear flux  (mag 0 = flux 1.0)
        const float brightness = pow(10.f, (0.f - (float)mag) / 2.5f);
        if (brightness < 1e-6f) continue;

        // Rotate from ICRS to world space (matrix built once, cached as static)
        static Matrix icrsToWorld = buildICRSToWorldMatrix();
        vec3 dirWorld = applyRotation(icrsToWorld, dir);
        dirWorld.nor();

        vec3 starColor{1.f, 1.f, 1.f}; // default white when bp_rp is missing

        if ((float)bprp > -99.f)
        {
            float T = 4600.f * (1.f/(0.92f*(float)bprp + 1.7f)
                              + 1.f/(0.92f*(float)bprp + 0.62f));
            T = std::clamp(T, 1000.f, 40000.f);
            float t = T / 100.f;

            float r = (T <= 6600.f)
                ? 1.f
                : std::clamp(329.698727446f * powf(t-60.f, -0.1332047592f) / 255.f, 0.f, 1.f);

            float g = (T <= 6600.f)
                ? std::clamp((99.4708025861f  * logf(t)      - 161.1195681661f) / 255.f, 0.f, 1.f)
                : std::clamp((288.1221695283f * powf(t-60.f, -0.0755148492f))   / 255.f, 0.f, 1.f);

            float b = (T >= 6600.f) ? 1.f
                    : (T <= 1900.f) ? 0.f
                    : std::clamp((138.5177312231f * logf(t-10.f) - 305.0447927307f) / 255.f, 0.f, 1.f);

            starColor = {r, g, b};
        }

        stars.push_back(Star{dirWorld, brightness, starColor});
    }

    fprintf(stderr, "Stars loaded: %zu\n", stars.size());
    return stars;
}

// ─────────────────────────────────────────────────────────────────────────────
//  starContribution
//
//  Accumulates the colour contribution of all stars visible along a ray.
// ─────────────────────────────────────────────────────────────────────────────
inline vec3 starContribution(const Ray& ray, const std::vector<Star>& stars,
                              float starBrightness)
{
    vec3  result{0,0,0};
    const float sigma        = 0.001f;
    const float twoSigmaSq   = 2.f * sigma * sigma;

    for (const auto& star : stars)
    {
        const float cosTheta = ray.dir * star.dir;
        //  Early-exit: if cosθ < 0.999 the star is more than ~2.5° away — the
        //  Gaussian is already negligible
        if (cosTheta < 0.999f) continue; 

        const float theta2  = 2.f * (1.f - cosTheta);
        const float profile = exp(-theta2 / twoSigmaSq);
        const float contrib = star.brightness * starBrightness * profile;

        result.x += contrib * star.color.x;
        result.y += contrib * star.color.y;
        result.z += contrib * star.color.z;
    }
    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
//  nebulaColor
//
//  Maps emission-line ratio data from SITELLE/ORB (Martin et al. 2021) to RGB 
//  (explained in the report)
// ─────────────────────────────────────────────────────────────────────────────
inline vec3 nebulaColor(float nii_ha, float sii_ha, float sii_sii, float density, float vel)
{
    // 1. Diagnostic False-Color Palette
    vec3 ha_color  { 0.02f, 0.25f, 0.90f }; 
    vec3 nii_color { 0.95f, 0.05f, 0.05f }; 
    vec3 sii_color { 1.00f, 0.50f, 0.00f }; 

    // 2. Direct Input Clamping (Linear Physical Space)
    float t_nii = std::clamp(nii_ha, 0.f, 1.f);
    float t_sii = std::clamp(sii_ha, 0.f, 1.f);

    // 3. Additive-Weighted Radiance Mixing
    float w_ha  = (1.f - t_nii) * (1.f - t_sii) * 0.6f;
    float w_nii = t_nii * (1.f - t_sii * 0.4f);
    float w_sii = t_sii;

    vec3 baseColor = ha_color * w_ha + nii_color * w_nii + sii_color * w_sii;

    // 4. Electron Density Modulation (Volumetric Chiaroscuro)
    if (density > 0.f && sii_sii > 0.05f) {
        float t_dense = std::clamp((1.f - sii_sii) * 0.5f, 0.f, 1.f);
        baseColor *= (1.f - t_dense * 0.5f);
    }

    // 5. Kinematic Doppler Shift
    if (density > 0.f && vel > 0.f) {
        float doppler  = (vel - 0.5f) * 2.f; 
        float strength = 0.12f;
        baseColor.x += doppler * strength;
        baseColor.z -= doppler * strength;
        baseColor.x = std::clamp(baseColor.x, 0.f, 1.f);
        baseColor.z = std::clamp(baseColor.z, 0.f, 1.f);
    }

    return baseColor;
}


// ─────────────────────────────────────────────────────────────────────────────
//  reinhard  (luminance-based tone mapping)
//  compute how bright the pixel is overall, scale all three channels by the same factor 
// ─────────────────────────────────────────────────────────────────────────────
inline vec3 reinhard(vec3 c, float exposure = 1.4f)
{
    c.x *= exposure; c.y *= exposure; c.z *= exposure;

    // Perceptual luminance weights (CIE/sRGB standard)
    float lum   = 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
    float scale = 1.0f / (1.0f + lum);
    return {c.x*scale, c.y*scale, c.z*scale};
}


 
// ─────────────────────────────────────────────────────────────────────────────
//  raybox
//
//  Tests whether a ray intersects an axis-aligned bounding box and, if so,
//  returns the entry (tmin) and exit (tmax) distances along the ray.
// ─────────────────────────────────────────────────────────────────────────────
inline bool raybox(const Ray& ray, const vec3 bounds[2], float& tmin, float& tmax)
{
    // ── X and Y slabs ────────────────────────────────────────────────────────
    float a = bounds[    ray.sign[0]].x - ray.orig.x;
    float b = bounds[1 - ray.sign[0]].x - ray.orig.x;
    float c = bounds[    ray.sign[1]].y - ray.orig.y;
    float d = bounds[1 - ray.sign[1]].y - ray.orig.y;
 
    float x0 = (a == 0.f) ? 0.f : a * ray.invDir.x;
    float x1 = (b == 0.f) ? 0.f : b * ray.invDir.x;
    float y0 = (c == 0.f) ? 0.f : c * ray.invDir.y;
    float y1 = (d == 0.f) ? 0.f : d * ray.invDir.y;
 
    if (x0 > y1 || y0 > x1) return false; // X and Y intervals don't overlap
 
    tmin = (y0 > x0) ? y0 : x0;
    tmax = (y1 < x1) ? y1 : x1;
 
    // ── Z slab ───────────────────────────────────────────────────────────────
    float e = bounds[    ray.sign[2]].z - ray.orig.z;
    float f = bounds[1 - ray.sign[2]].z - ray.orig.z;
 
    float z0 = (e == 0.f) ? 0.f : e * ray.invDir.z;
    float z1 = (f == 0.f) ? 0.f : f * ray.invDir.z;
 
    if (tmin > z1 || z0 > tmax) return false; // Z doesn't overlap with XY
 
    tmin = std::max(z0, tmin);
    tmax = std::min(z1, tmax);
    return true;
}
 
// ─────────────────────────────────────────────────────────────────────────────
//  lookup  — trilinear interpolation on a Grid
//
//  Given a world-space point p, returns the density by interpolating among the
//  8 surrounding voxels.  
// ─────────────────────────────────────────────────────────────────────────────
inline float lookup(const Grid& grid, const vec3& p)
{
    const vec3 gridSize = grid.bounds[1] - grid.bounds[0];
    const vec3 pLocal   = (p - grid.bounds[0]) / gridSize;          // [0,1]³
    const vec3 pVoxel   = pLocal * (float)grid.baseResolution;      // voxel space (float)
 
    // Shift so that floor gives the lower-left-back voxel index
    const vec3 pLattice = {pVoxel.x - 0.5f, pVoxel.y - 0.5f, pVoxel.z - 0.5f};
 
    const int xi = (int)std::floor(pLattice.x);
    const int yi = (int)std::floor(pLattice.y);
    const int zi = (int)std::floor(pLattice.z);
 
    float value = 0.f;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
            {
                float wx = 1.f - std::abs(pLattice.x - (xi + i));
                float wy = 1.f - std::abs(pLattice.y - (yi + j));
                float wz = 1.f - std::abs(pLattice.z - (zi + k));
                value += wx * wy * wz * grid(xi+i, yi+j, zi+k);
            }
    return value;
}

// ─────────────────────────────────────────────────────────────────────────────
//  parseArgs
//
//  Reads the number of frames from the command line.
//  Accepted forms:
//    ./main_bin 24
//    ./main_bin --frames 24
//    ./main_bin -f 24
//
//  Returns defaultFrames if no valid argument is found.
// ─────────────────────────────────────────────────────────────────────────────
inline int parseArgs(int argc, char* argv[], int defaultFrames = 10)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
 
        // Named flag: --frames N  or  -f N
        if (arg == "--frames" || arg == "-f")
        {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: '%s' requires a positive integer argument.\n", arg.c_str());
                exit(1);
            }
            int n = std::atoi(argv[++i]);
            if (n <= 0) {
                fprintf(stderr, "Error: number of frames must be > 0, got '%s'.\n", argv[i]);
                exit(1);
            }
            return n;
        }
 
        // Bare positional integer: ./main_bin 24
        if (!arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit))
        {
            int n = std::atoi(arg.c_str());
            if (n <= 0) {
                fprintf(stderr, "Error: number of frames must be > 0, got '%s'.\n", arg.c_str());
                exit(1);
            }
            return n;
        }
 
        fprintf(stderr, "Warning: unrecognised argument '%s' -- ignored.\n", arg.c_str());
    }
    return defaultFrames;
}
 
 
// ─────────────────────────────────────────────────────────────────────────────
//  makeGif
//
//  Assembles all rendered PPM frames into an animated GIF using ffmpeg.
//  Uses ffmpeg's two-pass palettegen workflow for best colour quality:
//
// ─────────────────────────────────────────────────────────────────────────────
inline void makeGif(const std::string& frameDir,
                    const std::string& frameGlob,
                    const std::string& outGif,
                    int fps = 12)
{
    std::string inputPattern = frameDir + "/" + frameGlob;
    std::string palette      = frameDir + "/palette.png";
 
    // -- Pass 1: generate palette ---------------------------------------------
    std::string pass1 =
        "ffmpeg -y"
        " -framerate " + std::to_string(fps) + //playback speed in frames per second
        " -i \""        + inputPattern + "\""
        " -vf \"palettegen=stats_mode=full\""
        " \""           + palette + "\""
        " 2>/dev/null";
 
    fprintf(stderr, "GIF pass 1: building palette...\n");
    int r1 = std::system(pass1.c_str());
    if (r1 != 0) {
        fprintf(stderr, "Warning: ffmpeg palette pass failed (exit %d).\n"
                        "  Is ffmpeg installed and on PATH?\n"
                        "  PPM frames are still available in %s/\n",
                r1, frameDir.c_str());
        return;
    }
 
    // -- Pass 2: encode GIF using the palette ---------------------------------
    std::string pass2 =
        "ffmpeg -y"
        " -framerate " + std::to_string(fps) +
        " -i \""        + inputPattern + "\""
        " -i \""        + palette + "\""
        " -lavfi \"paletteuse=dither=bayer:bayer_scale=3\""
        " \""           + outGif + "\""
        " 2>/dev/null";
 
    fprintf(stderr, "GIF pass 2: encoding GIF...\n");
    int r2 = std::system(pass2.c_str());
    if (r2 != 0) {
        fprintf(stderr, "Warning: ffmpeg encoding pass failed (exit %d).\n", r2);
        return;
    }
 
    std::filesystem::remove(palette);
    fprintf(stderr, "GIF saved: %s\n", outGif.c_str());
}

size_t extractResolution(const std::string& dir) {
    auto pos = dir.rfind('_');
    if (pos == std::string::npos)
        throw std::runtime_error("Cannot infer resolution from dir name: " + dir);
    return std::stoul(dir.substr(pos + 1));
}

