//g++-15 -fopenmp -std=c++17 \
  -isysroot $(xcrun --show-sdk-path) \
  -I/opt/homebrew/Cellar/openvdb/13.0.0_1/include \
  main.cpp -o main
//./main

#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#pragma warning(disable : 4146) // Spegne il fastidioso errore C4146 di Visual Studio


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

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/Ray.h>


//debug
std::atomic<int> highNii{0};
std::atomic<int> totalNonZero{0};

struct Matrix
{
	const float operator [] (size_t i) const { return (m)[i]; } //m per leggere const
	float& operator [] (size_t i) { return m[i]; } // per scrivere

	float m[16];

};

struct vec3 
{
	float x{ 0 };
	float y{ 0 };
	float z{ 0 };

	float length() const 
	{
		return sqrt(x * x + y * y + z * z);
	}

	vec3& nor()
	{     
		float len = x * x + y * y + z * z;
		if (len != 0)
			len = sqrt(len);
		x /= len, y /= len, z /= len;
		return *this;
	}

	float operator * (const vec3& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}
	vec3 operator - (const vec3& v) const
	{
		return vec3{ x - v.x, y - v.y, z - v.z };
	}
	vec3 operator - () const
	{
		return vec3{ -x, -y, -z };
	}
	vec3 operator + (const vec3& v) const
	{
		return vec3{ x + v.x, y + v.y, z + v.z };
	}
	vec3& operator += (const vec3& v)
	{
		x += v.x, y += v.y, z += v.z;
		return *this;
	}
	vec3& operator *= (const float& r)
	{
		x *= r, y *= r, z *= r;
		return *this;
	}
	friend vec3 operator * (const float& r, const vec3& v)
	{
		return vec3{ v.x * r, v.y * r, v.z * r };
	}
	friend std::ostream& operator << (std::ostream& os, const vec3& v)
	{
		os << v.x << " " << v.y << " " << v.z;
		return os;
	}
	friend vec3 operator / (const float& r, const vec3& v)
	{ 
		return vec3{ r / v.x, r / v.y, r / v.z }; 
	}
	vec3 operator * (const float& r) const
	{
		return vec3{ x * r, y * r, z * r };
	}
	vec3 operator / (const vec3& v) const
	{ 
		return vec3{ x / v.x, y / v.y, z / v.z }; 
	}
	vec3 operator * (const Matrix& m) const
	{
		vec3 v;
		v.x = m[0]*x + m[4]*y + m[8]*z;
		v.y = m[1]*x + m[5]*y + m[9]*z;
		v.z = m[2]*x + m[6]*y + m[10]*z;
		return v;
	}
	vec3& operator *= (const Matrix& m)
	{ 
		*this = *this * m;
		return *this; 
	}
};

vec3 cross(const vec3& a, const vec3& b)
{
	return vec3{
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}


vec3 transformPoint(const Matrix& m, const vec3& p) // per trasformare punti
{
	vec3 result;
	result.x = m[0]*p.x + m[4]*p.y + m[8]*p.z  + m[12];
	result.y = m[1]*p.x + m[5]*p.y + m[9]*p.z  + m[13];
	result.z = m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14];
	return result;
}

constexpr vec3 background_color{ 0,0,0 };

struct Grid 
{
	size_t baseResolution = 512; // size_t è un tipo intero senza segno, perfetto per dimensioni e indici. 1
	std::unique_ptr<float[]> densityData; // è l'array che conterrà tutti i valori di densità. Non lo inizializziamo qui — verrà riempito quando carichiamo il file binario.
	vec3 bounds[2]{ vec3{-30,-30,-30,}, vec3{30,30,30} }; //i due punti definiscono il bounding box della griglia nello spazio 3D

	float operator() (const int& xi, const int& yi, const int& zi) const // Serve per leggere la densità di un voxel dato il suo indice
	{
		// il controllo serve perché quando faremo l'interpolazione potremmo chiedere voxel appena fuori dai bordi — in quel caso restituiamo 0 (nessuna densità).
		if (xi<0 || xi > baseResolution - 1 ||
			yi < 0 || yi > baseResolution - 1 ||
			zi < 0 || zi > baseResolution - 1) 
			return 0;

		return densityData[(zi * baseResolution + yi) * baseResolution + xi];	}
};

struct Ray 
{ 
	//  sintassi : orig(orig), dir(dir) si chiama lista di inizializzazione — è il modo corretto in C++ per inizializzare i campi nel costruttore. 
	// Equivale a dire "il campo orig prende il valore del parametro orig".
	Ray(const vec3& orig, const vec3& dir) : orig(orig), dir(dir)
	{
		invDir = 1 / dir;

		// salviamo il segno: 1 se negativo, 0 se positivo
		sign[0] = (invDir.x < 0);
		sign[1] = (invDir.y < 0);
		sign[2] = (invDir.z < 0);

	}

	// Questo operatore permette di scrivere ray(t) per ottenere il punto lungo il raggio alla distanza t. 
	// È una comodità — invece di scrivere ray.orig + ray.dir * t ogni volta, scrivi semplicemente ray(t).
	vec3 operator() (const float& t) const
	{
		return orig + dir * t;
	}

	vec3 orig;
	vec3 dir;
	vec3 invDir;  //Servirà nell'algoritmo ray-box. Invece di dividere per dir.x ad ogni calcolo (lento), pre-calcoliamo 1/dir.x una volta sola e la usiamo come moltiplicazione (veloce).
	bool sign[3]; //Un array di 3 booleani, uno per x, uno per y, uno per z. Ogni valore sarà 1 se invDir su quell'asse è negativo, 0 se positivo. Serve per sapere da che lato arriva il raggio rispetto al box.
};

// STELLE
struct Star
{
	vec3 dir; //vettore unitario che punta verso la stella nel sistema di coordinate celesti
	float brightness; // flusso lineare, calcolato dalla magnitudine Gaia (// scala: mag 0 → 1.0, mag 5 → 0.01, mag 10 → 0.0001)
    vec3 color; // prendo il colore dal colour index (differenxa di magnitudine tra le due bande fotometriche BP e RP)
};

// Le stelle nel catalogo Gaia hanno coordinate nel sistema ICRS mentre ay marcher invece ha il suo sistema mondo dove la camera guarda verso -Z, la nebulosa è all'origine,
// La soluzione è costruire una matrice di rotazione che trasforma ICRS → spazio mondo:
Matrix buildICRSToWorldMatrix()
{
	// conversione coordinate della Crab in radianti 
	float raCrab  = (float)(83.63 * M_PI / 180.0);
	float decCrab = (float)(22.01 * M_PI / 180.0);

	// direzione verso la Crab come vettore in sistema ICRS
	vec3 crabDir;
	crabDir.x = cos(decCrab) * cos(raCrab);
	crabDir.y = cos(decCrab) * sin(raCrab);
	crabDir.z = sin(decCrab);

	// Mettiamo la Crab verso -Z mondo, in modo che la camera la veda al frame 0
	vec3 camZ = { -crabDir.x, -crabDir.y, -crabDir.z };

	// polo nord celeste come vettore "up" di riferimento
	vec3 northPole = { 0.f, 0.f, 1.f };

	// asse X = northPole × camZ: il vettore "destra" nel piano del cielo
	// vec3 camX;
	// camX.x = northPole.y * camZ.z - northPole.z * camZ.y;
	// camX.y = northPole.z * camZ.x - northPole.x * camZ.z;
	// camX.z = northPole.x * camZ.y - northPole.y * camZ.x;
	vec3 camX = cross(northPole, camZ); 
    camX.nor();

	// asse Y = camZ × camX: il vettore "su" ortogonale per costruzione
	// vec3 camY;
	// camY.x = camZ.y * camX.z - camZ.z * camX.y;
	// camY.y = camZ.z * camX.x - camZ.x * camX.z;
	// camY.z = camZ.x * camX.y - camZ.y * camX.x;
    vec3 camY = cross(camZ, camX); 
	camY.nor();

	// righe della matrice = assi del sistema mondo espressi in ICRS
	Matrix m;
	m.m[0] = camX.x;  m.m[1] = camX.y;  m.m[2] = camX.z;
	m.m[3] = camY.x;  m.m[4] = camY.y;  m.m[5] = camY.z;
	m.m[6] = camZ.x;  m.m[7] = camZ.y;  m.m[8] = camZ.z;
	// (solo 9 elementi, matrice 3x3 di rotazione)
	return m;
}

vec3 applyRotation(const Matrix& m, const vec3& v)
{
	return vec3{
		m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
		m.m[3]*v.x + m.m[4]*v.y + m.m[5]*v.z,
		m.m[6]*v.x + m.m[7]*v.y + m.m[8]*v.z
	};
}

std::vector<Star> loadStars(const std::string& csvPath)
{
	std::vector<Star> stars;

	std::ifstream f(csvPath);
	if (!f.is_open())  // se il file non si apre avvisa ma non crashare, il render continuerà senza stelle
	{
		fprintf(stderr, "Attenzione: impossibile aprire %s\n", csvPath.c_str());
		return stars;
	}

	std::string line;

	std::getline(f, line);    // la prima riga è l'header "source_id,ra,dec,phot_g_mean_mag" — la saltiamo

	while (std::getline(f, line))
	{
		long long sid; // source_id lo leggiamo ma non do usiamo
		double ra, dec, mag, bprp;

		// sscanf parsea la riga csv nei quattro valori
		// %lld = long long (source_id), %lf = double (ra, dec, mag)
		if (sscanf(line.c_str(), "%lld,%lf,%lf,%lf, %lf", &sid, &ra, &dec, &mag, &bprp) != 5)
			continue; // riga malformata o vuota, salta

		// conversion gradi in radianti
		float raRad = (float)(ra * M_PI / 180);
		float decRad = (float)(dec * M_PI / 180);

		//conversione coordinate sferice -> cartesiane
		// Il risultato è un vettore unitario in ICRS per ciascuna stella
		vec3 dir;
		dir.x = cos(decRad) * cos(raRad);
		dir.y = cos(decRad) * sin(raRad);
		dir.z = sin(decRad);

		// conversinoe di amagnitudine a flusso lineare (formula di Pogson)
		float brightness = pow(10.f, (0.f - (float)mag) / 2.5f);

		// non consideriamo il contributo delle stelle troppo deboli (mag 16 → brightness ≈ 2.5e-7, già quasi zero)
		if (brightness < 1e-6f) continue;

		static Matrix icrsToWorld = buildICRSToWorldMatrix(); //Il static è importante: la matrice viene costruita una volta sola al primo caricamento e riusata per tutte le stelle.
		vec3 dirWorld = applyRotation(icrsToWorld, dir);// Usa l'operatore nativo column-major! // rotazione di stelle da ICRS a spazio mondo
		dirWorld.nor();
		// stars.push_back(Star{ dirWorld, brightness });	}
        // bp_rp - > temperatura (Ballesteros 2012) → RGB
		vec3 starColor{ 1.f, 1.f, 1.f }; // colore di default bianco se bp_rp mancante
		
		if (float(bprp) > -99.f) // gestisce casi in cui gaia non ha misurato bp_rp
		{
			// formula di ballesteros converte colour indez in temepratura 
			float T = 4600.f * (1.f/(0.92f*(float)bprp + 1.7f) + 1.f/(0.92f*(float)bprp + 0.62f));
			T = std::clamp(T, 1000.f, 40000.f);

			// uso formule di Helland per convertire temperatura a RGB
			float t = T / 100.f; // faccio divisione prima invece che fare in ogni formula

			float r, g, b;

			// red
			r = (T <= 6600.f) ? 1.f : std::clamp(329.698727446f * powf(t - 60.f, -0.1332047592f) / 255.f, 0.f, 1.f);

			// green
			if (T <= 6600.f)
				g = std::clamp((99.4708025861f * logf(t) - 161.1195681661f) / 255.f, 0.f, 1.f);
			else
				g = std::clamp(288.1221695283f * powf(t - 60.f, -0.0755148492f) / 255.f, 0.f, 1.f);

			// blue
			if (T >= 6600.f) {
				b = 1.f;  
			}
			else if (T <= 1900.f) {
				b = 0.f;  
			}
			else {
				b = std::clamp((138.5177312231f * logf(t - 10.f) - 305.0447927307f) / 255.f, 0.f, 1.f);
			}

			starColor = vec3{ r,g,b };

		}

		stars.push_back(Star{ dirWorld, brightness, starColor });	}

	fprintf(stderr, "Stelle caricate: %zu\n", stars.size());
	return stars;

};

vec3 starContribution(const Ray& ray, const std::vector<Star>& stars, float starBrightness) // starBrightness è il parametro globale che si può bilanciare a mano per la luminosità delle stelle
{
	vec3 result{ 0,0,0 };

	float sigma = 0.001f; // sigma controlla la dimensione angolare apparente delle stelle più è basso più le stelle sono puntiformi e precise
	float twoSigmaSq = 2.f * sigma * sigma;     // pre-calcoliamo 2*sigma^2 fuori dal loop perchè è costante per tutte le stelle; evita di ricalcolarlo migliaia di volte

	for (const auto& star : stars)
	{
		float cosTheta = ray.dir * star.dir; // prodotto scalare tra direzione di raggio e stella; se puntano nella stessa direzione -> 1; direzione opposta -> -1

		// early exit: se cosTheta < 0.99 l'angolo è maggiore di ~8° -> la gaussiana a quell'angolo è già praticamente zero
		// questo salto evita il calcolo di exp() per quasi tutte le stelle e ottimizza la funzione
		if (cosTheta < 0.9999f) continue;

		float theta2 = 2.f * (1.f - cosTheta); // invece che usare acos utilizzo questa approssimazione perchè più veloce ed è valida per piccoli angoli
		float profile = exp(-theta2 / twoSigmaSq); // profilo gaussiano: vale ~1 quando theta=0, decade rapidamente

		float contrib = star.brightness * starBrightness * profile;

		// // la stella è bianca — sommiamo lo stesso valore a R, G, B
		// result.x += contrib;
		// result.y += contrib;
		// result.z += contrib;
        // contributo stella con colore
		result.x += contrib * star.color.x;
		result.y += contrib * star.color.y;
		result.z += contrib * star.color.z;
	}

	return result;
};


//NOT EXPLAINED IN METHOD
bool raybox(const Ray& ray, const vec3 bounds[2], float& tmin, float& tmax)
{
	// Per sapere quale bound usare come entrata e quale come uscita, 
	// usiamo ray.sign che ci dice il segno della direzione del raggio.
	// Se il raggio va in direzione positiva (sign=0), entra da bounds[0] ed esce da bounds[1].
	// Se va in direzione negativa (sign=1), è il contrario.

	// calcolo prima le differenze
	float a, b, c, d, e, f;
	a = bounds[    ray.sign[0]].x - ray.orig.x;
	b = bounds[1 - ray.sign[0]].x - ray.orig.x;
	c = bounds[    ray.sign[1]].y - ray.orig.y;
	d = bounds[1 - ray.sign[1]].y - ray.orig.y;

	// Usiamo invDir invece di dividere per dir per evitare divisioni per zero
	// gestisce il caso speciale in cui il raggio è esattamente parallelo a una faccia del box (direzione zero):
	// se la differenza a è zero, il risultato è 0, altrimenti moltiplica. Evita il caso 0 * infinito che darebbe NaN.
	float x0 = a == 0 ? 0 : a * ray.invDir.x; // operatore ternario      condizione ? valore_se_vera : valore_se_falsa
	float x1 = b == 0 ? 0 : b * ray.invDir.x;
	float y0 = c == 0 ? 0 : c * ray.invDir.y;
	float y1 = d == 0 ? 0 : d * ray.invDir.y;

	if ((x0 > y1 || y0 > x1)) return false; // controllo di interesezione tra i primi due slab

	//  Calcoliamo tmin e tmax finali — prima per X e Y, poi aggiornandoli con Z:
	tmin = (y0 > x0) ? y0 : x0;
	tmax = (y1 < x1) ? y1 : x1;


	e = bounds[    ray.sign[2]].z - ray.orig.z;
	f = bounds[1 - ray.sign[2]].z - ray.orig.z;

	float z0 = e == 0 ? 0 : e * ray.invDir.z;
	float z1 = f == 0 ? 0 : f * ray.invDir.z;

	if ((tmin > z1) || (z0 > tmax)) return false;

	tmin = std::max(z0, tmin);
	tmax = std::min(z1, tmax);

	return true;
}


//NOT EXPLAINED IN METHOD
// la funzione lookup quindi mi fornisce il valore della densità. per farlo trasforma il punto in cui siamo in coordinate locali, poi voxel, 
// poi trovo le coordinate del voxel in cui mi trovo, quindi utilizzando gli 8 voxel vicini faccio l'interpolazione trilineare con i valori delle densitò ai voxel vicini
// 1.  Spazio mondo → (83.5, 12.3, -45.2) — il punto reale nella scena
// 2. Coordinate locali → (0.3, 0.7, 0.1) — dove si trova dentro la griglia, da 0 a 1
// 3. Coordinate voxel → (38.4, 89.6, 12.8) — quale cella della griglia 128x128x128
// 4. pLattice → (37.9, 89.1, 12.3) — shiftato di 0.5 per centrare sul voxel
// 5. xi, yi, zi → (37, 89, 12) — indici interi del voxel in cui siamo
// 6. Interpolazione trilineare → guarda gli 8 voxel agli angoli del cubo che ci circonda, li mescola in base alla distanza, restituisce un valore continuo e morbido

float lookup(const Grid& grid, const vec3& p)
{
	vec3 gridSize = (grid.bounds[1] - grid.bounds[0]);
	vec3 pLocal = (p - grid.bounds[0]) / gridSize; // trovo le coordinate locali rispetto alla griglia
	vec3 pVoxel = pLocal * grid.baseResolution; // trasformiamo in coordinate voxel (index tuple ma ancora float quindi non identificano un voxel preciso)

	// Ora dobbiamo trovare il voxel in cui siamo, cioè convertire pVoxel in indici interi.
	// il valore dei voxel è però memorizzato al centro, non al bordo. Quando converti il punto in voxel space con il floor, ottieni l'indice del voxel. 
	// Ma il floor ti porta all'angolo del voxel, non al suo centro. Sottraendo 0.5 sposti il sistema di riferimento in modo che floor ti dia il voxel il cui centro è più vicino al tuo punto.
	vec3 pLattice = vec3{ pVoxel.x - 0.5f, pVoxel.y - 0.5f, pVoxel.z - 0.5f };

	int xi = static_cast<int>(std::floor(pLattice.x)); // cast per andare da float a int
	int yi = static_cast<int>(std::floor(pLattice.y));
	int zi = static_cast<int>(std::floor(pLattice.z));

	//interpolazione trilineare
	float value = 0;
	// for annidati che vanno da 0 a 1, uno per ogni asse. Per ogni combinazione (i, j, k) 
	// calcola un peso proporzionale alla distanza dal voxel e lo moltiplica per la densità di quel voxel.
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				//pesi
				float wx = 1 - std::abs(pLattice.x - (xi + i));
				float wy = 1 - std::abs(pLattice.y - (yi + j));
				float wz = 1 - std::abs(pLattice.z - (zi + k));

				// chiama l'operatore () della Grid che avevamo scritto all'inizio — legge la densità del voxel (xi+i, yi+j, zi+k) e la accumula pesata.
				value += wx * wy * wz * grid(xi + i, yi + j, zi + k);
			}
		}
	}

	return value;
}

// Con NanoVDB, tutta questa matematica complessa è già ottimizzata al massimo dentro una classe chiamata Sampler (nanovdb::SampleFromVoxels).
// La funzione per campionare il volume con Nanovdb:
inline float sampleGrid(const nanovdb::FloatGrid* grid, const vec3& p, nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>& sampler)
{
	if (!grid) return 0.0f;

	nanovdb::Vec3f nvWorldPos(p.x, p.y, p.z); // converte il vec3 nel vettore nanovdb
	nanovdb::Vec3f indexPos = grid->worldToIndexF(nvWorldPos); // // NanoVDB calcola istantaneamente dove si trova il punto all'interno dell'albero VDB

	return sampler(indexPos); // Il sampler legge i dati e fa l'interpolazione trilineare in automatico!
}


//NOT USED

float phaseHG(const vec3& view_dir, const vec3& light_dir, const float& g)
{
	float cos_theta = view_dir * light_dir; // l coseno dell'angolo tra la direzione di vista e la direzione della luce
	float denom = 1 + g * g - 2 * g * cos_theta;

	return  (1 / (4*M_PI)) * (1 - g * g) / (denom * sqrt(denom));
}

vec3 nebulaColor(float nii_ha, float sii_ha, float sii_sii, float density, float vel)
{
	// Mappatura a falsi colori basata sui dati FITS disponibili
	// (Martin et al. 2021, SITELLE/ORB).
	//
	// I dati disponibili sono RAPPORTI di righe di emissione, non intensità
	// assolute. Tutti i valori sono normalizzati in [0,1] dopo log-stretch
	// (normalizzazione lineare per vel).
	//
	// Range reali misurati a runtime sui voxel campionati:
	//   nii_ha  = [0.0000, 0.5544]
	//   sii_ha  = [0.0000, 0.6435]
	//   vel     = [0.0003, 0.9878] — centro fisico a 0.5 (range simmetrico)
	//
	// NOTA sulla palette: la Hubble Palette standard (SHO) richiederebbe
	// tre canali: SII→rosso, Hα→verde, OIII→blu. Nei dati SITELLE non sono
	// presenti né Hα assoluto né OIII — solo i rapporti NII/Hα e SII/Hα.
	// Si adotta quindi una palette alternativa fisicamente motivabile:
	//
	//   Hα implicito (nii_ha e sii_ha bassi) → blu/viola
	//     — coerente con l'aspetto ottico reale della Crab a banda stretta,
	//       dove le zone a bassa emissione di NII appaiono dominate dal
	//       continuo e dall'emissione diffusa.
	//   NII dominante (nii_ha alto) → rosso
	//     — ejecta freddi e densi nella shell esterna (658 nm).
	//   SII forte (sii_ha alto)     → arancio
	//     — fronte di shock al bordo della nebulosa (671/673 nm).
	//   SII6716/6731 (sii_sii)      → diagnostico densità elettronica,
	//                                  modula saturazione (non ha colore proprio).
	//   vel                         → Doppler shift cromatico sottile.
	//
	// NOTA: il sincrotrone della PWN non è presente nei FITS SITELLE.
	// Andrà aggiunto come componente separata in futuro.
	//
	// FISICA dei rapporti (dal paper Martin et al. 2021):
	// - NII/Ha: basso = gas più ionizzato, alto = ejecta freddi e densi.
	// - SII/Ha: alto dove il fronte di espansione comprime il gas termico.
	// - SII6716/SII6731: rapporto alto (~1.4) = gas rarefatto (~100 cm^-3),
	//   rapporto basso (~0.4) = gas denso (~10000 cm^-3).
	//   Dopo normalizzazione: valore basso nel bin = gas denso
	//   (il minimo fisico 0.4 diventa 0 nel bin).

	vec3 ha_color  { 0.05f, 0.2f,  0.85f  }; // Hα implicito → blu/viola
	vec3 nii_color { 1.0f,  0.1f,  0.0f  }; // NII → rosso (658 nm)
	vec3 sii_color { 1.0f,  0.6f,  0.0f  }; // SII → arancio (671/673 nm)

	// --- Blend 1: NII/Ha → mix Hα implicito / NII ---
	// t_nii=0: NII assente, Hα domina → blu/viola
	// t_nii=1: NII domina → rosso
	// Fattore 3.5: porta il range reale [0, 0.55] a saturare correttamente
	// in [0,1] senza che il blu/viola schiacci i filamenti rossi.
	float t_nii = std::clamp(nii_ha * 5.0f, 0.f, 1.f);  
	vec3 baseColor{
		ha_color.x * (1.f - t_nii) + nii_color.x * t_nii,
		ha_color.y * (1.f - t_nii) + nii_color.y * t_nii,
		ha_color.z * (1.f - t_nii) + nii_color.z * t_nii
	};

	// --- Blend 2: SII/Ha → spinge verso arancio ---
	// alto SII/Ha = fronte di shock al bordo della nebulosa.
	// Fattore 3.0: porta il range reale [0, 0.64] a coprire bene [0,1].
	float t_sii = std::clamp(sii_ha * 4.5f, 0.f, 1.f);  
	baseColor.x = baseColor.x * (1.f - t_sii) + sii_color.x * t_sii;
	baseColor.y = baseColor.y * (1.f - t_sii) + sii_color.y * t_sii;
	baseColor.z = baseColor.z * (1.f - t_sii) + sii_color.z * t_sii;

	// --- Blend 3: SII6716/SII6731 → densità elettronica ---
	// Non è una riga spettrale con un colore proprio — è un rapporto
	// diagnostico che indica quanto è compresso il gas.
	// Lo usiamo per modulare la saturazione del colore base:
	//   gas denso   (sii_sii basso → t_dense alto) → desatura, tende al scuro
	//   gas rarefatto (sii_sii alto → t_dense basso) → mantiene saturazione
	// La soglia sii_sii > 0.05 esclude i voxel senza misura SII6716/6731
	// (sii_sii=0 significa assenza di dato, non gas denso).
	if (density > 0.f && sii_sii > 0.05f) {
		float t_dense = std::clamp((1.f - sii_sii) * 0.4f, 0.f, 1.f);
		baseColor.x *= (1.f - t_dense * 0.3f); // desatura R leggermente
		baseColor.y *= (1.f - t_dense * 0.5f); // desatura G più forte
		baseColor.z *= (1.f - t_dense * 0.2f); // desatura B poco
	}

	// --- Blend 4: velocità radiale → Doppler shift cromatico ---
	//
	// FISICA: la nebulosa si espande a ±1500 km/s. Le righe di emissione
	// delle zone che si allontanano da noi (redshift) appaiono spostate
	// verso il rosso; quelle che si avvicinano (blueshift) verso il blu.
	// Questo è il meccanismo usato da SITELLE/ORB per ricostruire la
	// struttura 3D: ogni filamento ha una coordinata Z ricavata dalla
	// sua velocità radiale misurata.
	//
	// Nel bin normalizzato (normalizzazione lineare su range fisico):
	//   0.0   = voxel vuoto (nessun dato)
	//   0.5   = gas fermo (vel ≈ 0 km/s) — range misurato [0.0003, 0.9878],
	//           quasi simmetrico attorno a 0.5.
	//   > 0.5 = redshift (gas che si allontana da noi)
	//   < 0.5 = blueshift (gas che si avvicina a noi)
	//
	// La soglia density > 0 && vel > 0 esclude i voxel vuoti dove vel=0.0
	// non ha significato fisico.
	// strength = 0.15 è volutamente sottile: il Doppler è un effetto
	// secondario rispetto alla composizione chimica (NII, SII).
	if (density > 0.f && vel > 0.f) {
		float doppler  = (vel - 0.5f) * 2.f; // in [-1, +1]
		float strength = 0.15f;
		baseColor.x += doppler * strength; // R: sale con redshift
		baseColor.z -= doppler * strength; // B: sale con blueshift
		baseColor.x = std::clamp(baseColor.x, 0.f, 1.f);
		baseColor.z = std::clamp(baseColor.z, 0.f, 1.f);
	}

	return baseColor;
}


// L è la radianza raccolta (luminanza del volume, cioè il colore finale)
// T è la trasmittanza finale (la transparency)
void integrate(const Ray& ray, const float& tMin, const float& tMax,
				vec3& L, float& T,
				const Grid& grid, const Grid& niiGrid, const Grid& siiGrid,
				const Grid& siiSiiGrid, const Grid& velGrid, const Grid& synch, 
				std::default_random_engine& rng,
				std::uniform_real_distribution<float>& dist) {
	float stepSize = 0.04;
	float sigma_a = 0.4; // assorbimento
	float sigma_s = 0.0; // scattering trascurabile
	float sigma_t = sigma_a + sigma_s;
	float g = 0;
	vec3 synchColor{0.15f, 0.55f, 0.85f};
	// uint8_t d = 2; // "probabilità" per la Russian Roulette

	size_t numSteps = std::ceil((tMax - tMin) / stepSize); // numero di passi
	float stride = (tMax - tMin) / numSteps; // ricalcolo lo step_size giusto per non strabordare

	// luce (togliendo l'in -scattering, la luce è solo quella emessa dal volume, non c'è illuminazione diretta da fonti esterne)
	// vec3 light_dir{ -0.315798, 0.719361, 0.618702 };
	// vec3 light_color{ 5, 5, 5 };

	float Tvol = 1; //transparency
	vec3 Lvol{ 0,0,0 }; //colore finale
	float shadowOpacity = 1; // parametro che controlla quanto le ombre sono intense.

	for (size_t n = 0; n < numSteps; n++) 
	{
		float t = tMin + stride * (n + dist(rng)); // jitter  dist(rng) estrae un float casuale in [0,1) ad ogni step.
		t = std::min(t, tMax); // clamp garantisce che t non superi mai tMax,
		vec3 samplePos = ray(t); // l'operatore che avevo creato

		float density = lookup(grid,    samplePos);
		float nii_ha = lookup(niiGrid, samplePos);
		float sii_ha = lookup(siiGrid, samplePos);
		float sii_sii = lookup(siiSiiGrid, samplePos);
		float vel = lookup(velGrid, samplePos);
		float syn = lookup(synch, samplePos);
		syn = pow(syn, 1.5); //compress high values to concentrate contribute in high density zones

		// debug una volta sola — conta quanti voxel hanno nii_ha > 0.3
		if (density > 0.01f) {
			totalNonZero++;
			if (nii_ha > 0.3f) highNii++;
		}
		
		float emissivity = 7;
		
		vec3 emColor = nebulaColor(nii_ha, sii_ha, sii_sii, density, vel);

		float Tsample = exp(-density * stride * sigma_t); // attenuazione del sample
		Tvol *= Tsample; // aggiorna trasparenza

		// Non introduce bias fisico rilevante per la Crab perché i filamenti non sono mai abbastanza densi da rendere il volume completamente opaco in modo uniforme
		// stai lavorando con densità massima ~0.72 e sigma_t = 0.42. Il contributo oltre quella soglia è inferiore a 0.01% della radianza finale. 
		// La Russian Roulette è overkill qui — è utile per materiali con forte scattering multiplo, non per emissione quasi pura.
		
		if (Tvol < 1e-4f) break; 

		//emColor × density × emissivity is the Le of the report
		Lvol += emColor * density * emissivity * Tvol * stride; // contributo di emissione
		if (syn > 0.02f) // ignora il rumore di fondo
    		Lvol += synchColor * syn * emissivity * 0.04 * Tvol * stride;
		/*
		// In-scattering
		float tlMin, tlMax; 
		Ray lightRay(samplePos, light_dir);
		// ha senso calcolare solo se c'è densità && il raggio di luce colpisce il box && il box è davanti a noi, non dietro
		if (density > 0 && raybox(lightRay, grid.bounds, tlMin, tlMax) && tlMax > 0 )
		{

			size_t numStepsLight = std::min((size_t)std::ceil(tlMax / stepSize), (size_t)16); // numero di passi, l'avevo modificato per la nebula crab ma in modo sbagliato
			float strideLight= tlMax / numStepsLight; // ricalcolo lo step_size giusto per non strabordare

			float tau = 0; // spessore ottico: somma di tutte le densità
			for (size_t n1 = 0; n1 < numStepsLight; ++n1)
			{
				float t1 = strideLight * (n1 + 0.5); // Usiamo $+0.5$ per campionare esattamente in mezzo al passo, invece di usare il rumore casuale (è più stabile per le ombre).
				vec3 samplePosScatt = lightRay(t1);
				tau += lookup(grid, samplePosScatt);
			}

			float light_attenuation = exp(-tau * strideLight * sigma_t * shadowOpacity);
			Lvol += light_color * light_attenuation * density * sigma_s * phaseHG(-ray.dir, light_dir, g) * Tvol * stride;

		}
		*/




		//if (Tvol < 1e-3)
		//{
		//	if (distribution(generator) > 1.0f / d)
		//	{
		//		break;
		//	}
		//	else 
		//	{
		//		Tvol *= d;
		//	}
		//}

	}

	L = Lvol;
	T = Tvol;
}


void integrate(const Ray& ray, const float& tMin, const float& tMax,
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
	float sigma_t  = 0.45f;
	float emissivity = 12.f;
	T = 1.0f;
	L = vec3{ 0.f, 0.f, 0.f };

	// Costruiamo il raggio NanoVDB in WORLD SPACE
	nanovdb::Vec3f nvOrig(ray.orig.x, ray.orig.y, ray.orig.z);
	nanovdb::Vec3f nvDir (ray.dir.x,  ray.dir.y,  ray.dir.z);
	nanovdb::math::Ray<float> nvRay(nvOrig, nvDir, tMin, tMax);

	// Convertiamo in INDEX SPACE — qui t è scalato diversamente
	nanovdb::math::Ray<float> idxRay = nvRay.worldToIndexF(*densityGrid);

	// Clipping sulla bbox della griglia in index space
	float t0 = idxRay.t0();
	float t1 = idxRay.t1();
	if (!idxRay.intersects(densityGrid->indexBBox(), t0, t1))
		return;

	// Sampler — uno per griglia, thread-local perché siamo dentro il parallel for
	using SamplerT = nanovdb::math::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
	SamplerT densitySampler(densityGrid->tree());
	SamplerT niiSampler    (niiGrid->tree());
	SamplerT siiSampler    (siiGrid->tree());
	SamplerT siiSiiSampler (siiSiiGrid->tree());
	SamplerT velSampler    (velGrid->tree());
	SamplerT synSampler(synGrid->tree());

	// stepSize in INDEX SPACE (1 voxel per step è un buon punto di partenza)
	float stepSize = 1.0f;
	size_t numSteps = std::max((size_t)1, (size_t)std::ceil((t1 - t0) / stepSize));
	float stride = (t1 - t0) / numSteps;

	for (size_t n = 0; n < numSteps; n++) {
		float t = t0 + stride * (n + dist(rng));
		t = std::min(t, t1);

		// Posizione in INDEX SPACE — la passiamo direttamente al sampler
		nanovdb::Vec3f idxPos = idxRay(t);

		float density = densitySampler(idxPos);

		if (density > 0.01f) {
			float nii_ha  = niiSampler(idxPos);
			float sii_ha  = siiSampler(idxPos);
			float sii_sii = siiSiiSampler(idxPos);
			float vel     = velSampler(idxPos);

			vec3 emColor = nebulaColor(nii_ha, sii_ha, sii_sii, density, vel);

			// stride in index space — lo convertiamo in world space per
			// avere unità fisiche coerenti con sigma_t
			float worldStride = stride * densityGrid->voxelSize()[0];

			float Tsample = exp(-density * worldStride * sigma_t);
			T *= Tsample;
			if (T < 1e-4f) return;

			L += emColor * density * emissivity * T * worldStride;

			// dentro if (density > 0.01f)
			static std::atomic<int> dbgCount{0};
			static float niiMin=1,niiMax=0,siiMin=1,siiMax=0,velMin=1,velMax=0;
			int cnt = dbgCount.fetch_add(1);
			if (nii_ha < niiMin) niiMin = nii_ha;
			if (nii_ha > niiMax) niiMax = nii_ha;
			if (sii_ha < siiMin) siiMin = sii_ha;
			if (sii_ha > siiMax) siiMax = sii_ha;
			if (vel < velMin) velMin = vel;
			if (vel > velMax) velMax = vel;
			if (cnt == 500000)
				fprintf(stderr, "nii=[%.4f,%.4f] sii=[%.4f,%.4f] vel=[%.4f,%.4f]\n",
					niiMin,niiMax,siiMin,siiMax,velMin,velMax);
		}
		float syn = synSampler(idxPos);
		syn = pow(syn, 1.5f); // comprimi i valori alti come nel vecchio codice
		if (syn > 0.02f) {
			vec3 synchColor{ 0.15f, 0.55f, 0.85f };
			float worldStride = stride * densityGrid->voxelSize()[0];
			L += synchColor * syn * 12.f * 0.04f * T * worldStride;
		}
	}
}



// facciamo un tone mapping con Reinhard sulla luminanza
// Il ray marcher accumula radianza fisica lungo il raggio: i valori di L non sono bounded a [0,1], specialmente nelle zone dense dove molti sample contribuiscono.
// Un semplice clamp taglia tutto sopra 1.0 portando a saturazione piatta (bianco), perdendo informazione strutturale.
// La versione sulla luminanza è fisicamente più corretta di applicarlo canale per canale: si calcola quanto è brillante il pixel complessivamente,e si scala tutti e tre i canali dello stesso fattore. 
// Così i rapporti di colore tra R, G e B rimangono invariati — un pixel arancio rimane arancio, diventa solo meno intenso. Il Reinhard per canale invece avvicina tutti i canali a valori simili, desaturando l'immagine.
vec3 reinhard(vec3 c, float exposure = 1.4f)
{
	c.x *= exposure; c.y *= exposure; c.z *= exposure; // per avere colori più saturi
	float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; // (coefficienti 0.2126/0.7152/0.0722, che sono i pesi percettivi CIE dello spazio sRGB)
	float scale = 1.0f / (1.0f + lum); // // Il Reinhard comprime i valori alti con la curva x/(1+x), che è asintoticamente limitata a 1 ma non taglia mai bruscamente.
	return vec3{ c.x * scale, c.y * scale, c.z * scale };
}

// matrice della camera come variabile globale
// Matrix cameraToWorld{ 1,0,0,0, 0,1,0,0, 0,0,-1,0, 0,0,80,1 };

Matrix buildOrbitCamera(float orbitAngle, float distance, vec3 target) 
{
	// posizione della camera sull'orbita (piano XZ del world space(
	vec3 camPos{ distance * sin(orbitAngle), 0.f, distance * cos(orbitAngle) };

	// La nebulosa sta all'origine (0, 0, 0). La camera sta in camPos. Quindi:forward = destinazione - origine -> forward = (0,0,0) - camPos
	//  Per usarlo come direzione serve normalizzarlo 
	vec3 forward{ -camPos.x / distance, -camPos.y / distance, -camPos.z / distance }; //asse 1


	// vec3 camPos{ 
    //     target.x + distance * sin(orbitAngle), 
    //     target.y, 
    //     target.z + distance * cos(orbitAngle) 
    // };

    // vec3 forward{ 
    //     (target.x - camPos.x) / distance, 
    //     (target.y - camPos.y) / distance, 
    //     (target.z - camPos.z) / distance 
    // };
	// asse y del world space usato come riferimento per asse "su". poi dopo calcoleremo quella effettiva
	vec3 worldUp{ 0.f, 1.f, 0.f }; 

	//  asse come prodotto vettoriale del riferimento e forward
	vec3 right = cross(forward, worldUp); // asse 2
	right.nor();

	// calcolo effettivo asse up
	vec3 up = cross(right, forward);
	up.nor();

	vec3 camZ{ -forward.x, -forward.y, -forward.z };// -forward perché il ray marcher spara raggi verso -Z locale

	// Costruzione matrice column-major
	Matrix m;
	m.m[0]=right.x;  m.m[4]=up.x;  m.m[8] =camZ.x;  m.m[12]=camPos.x;
	m.m[1]=right.y;  m.m[5]=up.y;  m.m[9] =camZ.y;  m.m[13]=camPos.y;
	m.m[2]=right.z;  m.m[6]=up.z;  m.m[10]=camZ.z;  m.m[14]=camPos.z;
	m.m[3]=0.f;      m.m[7]=0.f;   m .m[11]=0.f;     m.m[15]=1.f;

	return m;
}

auto loadBinary = [](const std::string& path, std::unique_ptr<float[]>& data, size_t N)
{
    std::ifstream ifs(path, std::ios::binary);

    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    ifs.read(reinterpret_cast<char*>(data.get()), sizeof(float) * N);

    if (!ifs) {
        throw std::runtime_error("Error reading file (incomplete or corrupted): " + path);
    }
};

void render()
{
	fprintf(stderr, "Rendering frame"); 

	
	fprintf(stderr, "caricament file nvdb \n");
	// dichiariamo l'handle e i puntatori fuori dal blocco try-catch per poterli usare nel resto della funzione
	// Dichiariamo 5 Handle separati, uno per ogni griglia!
	nanovdb::GridHandle densityHandle;
	nanovdb::GridHandle niiHandle;
	nanovdb::GridHandle siiHandle;
	nanovdb::GridHandle siiSiiHandle;
	nanovdb::GridHandle velHandle;
	nanovdb::GridHandle synHandle;

	const nanovdb::FloatGrid* densityGrid = nullptr;
	const nanovdb::FloatGrid* niiGrid = nullptr;
	const nanovdb::FloatGrid* siiGrid = nullptr;
	const nanovdb::FloatGrid* siiSiiGrid = nullptr;
	const nanovdb::FloatGrid* velGrid = nullptr;
	const nanovdb::FloatGrid* synGrid = nullptr;

	try{
		
		// Chiediamo a NanoVDB di caricare ogni griglia specificando il suo NOME
		densityHandle = nanovdb::io::readGrid("density.nvdb");
		niiHandle     = nanovdb::io::readGrid("nii_ha.nvdb");
		siiHandle     = nanovdb::io::readGrid("sii_ha.nvdb");
		siiSiiHandle  = nanovdb::io::readGrid("sii_sii.nvdb");
		velHandle     = nanovdb::io::readGrid("vel.nvdb");
		synHandle     = nanovdb::io::readGrid("syn.nvdb");

		// Ora ogni Handle contiene SOLO la sua griglia, che sarà quindi all'indice 0!
		densityGrid = densityHandle.grid<float>(0);
		niiGrid     = niiHandle.grid<float>(0);
		siiGrid     = siiHandle.grid<float>(0);
		siiSiiGrid  = siiSiiHandle.grid<float>(0);
		velGrid     = velHandle.grid<float>(0);
		synGrid     = synHandle.grid<float>(0);
        auto synBbox = synGrid->worldBBox();
            fprintf(stderr, "synchrotron worldBBox: (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f)\n",
                synBbox.min()[0], synBbox.min()[1], synBbox.min()[2],
                synBbox.max()[0], synBbox.max()[1], synBbox.max()[2]);

		if (!densityGrid || !niiGrid || !siiGrid || !siiSiiGrid || !velGrid || !synGrid) 
		{		
			fprintf(stderr, "Errore, una o più griglie non trovate nel ndvb \n");
			return;
		}
		fprintf(stderr, "File .nvdb caricato con successo e griglie collegate!\n");

	} catch (const std::exception& e) {

		fprintf(stderr, "Eccezione durante il caricamento del VDB: %s\n", e.what());
		return;
	}

	// controllo per voxel cubico
	auto vs = densityGrid->voxelSize();
	fprintf(stderr, "voxelSize: %.4f %.4f %.4f\n", vs[0], vs[1], vs[2]);

	// controllo posizione della box
	auto bbox = densityGrid->indexBBox();
	fprintf(stderr, "indexBBox: (%d,%d,%d) -> (%d,%d,%d)\n",
		bbox.min()[0], bbox.min()[1], bbox.min()[2],
		bbox.max()[0], bbox.max()[1], bbox.max()[2]);

	// controllo per capire dove il centro della griglia in worldspace
	auto wbbox = densityGrid->worldBBox();
	fprintf(stderr, "worldBBox: (%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f)\n",
		wbbox.min()[0], wbbox.min()[1], wbbox.min()[2],
		wbbox.max()[0], wbbox.max()[1], wbbox.max()[2]);
	float cx = (wbbox.min()[0] + wbbox.max()[0]) * 0.5f;
	float cy = (wbbox.min()[1] + wbbox.max()[1]) * 0.5f;
	float cz = (wbbox.min()[2] + wbbox.max()[2]) * 0.5f;
	fprintf(stderr, "centro world: (%.2f, %.2f, %.2f)\n", cx, cy, cz);

	// carica il catalogo stelle Gaia
	std::vector<Star> stars = loadStars("gaia_stars.csv");
	//debug
	for (int i = 0; i < std::min(5, (int)stars.size()); ++i)
		fprintf(stderr, "stella %d: dir=(%+.3f, %+.3f, %+.3f)\n",
			i, stars[i].dir.x, stars[i].dir.y, stars[i].dir.z);

	// DEBUG — stampa alcuni valori delle griglie
	//float minD = 1e9, maxD = 0, minT = 1e9, maxT = 0;
	//for (int i = 0; i < 512*512*512; i++) {
	//	float d = grid.densityData[i];
	//	float t = NiiGrid.densityData[i];
	//	if (d > maxD) maxD = d;
	//	if (d > 0 && d < minD) minD = d;
	//	if (t > maxT) maxT = t;
	//	if (t > 0 && t < minT) minT = t;
	//}

	//fprintf(stderr, "Density  — min=%.4f  max=%.4f\n", minD, maxD);
	//fprintf(stderr, "Temperat — min=%.1f  max=%.1f\n", minT, maxT);

	size_t width = 800;
	size_t height = 800;

	// usiamo unnisgned invece che signed poichè char è signed (-128/+127).
	// Qualsiasi valore di pixel sopra 127 veniva interpretato come negativo, corrompendo il file PPM con colori sbagliati. 
	// unsigned char copre correttamente 0-255, che è il range standard per immagini a 8 bit per canale.
	std::unique_ptr<unsigned char[]> imgbuf = std::make_unique<unsigned char[]>(width * height * 3);

	// vec3 rayOrig = transformPoint(cameraToWorld, vec3{ 0,0,0 }); // Calcolo origine del raggio; non più usato qui per animazione

	auto frameAspectRatio = width / float(height);
	float fov = 45;
	float focal = tan(M_PI / 180 * fov * 0.5);

	// JITTER
	// Crea un generatore per ogni thread disponibile.
	// omp_get_max_threads() restituisce quanti thread OpenMP userà il sistema.
	std::vector<std::default_random_engine> rngs(omp_get_max_threads());
	// std::random_device legge entropia hardware (diversa ad ogni run del programma).
	// Senza questo, ogni esecuzione produrrebbe lo stesso jitter — non vogliamo.
	std::random_device rd;
	// Inizializza ogni generatore con un seed unico.
	// rd() è casuale, +t separa i thread anche se rd() desse valori simili.
	for (int t = 0; t < omp_get_max_threads(); ++t)
		rngs[t].seed(rd() + t);
	// La distribuzione è stateless (non ha memoria interna) — può essere condivisa.
	// Produce float uniformi in [0.0, 1.0).
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	
	fprintf(stderr, "Thread disponibili: %d\n", omp_get_max_threads());

	// Inizio loop animazione

	// per test facciamo solo d1 frame
	float testAngles[] = {0.f}; 
	int numFrames = 1;

	//int numFrames = 120;
	for (int frame = 0; frame < numFrames; ++frame) {

		fprintf(stderr, "\n=== Rendering frame %d / %d ===\n", frame + 1, numFrames);

		//float angle = 2.f * M_PI * frame / numFrames; //calcolo angolo per ciascun frame //MORE FRAMES
		float angle = testAngles[76]; // Prende l'angolo dall'array per il test
		vec3 gridCenter{ cx, cy, cz };
		Matrix cameraToWorld = buildOrbitCamera(angle, 120.f, gridCenter); // costruisco matrice di rotazione della telecamera per ciascun frame
		vec3 rayOrig = transformPoint(cameraToWorld, vec3{ 0,0,0 }); // il ray ovviamente dipende da dov è la camera

		std::atomic<int> rowsDone{0};

		#pragma omp parallel for schedule(dynamic, 4)
		for (int j = 0; j < (int)height; ++j) {

			// Ogni thread recupera il SUO generatore tramite il suo indice.
			// & è fondamentale: è un riferimento, non una copia.
			// Senza & il generatore verrebbe copiato e lo stato non avanzerebbe tra pixel.
			auto& rng = rngs[omp_get_thread_num()];

			for (unsigned int i = 0; i < width; ++i) {
				vec3 rayDir;
				rayDir.x = (2.f * (i + 0.5f) / width - 1) * focal;
				rayDir.y = (1 - 2.f * (j + 0.5f) / height) * focal * 1 / frameAspectRatio;
				rayDir.z = -1.f;
				rayDir *= cameraToWorld; //transformo la direzione con matrice
				rayDir.nor();

				Ray ray(rayOrig, rayDir);

				vec3 L;
				float transmittance = 1;

				//float tmin, tmax;
				//if (raybox(ray, grid.bounds, tmin, tmax)) {
				//	integrate(ray, tmin, tmax, L, transmittance, grid, NiiGrid, SiiGrid, SiiSiiGrid, VelGrid, rng, dist);
				//}

				// tMin = 0.0f, tMax = 1000.0f (distanza massima sicura)
				integrate(ray, 0.0f, 1000.0f, L, transmittance, densityGrid, niiGrid, siiGrid, siiSiiGrid, velGrid, synGrid, rng, dist);

				float starBrightness = 40.f; // parametro di bilanciamento luminosità stelle


				vec3 starL = starContribution(ray, stars, starBrightness);

				//  alla luce già accumulata dalla nebulosa, aggiungo anche la luce delle stelle filtrata dalla trasparenza. L diventa la somma di tutto — nebulosa più stelle.
				L += starL * transmittance;

				vec3 pixelColor = background_color * transmittance + L;

				size_t pixelOffset = (j * width + i) * 3;

				vec3 mapped = reinhard(pixelColor);
				imgbuf[pixelOffset + 0] = static_cast<unsigned char>(std::clamp(mapped.x, 0.f, 1.f) * 255);
				imgbuf[pixelOffset + 1] = static_cast<unsigned char>(std::clamp(mapped.y, 0.f, 1.f) * 255);
				imgbuf[pixelOffset + 2] = static_cast<unsigned char>(std::clamp(mapped.z, 0.f, 1.f) * 255);
			}

			#pragma omp critical
			{
				++rowsDone;
				fprintf(stderr, "\r%.1f%%", 100.0f * rowsDone / height);
			}
		}

		// Salvataggio con nome sequenziale per poterli unire in video
		char filename[64];
        std::filesystem::create_directories("nebula");

		snprintf(filename, sizeof(filename), "nenebula_%04d.ppm", frame);
		std::ofstream ofs(filename, std::ios::binary);
		ofs << "P6\n" << width << " " << height << "\n255\n";
		ofs.write(reinterpret_cast<const char*>(imgbuf.get()), width * height * 3);
		ofs.close();

	}

	fprintf(stderr, "Render di tutti i frame completato.\n");
}


int main() 
{
	// TEST ζ Tauri — verifica orientamento stelle
	float ra  = 84.41f * M_PI / 180.f;
	float dec = 21.14f * M_PI / 180.f;
	vec3 zetaTau{ cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec) };

	Matrix icrsToWorld = buildICRSToWorldMatrix();

	// USA APPLYROTATION, NON L'OPERATORE *
	vec3 v = applyRotation(icrsToWorld, zetaTau); 

	v.nor();
	fprintf(stderr, "zetaTau world: (%+.3f, %+.3f, %+.3f)\n", v.x, v.y, v.z);
	// Atteso: Z circa -1.00, X leggermente positivo, Y leggermente negativo

	render();
	return 0;
}