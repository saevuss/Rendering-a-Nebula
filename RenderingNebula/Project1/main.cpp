#define _USE_MATH_DEFINES
#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <vector>
#include <random>
#include <limits>
#include <cassert>

#include <string>
#include <sstream>

#include <atomic> 
#include <omp.h>

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

		// inverti yi: invece di leggere yi leggi (baseResolution-1 - yi)
		// I dati SITELLE/FITS hanno l'asse Y con origine in basso 
		int yi_flipped = (int)baseResolution - 1 - yi;
		int xi_flipped = (int)baseResolution - 1 - xi;
		return densityData[(zi * baseResolution + yi_flipped) * baseResolution + xi_flipped];
	}
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
};

Matrix buildICRSToWorldMatrix()
{
	float raCrab  = (float)(83.63 * M_PI / 180.0);
	float decCrab = (float)(22.01 * M_PI / 180.0);

	// direzione verso la Crab in ICRS
	vec3 crabDir;
	crabDir.x = cos(decCrab) * cos(raCrab);
	crabDir.y = cos(decCrab) * sin(raCrab);
	crabDir.z = sin(decCrab);

	// la camera guarda verso +Z mondo, quindi crabDir ICRS → +Z mondo
	vec3 camZ = { crabDir.x, crabDir.y, crabDir.z };

	// polo nord celeste come vettore "up" di riferimento
	vec3 northPole = { 0.f, 0.f, 1.f };

	// asse X = northPole × camZ: il vettore "destra" nel piano del cielo
	vec3 camX;
	camX.x = northPole.y * camZ.z - northPole.z * camZ.y;
	camX.y = northPole.z * camZ.x - northPole.x * camZ.z;
	camX.z = northPole.x * camZ.y - northPole.y * camZ.x;
	camX.nor();

	// asse Y = camZ × camX: il vettore "su" ortogonale per costruzione
	vec3 camY;
	camY.x = camZ.y * camX.z - camZ.z * camX.y;
	camY.y = camZ.z * camX.x - camZ.x * camX.z;
	camY.z = camZ.x * camX.y - camZ.y * camX.x;
	camY.nor();

	// righe della matrice = assi del sistema mondo espressi in ICRS
	Matrix m;
	m.m[0] = camX.x;  m.m[1] = camX.y;  m.m[2] = camX.z;
	m.m[3] = camY.x;  m.m[4] = camY.y;  m.m[5] = camY.z;
	m.m[6] = camZ.x;  m.m[7] = camZ.y;  m.m[8] = camZ.z;

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
	if (!f.is_open())  //   // se il file non si apre avvisa ma non crashare, il render continuerà senza stelle
	{
		fprintf(stderr, "Attenzione: impossibile aprire %s\n", csvPath.c_str());
		return stars;
	}

	std::string line;

	std::getline(f, line);    // la prima riga è l'header "source_id,ra,dec,phot_g_mean_mag" — la saltiamo

	while (std::getline(f, line))
	{
		long long sid; // source_id lo leggiamo ma non do usiamo
		double ra, dec, mag;

		// sscanf parsea la riga csv nei quattro valori
		// %lld = long long (source_id), %lf = double (ra, dec, mag)
		if (sscanf(line.c_str(), "%lld,%lf,%lf,%lf", &sid, &ra, &dec, &mag) != 4)
			continue; // riga malformata o vuota, salta

		// conversion gradi in radianti
		float raRad = (float)(ra * M_PI / 180);
		float decRad = (float)(dec * M_PI / 180);

		//conversione coordinate sferice -> cartesiane
		vec3 dir;
		dir.x = cos(decRad) * cos(raRad);
		dir.y = cos(decRad) * sin(raRad);
		dir.z = sin(decRad);

		// conversinoe di amagnitudine a flusso lineare (formula di Pogson)
		float brightness = pow(10.f, (0.f - (float)mag) / 2.5f);

		// non consideriamo il contributo delle stelle troppo deboli (mag 16 → brightness ≈ 2.5e-7, già quasi zero)
		if (brightness < 1e-6f) continue;

		static Matrix icrsToWorld = buildICRSToWorldMatrix();
		vec3 dirWorld = applyRotation(icrsToWorld, dir);
		dirWorld.nor();
		stars.push_back(Star{ dirWorld, brightness });	}

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

		// la stella è bianca — sommiamo lo stesso valore a R, G, B
		result.x += contrib;
		result.y += contrib;
		result.z += contrib;
	}

	return result;
};


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



float phaseHG(const vec3& view_dir, const vec3& light_dir, const float& g)
{
	float cos_theta = view_dir * light_dir; // l coseno dell'angolo tra la direzione di vista e la direzione della luce
	float denom = 1 + g * g - 2 * g * cos_theta;

	return  (1 / (4*M_PI)) * (1 - g * g) / (denom * sqrt(denom));
}

vec3 nebulaColor(float nii_ha, float sii_ha, float sii_sii, float density, float vel)
{
	// Tutti i valori sono in [0,1] dopo log-stretch.
	//
	// FISICA (dal paper Martin et al. 2021, Crab Nebula con SITELLE):
	// - La ionizzazione è da shock del pulsar wind, NON da fotoionizzazione UV.
	// - NII/Ha traccia lo stato di ionizzazione: basso = gas più ionizzato
	//   (direttamente esposto al vento della pulsar), alto = ejecta più freddi
	//   e densi nelle shell esterne.
	// - SII/Ha traccia le regioni di shock: alto dove il fronte di espansione
	//   comprime il gas termico contro la nebulosa di sincrotrone.
	// - SII6716/SII6731 (sii_sii) è diagnostico della densità elettronica:
	//   rapporto alto (~1.4) = gas rarefatto (~100 cm^-3),
	//   rapporto basso (~0.4) = gas denso (~10000 cm^-3).
	//   Nel bin normalizzato: valore alto = denso, valore basso = rarefatto.
	//   (la normalizzazione inverte: il minimo fisico 0.4 diventa 0 nel bin)

	// Colori ancorati alla fisica della Crab:
	// gas ionizzato dal pulsar wind → tende al blu-violaceo (come nelle
	// immagini ottiche HST della Crab dove la PWN è blu per sincrotrone)
	vec3 pwn    { 0.3f,  0.5f,  1.0f  }; // blu — gas ionizzato dal pulsar wind (basso NII/Ha)

	// ejecta termici nella shell → rosso-arancio (dominati da NII, SII)
	vec3 ejecta { 0.9f,  0.15f, 0.02f };  // rosso — ejecta densi, basso grado ionizzazione

	// zona di shock al fronte di espansione → arancio caldo (alto SII/Ha)
	// fisicamente: interfaccia tra nebulosa sincrotrone e gas termico
	// è dove Rayleigh-Taylor instabilities creano i filamenti (Hester 1996, citato nel paper)
	vec3 shock  { 1.0f,  0.5f,  0.0f  }; // arancio — fronte di shock

	// gas denso (basso sii_sii) → più opaco, tende al rosso scuro
	// gas rarefatto (alto sii_sii) → più trasparente, tende al blu
	// NON usiamo "bianco caldo" perché non c'è una stella centrale calda —
	// il centro della Crab è dominato dalla PWN non da emissione termica densa.
	vec3 dense  { 0.6f,  0.08f, 0.02f }; // rosso scuro — filamenti ad alta densità elettronica

	// --- Blend 1: NII/Ha → ionizzazione ---
	// basso NII/Ha (vicino al pulsar wind) = pwn (blu)
	// alto NII/Ha (ejecta esterni) = ejecta (rosso)
	// dopo — stile HST, density come peso aggiuntivo:
	float t_nii = std::clamp(nii_ha * 1.5f + density * 0.8f, 0.f, 1.f);
	vec3 baseColor{
		pwn.x * (1.f - t_nii) + ejecta.x * t_nii,
		pwn.y * (1.f - t_nii) + ejecta.y * t_nii,
		pwn.z * (1.f - t_nii) + ejecta.z * t_nii
	};

	// --- Blend 2: SII/Ha → shock ---
	// alto SII/Ha = zona di shock al fronte di espansione
	float t_sii = std::clamp(sii_ha * 1.5f, 0.f, 1.f);
	baseColor.x = baseColor.x * (1.f - t_sii) + shock.x * t_sii;
	baseColor.y = baseColor.y * (1.f - t_sii) + shock.y * t_sii;
	baseColor.z = baseColor.z * (1.f - t_sii) + shock.z * t_sii;

	// --- Blend 3: SII6716/SII6731 → densità elettronica ---
	// sii_sii basso nel bin = rapporto fisico basso = alta densità elettronica
	// usiamo (1 - sii_sii) per avere t_dense alto dove il gas è denso
	float t_dense = 0.f;
	if (density > 0.f && sii_sii > 0.05f)  // soglia: esclude voxel senza dati reali
		t_dense = std::clamp((1.f - sii_sii) * 1.5f, 0.f, 1.f);
	baseColor.x = baseColor.x * (1.f - t_dense) + dense.x * t_dense;
	baseColor.y = baseColor.y * (1.f - t_dense) + dense.y * t_dense;
	baseColor.z = baseColor.z * (1.f - t_dense) + dense.z * t_dense;

	// --- Blend 4: velocità radiale → Doppler shift del colore ---
	//
	// FISICA: la nebula si espande a ±1500 km/s. Le righe di emissione
	// delle zone che si allontanano da noi (redshift) sono spostate
	// verso il rosso; quelle che si avvicinano (blueshift) verso il blu.
	// Questo è esattamente il meccanismo usato nel paper per ricostruire
	// la struttura 3D: ogni filamento ha una Z ricavata dalla sua velocità.
	//
	// Nel bin: 0.0 = voxel vuoto (nessun dato)
	//          ~0.46 = gas fermo (vel ≈ 0 km/s)
	//          > 0.46 = redshift (si allontana)
	//          < 0.46 = blueshift (si avvicina)
	//
	// Usiamo 0.46 come zero fisico (valore misurato dalla voxelizzazione).
	// La soglia density > 0 esclude i voxel vuoti dove vel=0.0 non ha
	// significato fisico — senza di essa tutto il vuoto sembrerebbe gas fermo.

	if (density > 0.f && vel > 0.f) {
		// doppler in [-1, +1]: negativo = blueshift, positivo = redshift
		float doppler = (vel - 0.477f) * 2.f;
		float strength = 0.25f; // quanto il Doppler modifica il colore (0=niente, 1=totale)

		// redshift: spinge verso il rosso, toglie blu
		// blueshift: spinge verso il blu, toglie rosso
		baseColor.x += doppler * strength;        // R: sale con redshift
		baseColor.z -= doppler * strength;        // B: sale con blueshift
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
				const Grid& siiSiiGrid, const Grid& velGrid,
				std::default_random_engine& rng,
				std::uniform_real_distribution<float>& dist) {
	float stepSize = 0.04;
	float sigma_a = 0.4; // assorbimento
	float sigma_s = 0.0; // scattering trascurabile
	float sigma_t = sigma_a + sigma_s;
	float g = 0;
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

		Lvol += emColor * density * emissivity * Tvol * stride; // contributo di emissione

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

// facciamo un tone mapping con Reinhard sulla luminanza
// Il ray marcher accumula radianza fisica lungo il raggio: i valori di L non sono bounded a [0,1], specialmente nelle zone dense dove molti sample contribuiscono.
// Un semplice clamp taglia tutto sopra 1.0 portando a saturazione piatta (bianco), perdendo informazione strutturale.
// La versione sulla luminanza è fisicamente più corretta di applicarlo canale per canale: si calcola quanto è brillante il pixel complessivamente,e si scala tutti e tre i canali dello stesso fattore. 
// Così i rapporti di colore tra R, G e B rimangono invariati — un pixel arancio rimane arancio, diventa solo meno intenso. Il Reinhard per canale invece avvicina tutti i canali a valori simili, desaturando l'immagine.
vec3 reinhard(vec3 c, float exposure = 0.9f)
{
	c.x *= exposure; c.y *= exposure; c.z *= exposure; // per avere colori più saturi
	float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; // (coefficienti 0.2126/0.7152/0.0722, che sono i pesi percettivi CIE dello spazio sRGB)
	float scale = 1.0f / (1.0f + lum); // // Il Reinhard comprime i valori alti con la curva x/(1+x), che è asintoticamente limitata a 1 ma non taglia mai bruscamente.
	return vec3{ c.x * scale, c.y * scale, c.z * scale };
}

// matrice della camera come variabile globale
Matrix cameraToWorld{ 1,0,0,0, 0,1,0,0, 0,0,-1,0, 0,0,80,1 };
void render()
{
	fprintf(stderr, "Rendering frame"); 

	// Carica density.bin
	Grid grid;
	grid.densityData = std::make_unique<float[]>(grid.baseResolution * grid.baseResolution * grid.baseResolution);
	{
		std::ifstream ifs("density512.bin", std::ios::binary);
		ifs.read((char*)grid.densityData.get(), sizeof(float) * 512 * 512 * 512);
	}

	Grid NiiGrid;
	NiiGrid.densityData = std::make_unique<float[]>(NiiGrid.baseResolution * NiiGrid.baseResolution * NiiGrid.baseResolution);
	{
		std::ifstream ifs("nii_ha512.bin", std::ios::binary);
		ifs.read((char*)NiiGrid.densityData.get(), sizeof(float) * 512 * 512 * 512); 
	}

	Grid SiiGrid;
	SiiGrid.densityData = std::make_unique<float[]>(SiiGrid.baseResolution * SiiGrid.baseResolution * SiiGrid.baseResolution);
	{
		std::ifstream ifs("sii_ha512.bin", std::ios::binary);
		ifs.read((char*)SiiGrid.densityData.get(), sizeof(float) * 512 * 512 * 512); 
	}

	Grid SiiSiiGrid;
	SiiSiiGrid.densityData = std::make_unique<float[]>(SiiSiiGrid.baseResolution * SiiSiiGrid.baseResolution * SiiSiiGrid.baseResolution);
	{
		std::ifstream ifs("sii_sii512.bin", std::ios::binary);
		ifs.read((char*)SiiSiiGrid.densityData.get(), sizeof(float) * 512 * 512 * 512); 
	}

	Grid VelGrid;
	VelGrid.densityData = std::make_unique<float[]>(VelGrid.baseResolution * VelGrid.baseResolution * VelGrid.baseResolution);
	{
		std::ifstream ifs("vel512.bin", std::ios::binary);
		ifs.read((char*)VelGrid.densityData.get(), sizeof(float) * 512 * 512 * 512);
	}

	// carica il catalogo stelle Gaia
	std::vector<Star> stars = loadStars("gaia_stars.csv");
	//debug
	for (int i = 0; i < std::min(5, (int)stars.size()); ++i)
		fprintf(stderr, "stella %d: dir=(%+.3f, %+.3f, %+.3f)\n",
			i, stars[i].dir.x, stars[i].dir.y, stars[i].dir.z);

	// DEBUG — stampa alcuni valori delle griglie
	float minD = 1e9, maxD = 0, minT = 1e9, maxT = 0;
	for (int i = 0; i < 512*512*512; i++) {
		float d = grid.densityData[i];
		float t = NiiGrid.densityData[i];
		if (d > maxD) maxD = d;
		if (d > 0 && d < minD) minD = d;
		if (t > maxT) maxT = t;
		if (t > 0 && t < minT) minT = t;
	}

	fprintf(stderr, "Density  — min=%.4f  max=%.4f\n", minD, maxD);
	fprintf(stderr, "Temperat — min=%.1f  max=%.1f\n", minT, maxT);

	size_t width = 800;
	size_t height = 800;

	// usiamo unnisgned invece che signed poichè char è signed (-128/+127).
	// Qualsiasi valore di pixel sopra 127 veniva interpretato come negativo, corrompendo il file PPM con colori sbagliati. 
	// unsigned char copre correttamente 0-255, che è il range standard per immagini a 8 bit per canale.
	std::unique_ptr<unsigned char[]> imgbuf = std::make_unique<unsigned char[]>(width * height * 3);

	vec3 rayOrig = transformPoint(cameraToWorld, vec3{ 0,0,0 }); // Calcolo origine del raggio

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

	std::atomic<int> rowsDone{0};
	fprintf(stderr, "Thread disponibili: %d\n", omp_get_max_threads());
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

			float tmin, tmax;
			if (raybox(ray, grid.bounds, tmin, tmax)) {
				integrate(ray, tmin, tmax, L, transmittance, grid, NiiGrid, SiiGrid, SiiSiiGrid, VelGrid, rng, dist);
			}

			float starBrightness = 50.f; // parametro di bilanciamento luminosità stelle


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

	fprintf(stderr, "NII alto (>0.3): %d / %d totali non-zero\n", 
		highNii.load(), totalNonZero.load());


	fprintf(stderr, "Render completato.\n");


	const char* filename = "nebula.ppm";
	std::ofstream ofs;
	ofs.open(filename, std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	ofs.write(reinterpret_cast<const char*>(imgbuf.get()), width * height * 3);
	ofs.close();
}


int main() 
{
	render();
	return 0;
}

