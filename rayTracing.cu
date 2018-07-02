#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define WIDTH 1600
#define HEIGHT 1600
#define BLOCK 256
#define MAX_N_OBJECT 10
#define MIN_THRES 1e-5
#define MAX_REPEAT 5
#define AmbientColor make_float3(0.8,0.8,0.8)

/* math operations */
__device__ __host__ inline float3 operator+(float3 a, float3 b){ return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);}
__device__ __host__ inline float3 operator-(float3 a, float3 b){ return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);}
__device__ __host__ inline float3 operator*(float a, float3 b){return make_float3(a*b.x, a*b.y, a*b.z);}
__device__ __host__ inline float3 operator*(float3 b, float a){return make_float3(a*b.x, a*b.y, a*b.z);}
__device__ __host__ inline float3& operator*=(float a, float3& b){b.x*=a;b.y*=a;b.z*=a;return b;}
__device__ __host__ inline bool operator==(float3 a, float3 b){return (a.x==b.x)&&(a.y==b.y)&&(a.z==b.z);}
__device__ __host__ inline bool operator!=(float3 a, float3 b){return !(a==b);}
__device__ inline float3& operator +=(float3& a, float3 b){a.x+= b.x; a.y+=b.y; a.z+=b.z; return a;}
__device__ __host__ inline float dot(float3 a, float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ __host__ inline float len(float3 a){return sqrtf(dot(a,a));}
__device__ __host__ inline float3 normalize(float3 a){return make_float3(a.x/len(a),a.y/len(a),a.z/len(a));}
__device__ __host__ inline float3 cross(float3 a, float3 b) { return make_float3( -a.z * b.y + a.y * b.z, a.z * b.x - a.x * b.z, -a.y * b.x + a.x * b.y );}
__device__ __host__ inline float3 eleProd(float3 a, float3 b){return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);}

struct Color {
    unsigned char b, g, r, a;
  };
  
Color* image;
void writebmpheader(FILE* f, int width, int height) {
    int size = width * height * sizeof(Color);
    struct {
        uint32_t filesz;
        uint16_t creator1;
        uint16_t creator2;
        uint32_t bmp_offset;
    } bmpheader = 
        { size + 54, 0, 0, 54};
    struct {
        uint32_t header_sz;
        int32_t width;
        int32_t height;
        uint16_t nplanes;
        uint16_t bitspp;
        uint32_t compress_type;
        uint32_t bmp_bytesz;
        int32_t hres;
        int32_t vres;
        uint32_t ncolors;
        uint32_t nimpcolors;    
    } dibheader = 
        {40, width, height, 1, 32, 0, size, 0, 0, 0, 0};
    fwrite("BM", 2, 1, f);
    fwrite(&bmpheader, sizeof(bmpheader), 1, f);
    fwrite(&dibheader, sizeof(dibheader), 1, f);
}
void writebmp(const char* filename, const Color* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    writebmpheader(f, width, height);
    fwrite(data, sizeof(Color), width * height, f);
    fclose(f);
}

struct Ray{
    float3 StartPoint, direction;

    __device__ float3 getPoint(float distance)
    {
        return StartPoint+distance*normalize(direction);
    }
};

enum ObjectType {SPHERE, PLANE, CYLINDER, TRIANGLE};

struct IntersectionResult{
    float3 point, normal;
    float distance;
    
    float3 Ka, Kd, Ks;
    float reflectiveness, shininess;

    ObjectType InterType;
    int  objectId;
};

struct Sphere{
    float3 center; 
    float radius;
    int ObjID;
    float3 Ka, Kd, Ks;
    float reflectiveness, shininess;
    __device__ bool DetectIntersection(Ray* ray, IntersectionResult* ir)
    {
        float3 v = center - ray->StartPoint;
        float vProjectToDirection = dot(v, normalize(ray->direction));
        float vProjectToDirectionSqr = vProjectToDirection*vProjectToDirection;
        if (vProjectToDirection <= 0) return false;
        float h = dot(v,v) - vProjectToDirectionSqr;
        if (h < radius*radius)
        {
            // float distance = vProjectToDirection - sqrtf(radius*radius - h);
            float distance = dot(ray->direction, v) - sqrt(radius*radius-h);
            if(distance < ir->distance)
            {
                ir->distance = distance;
                ir->point = ray->getPoint(distance);
                ir->normal = normalize(ir->point - center);
                ir->InterType = SPHERE;
                ir->objectId = ObjID;
                ir->Ka = Ka;
                ir->Kd = Kd;
                ir->Ks = Ks;
                ir->reflectiveness = reflectiveness;
                ir->shininess = shininess;
                return true;
            }
        }
        return false;
    }
};

struct Plane{
    float3 normal, position;
    float scale;
    int ObjID;
    float3 Ka, Kd, Ks;
    float reflectiveness, shininess;
    __device__ bool DetectIntersection(Ray* ray, IntersectionResult* ir)
    {
        float3 v = position - ray->StartPoint;
        float den = dot(ray->direction, normal);
        if (den >= 0) return false;
        float distance = dot(v, normal)/den;
        if (distance < 0) return false;
        if (distance < ir->distance)
        {
            ir->distance = distance;
            ir->point = ray->getPoint(distance);
            ir->normal = normal;
            ir->InterType = PLANE;
            ir->objectId = ObjID;
            ir->Ka = Ka;
            ir->Kd = Kd;
            ir->Ks = Ks;
            ir->reflectiveness = reflectiveness;
            ir->shininess = shininess;
            return true;
        }
        return false;
    }
};

struct Cylinder{
    // the direction equal to p2 - p1, where p2 at the top and p1 at the bottom
    float3 p1, p2, direction;
    float radius;
    int ObjID;
    float3 Ka, Kd, Ks;
    float reflectiveness, shininess;
    
    __device__ bool DetectIntersection(Ray* ray, IntersectionResult* ir)
    {
        float originalDistance = ir->distance;
        float A = dot(ray->direction-dot(ray->direction,direction)*direction,ray->direction-dot(ray->direction,direction)*direction);
        float3 deltaP = ray->StartPoint-p1;
        float B = 2.0*dot(deltaP-dot(deltaP,direction)*direction,ray->direction-dot(ray->direction,direction)*direction);
        float C = dot(deltaP-dot(deltaP,direction)*direction,deltaP-dot(deltaP,direction)*direction)-radius*radius;
        float b24ac = B*B-4*A*C;
        if (b24ac<0 && (normalize(ray->direction)-direction)!=make_float3(0,0,0)) return false;
        float distance1 = (-1.0*B+sqrtf(b24ac))/(2.0*A);
        float3 q1 = ray->getPoint(distance1);
        if (distance1 >=0 && dot(direction, q1-p1)>0 && dot(direction, q1-p2)<0)
        {
            if (distance1<ir->distance)
            {
                ir->distance = distance1;
                ir->normal = normalize(q1-p1-dot(q1-p1,direction)*direction);
                ir->point = ray->getPoint(distance1);
                ir->InterType = CYLINDER;
                ir->objectId = ObjID;
                ir->Ka = Ka;
                ir->Kd = Kd;
                ir->Ks = Ks;
                ir->reflectiveness = reflectiveness;
                ir->shininess = shininess;
            }
        }
        float distance2 = (-1.0*B-sqrtf(b24ac))/(2.0*A);
        float3 q2 = ray->getPoint(distance2);
        if (distance2 >=0 && dot(direction, q2-p1)>0 && dot(direction, q2-p2)<0)
        {
            if (distance2<ir->distance)
            {
                ir->distance = distance2;
                ir->normal = normalize(q2-p1-dot(q2-p1,direction)*direction);
                ir->point = ray->getPoint(distance2);
                ir->InterType = CYLINDER;
                ir->objectId = ObjID;
                ir->Ka = Ka;
                ir->Kd = Kd;
                ir->Ks = Ks;
                ir->reflectiveness = reflectiveness;
                ir->shininess = shininess;
            }
        }
        float3 normal = normalize(-1*direction);
        float3 v1 = p1 - ray->StartPoint;
        float den = dot(ray->direction, normal);
        if (den != 0)
        {
            float distance = dot(v1, normal)/den;
            if (distance >0 && len(ray->getPoint(distance)-p1)<=radius && distance < ir->distance)
            {
                ir->distance = distance;
                ir->point = ray->getPoint(distance);
                ir->normal = normal;
                ir->InterType = CYLINDER;
                ir->objectId = ObjID;
                ir->Ka = Ka;
                ir->Kd = Kd;
                ir->Ks = Ks;
                ir->reflectiveness = reflectiveness;
                ir->shininess = shininess;
            }
        }
        normal = -1.0*normal;
        float3 v2 = p2 - ray->StartPoint;
        den = dot(ray->direction, normal);
        if (den != 0)
        {
            float distance = dot(v2, normal)/den;
            if (distance >0 && len(ray->getPoint(distance)-p2)<=radius && distance < ir->distance)
            {
                ir->distance = distance;
                ir->point = ray->getPoint(distance);
                ir->normal = normal;
                ir->InterType = CYLINDER;
                ir->objectId = ObjID;
                ir->Ka = Ka;
                ir->Kd = Kd;
                ir->Ks = Ks;
                ir->reflectiveness = reflectiveness;
                ir->shininess = shininess;
            }
        }
        if (ir->distance == originalDistance)
        {
            return false;
        }
        else{
            return true;
        }
    }
};

struct Triangle{
    float3 VertexA, VertexB, VertexC;
    int ObjID;
    float3 Ka, Kd, Ks;
    float reflectiveness, shininess;

    __device__ bool DetectIntersection(Ray* ray, IntersectionResult* ir)
    {
        float a = VertexA.x - VertexB.x;
        float b = VertexA.y - VertexB.y;
        float c = VertexA.z - VertexB.z;
        float d = VertexA.x - VertexC.x;
        float e = VertexA.y - VertexC.y;
        float f = VertexA.z - VertexC.z;
        float g = ray->direction.x;
        float h = ray->direction.y;
        float i = ray->direction.z;
        float3 AdE = VertexA - ray->StartPoint;
        float j = AdE.x;
        float k = AdE.y;
        float l = AdE.z;
        float M = a*(e*i-h*f)+b*(g*f-d*i)+c*(d*h-e*g);
        float distance = -1.0 * (f*(a*k-j*b)+e*(j*c-a*l)+d*(b*l-k*c)) / M;
        if(distance > ir->distance || distance<0) return false;
        float gamma = (i*(a*k-j*b)+h*(j*c-a*l)+g*(b*l-k*c))/M;
        if (gamma<0 || gamma >1) return false;
        float beta = (j*(e*i-h*f)+k*(g*f-d*i)+l*(d*h-e*g))/M;
        if (beta<0 || beta>1-gamma) return false;
        ir->distance = distance;
        ir->point = ray->getPoint(distance);
        ir->normal = normalize(cross(VertexA-VertexB, VertexA-VertexC));
        if(dot(ir->normal, ray->direction)>0) ir->normal = -1 * ir->normal;
        ir->InterType = TRIANGLE;
        ir->objectId = ObjID;
        ir->Ka = Ka;
        ir->Kd = Kd;
        ir->Ks = Ks;
        ir->reflectiveness = reflectiveness;
        ir->shininess = shininess;
        return true;
    }
};

struct LightSource{
    float3 position, color; 
};

LightSource makeLightSource(float3 position, float3 color)
{
    LightSource ls={position, color};
    return ls;
}


Sphere makeSphere(float3 center, float radius, int id,float3 Ka, float3 Kd, float3 Ks, float reflectiveness, float shininess)
{
    Sphere tmpSp={center, radius, id, Ka, Kd, Ks, reflectiveness, shininess};
    return tmpSp;
}


Plane makePlane(float3 normal, float3 position, float scale, int id, float3 Ka, float3 Kd, float3 Ks, float reflectiveness, float shininess)
{
    Plane tmpPl={ normal, position, scale, id, Ka, Kd, Ks, reflectiveness, shininess};
    return tmpPl;
}


Cylinder makeCylinder(float3 p1, float3 p2, float radius, int id, float3 Ka, float3 Kd, float3 Ks, float reflectiveness, float shininess)
{
    Cylinder tmpCylinder={
        p1, p2, normalize(p2-p1),
        radius, id,
        Ka, Kd, Ks, reflectiveness, shininess
    };
    return tmpCylinder;
}

Triangle makeTriangle(float3 VertexA, float3 VertexB, float3 VertexC, int id, float3 Ka, float3 Kd, float3 Ks, float reflectiveness, float shininess)
{
    Triangle tmpTri={ VertexA, VertexB, VertexC, id, Ka, Kd, Ks, reflectiveness, shininess };
    return tmpTri;
}

struct PerspectiveCamera{
    float3 position, fov, up, right;

    __device__ Ray generateRay(int x, int y){
        float3 r = right * ((float(x)/HEIGHT-0.5));
        float3 u = up * ((float(y)/HEIGHT-0.5));
        Ray tmpRay={position, normalize(fov+r+u)};
        return tmpRay;
    }
}camera={make_float3(0,15,40), make_float3(0,0,-1), make_float3(0, 1, 0), make_float3(1,0,0)};

struct Scene{
    PerspectiveCamera camera;
    int nSphere, nPlane, nCylinder, nTriangle, nLightSource;
    Sphere spheres[MAX_N_OBJECT];
    Plane planes[MAX_N_OBJECT];
    Cylinder cylinders[MAX_N_OBJECT];
    Triangle triangles[MAX_N_OBJECT];
    LightSource lightSources[MAX_N_OBJECT];
}cpuScene={
    camera,
    2,// #Sphere
    1,// #Plane
    1,// #Cylinder
    3,// #Triangle
    2,// #LightSource
    {
        makeSphere(make_float3(0,0,-10),5,0,make_float3(0.5,0.5,0.5),make_float3(0.1,0.25,0.0),make_float3(1.0,0.0,1.0),0.0,4.0),
        makeSphere(make_float3(-12,0,-10),5,0,make_float3(0.5,0.5,0.5),make_float3(0.1,0.25,0.0),make_float3(1.0,0.0,1.0),0.0,4.0),
    },// array of Sphere
    {
        makePlane(make_float3(0,1,0), make_float3(0,-10,-10), 1, 1,make_float3(0.5,0.5,0.5),make_float3(0.1,0.25,0.0),make_float3(1.0,0.0,1.0), 0.0, 4.0),
    },// array of Plane
    {
        makeCylinder(make_float3(10,-10,-20),make_float3(10,10,-20),5,2,make_float3(0.1,0.1,0.1),make_float3(0.1,0.9,0.1),make_float3(0.8,0.0,1.0),0.0,4.0),
    },// array of Cylinde
    {
        makeTriangle(make_float3(10,0,-10),make_float3(5,-10,-10),make_float3(15,-10,-8),0,make_float3(0.1,0.1,0.1),make_float3(0.1,0.9,0.1),make_float3(0.8,0.0,1.0),0.0,4.0),
        makeTriangle(make_float3(10,0,-10),make_float3(15,-10,-8),make_float3(20,-10,-15),0,make_float3(0.1,0.1,0.1),make_float3(0.1,0.9,0.1),make_float3(0.8,0.0,1.0),0.0,4.0),
        makeTriangle(make_float3(10,0,-10),make_float3(5,-10,-10),make_float3(20,-10,-15),0,make_float3(0.1,0.1,0.1),make_float3(0.1,0.9,0.1),make_float3(0.8,0.0,1.0),0.0,4.0),
    },// array of Triangle
    {
        makeLightSource(make_float3(20,15,0),make_float3(0.8,0.8,0.8)),
        makeLightSource(make_float3(-20,10,0),make_float3(1,1,1)),
    }// array of LightSource
};
__constant__ Scene gpuScene;

__device__ bool DetectIntersection(Ray *ray, IntersectionResult* ir)
{
    bool ifInter = false;
    int i;
    for (i=0;i<gpuScene.nSphere;i++){ifInter = gpuScene.spheres[i].DetectIntersection(ray,ir)||ifInter;}
    for (i=0;i<gpuScene.nPlane;i++){ifInter = gpuScene.planes[i].DetectIntersection(ray,ir)||ifInter;}
    for (i=0;i<gpuScene.nCylinder;i++){ifInter = gpuScene.cylinders[i].DetectIntersection(ray,ir)||ifInter;}
    for (i=0;i<gpuScene.nTriangle;i++){ifInter = gpuScene.triangles[i].DetectIntersection(ray,ir)||ifInter;}
    return ifInter;
}
__device__ float3 computeColor(Ray *ray, IntersectionResult*ir)
{
    float3 p = ray->getPoint(ir->distance);
    float3 color = eleProd(AmbientColor, ir->Ka);
    for(int lightSrc=0;lightSrc<gpuScene.nLightSource;lightSrc+=1)
    {
        Ray p2l = {p+(ir->normal*1e-2), gpuScene.lightSources[lightSrc].position-p};
        IntersectionResult p2lIR;
        p2lIR.distance = FLT_MAX;
        if (DetectIntersection(&p2l,&p2lIR)) continue;
        float3 l = normalize(gpuScene.lightSources[lightSrc].position-p);
        float3 Id = fmaxf(0, dot(ir->normal, l))*(eleProd(ir->Kd, gpuScene.lightSources[lightSrc].color));
        float3 h = normalize(normalize(l)-normalize(ray->direction));
        float3 Is = __powf(fmaxf(0,dot(ir->normal, h)),ir->shininess)*eleProd(ir->Ks,gpuScene.lightSources[lightSrc].color);
        color += (Id+Is);
    }
    return color;
}

__device__ inline float3 sample(Ray* ray)
{
    float3 FinalColor = make_float3(0,0,0);// default Background Color
    int repeat = 0;
    float reflectionFactor = 1;
    float r = 1.0;
    float3 tmpColor = make_float3(0,0,0);
    while (repeat < MAX_REPEAT)
    {
        IntersectionResult ir;
        ir.distance = FLT_MAX;
        if (!DetectIntersection(ray, &ir)) break;

        FinalColor += (1-r)*reflectionFactor*tmpColor;
        reflectionFactor = reflectionFactor * r;
        r = ir.reflectiveness;
        tmpColor = computeColor(ray, &ir);
        repeat += 1;

        if (r>0 && reflectionFactor>MIN_THRES){
            float3 d = normalize(ray->direction);
            float3 n = normalize(ir.normal);
            float3 newRayE = ray->getPoint(ir.distance);
            float3 newRayD = normalize(d+2.0*dot(d,n)*n);
            ray->StartPoint = newRayE;
            ray->direction = newRayD;
        }
        else break;
    }
    return FinalColor + reflectionFactor*tmpColor;
}

__global__ void RayTracing(unsigned * gpuImage){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int x = index%WIDTH, y = index / WIDTH;
    Ray pixelRay = gpuScene.camera.generateRay(x,y);
    float3 c = sample(&pixelRay);
    unsigned char c4[] = {
        __saturatef(c.z) * 255,
        __saturatef(c.y) * 255,
        __saturatef(c.x) * 255,
        255};

    unsigned ct = *reinterpret_cast<unsigned*>(c4); 
    gpuImage[index] = ct;
}

int main()
{
    unsigned *gpuImage;
    cudaSetDevice(0);
    cudaMallocHost(&image, WIDTH*HEIGHT*sizeof(Color));
    cudaMalloc(&gpuImage, sizeof(Color)*WIDTH*HEIGHT);
    cudaMemcpyToSymbol(gpuScene, &cpuScene, sizeof(Scene));
    RayTracing<<<WIDTH*HEIGHT/BLOCK, BLOCK>>>(gpuImage);
    cudaMemcpy(image, gpuImage, sizeof(Color)*WIDTH*HEIGHT,cudaMemcpyDeviceToHost);
    cudaFree(gpuImage);
    writebmp("ratracing_out.bmp",image,WIDTH,HEIGHT);
    cudaFreeHost(image);
    return 0;
}