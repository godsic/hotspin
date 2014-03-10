#include "brillouin.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__device__ double findroot_Ridders_Ts(funcTs* f, double J, double mult, double C, double xa, double xb)
{

    double ya = f[0](xa, J, mult, C);
    if (fabs(ya) < zero) return xa;
    double yb = f[0](xb, J, mult, C);
    if (fabs(yb) < zero) return xb;

    double y1 = ya;
    double x1 = xa;
    double y2 = yb;
    double x2 = xb;

    double x = 1.0e10;
    double y = 1.0e10;
    double tx = x;

    double teps = x;

    double x3 = 0.0;
    double y3 = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    int iter = 0;
    while (teps > eps && iter < 1000)
    {

        x3 = 0.5 * (x2 + x1);
        y3 = f[0](x3, J, mult, C);

        dy = (y3 * y3 - y1 * y2);
        if (dy == 0.0)
        {
            x = x3;
            break;
        }

        dx = (x3 - x1) * sign(y1 - y2) * y3 / (sqrt(dy));

        x = x3 + dx;
        y = f[0](x, J, mult, C);

        y2 = (signbit(y) == signbit(y3)) ? y2 : y3;
        x2 = (signbit(y) == signbit(y3)) ? x2 : x3;

        y2 = (signbit(y) == signbit(y1) || x2 == x3) ? y2 : y1;
        x2 = (signbit(y) == signbit(y1) || x2 == x3) ? x2 : x1;

        y1 = y;
        x1 = x;

        teps = fabs((x - tx) / (tx + x));

        tx = x;
        iter++;

    }
    return x;
}

// here n = <Sz>/ S
// <Sz> = n * S
// <Sz> = S * Bj(S*J0*<Sz>/(kT))

__device__ double ModelTs(double n, double J, double pre, double C)
{
    double x = (n == 0.0) ? 1.0e38 : pre / n;
    double val = Bj(J, x) - C;
    return val;
}

__device__ funcTs pModelTs = ModelTs;

__global__ void tsKern(double* __restrict__ Ts,
                      double* __restrict__ msatMsk,
                      double* __restrict__ msat0T0Msk,
                      double* __restrict__ TcMsk,
                      double* __restrict__ SMsk,
                      const double msatMul,
                      const double msat0T0Mul,
                      const double TcMul,
                      const double SMul,
                      int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        double msat0T0 = msat0T0Mul * getMaskUnity(msat0T0Msk, i);
        if (msat0T0 == 0.0)
        {
            Ts[i] = 0.0;
            return;
        }

        double msat = msatMul * getMaskUnity(msatMsk, i);
        if (msat == msat0T0) {
        	Ts[i] = 0.0;
        	return;
        }

        double Tc = TcMul * getMaskUnity(TcMsk, i);
        if (msat == 0.0)
        {
            Ts[i] = Tc;
            return;
        }

        double S  = (SMsk  == NULL) ? SMul  : SMul  * SMsk[i];

        double J0  = 3.0 * Tc / (S * (S + 1.0));
        double m = msat / msat0T0;
        double pre = S * S * J0 * m;
        double T = findroot_Ridders_Ts(&pModelTs, S, pre, m, 0.0, Tc);

        Ts[i] = (double)T;
    }
}

__export__ void tsAsync(double* Ts,
                              double* msat,
                              double* msat0T0,
                              double* Tc,
                              double* S,
                              const double msatMul,
                              const double msat0T0Mul,
                              const double TcMul,
                              const double SMul,
                              int Npart,
                              CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    tsKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Ts,
            msat,
            msat0T0,
            Tc,
            S,
            msatMul,
            msat0T0Mul,
            TcMul,
            SMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
