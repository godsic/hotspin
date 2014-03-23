#include "brillouin.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__device__ double findroot_Ridders(func* f, double J, double mult, double xa, double xb)
{

    double ya = f[0](xa, J, mult);
    if (fabs(ya) < zero) return xa;
    double yb = f[0](xb, J, mult);
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
        y3 = f[0](x3, J, mult);

        dy = (y3 * y3 - y1 * y2);
        if (dy == 0.0)
        {
            x = x3;
            break;
        }

        dx = (x3 - x1) * sign(y1 - y2) * y3 / (sqrt(dy));

        x = x3 + dx;
        y = f[0](x, J, mult);

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


// here n = m / me
// <Sz> = n * J
// <Sz> = J * Bj(S*J0*<Sz>/(kT))

__device__ double Model(double n, double J, double pre)
{
    double x = pre * n;
    double val = (J > INFINITESPINLIMIT) ? L(x) : Bj(J, x);
    val = val - n;
    return val;
}

__device__ func pModel = Model;

__global__ void brillouinKern(double* __restrict__ msat0Msk,
                              double* __restrict__ msat0T0Msk,
                              double* __restrict__ T,
                              double* __restrict__ TcMsk,
                              double* __restrict__ SMsk,
                              const double msat0Mul,
                              const double msat0T0Mul,
                              const double TcMul,
                              const double SMul,
                              int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double Temp = T[i];

        double msat0T0 = msat0T0Mul * getMaskUnity(msat0T0Msk, i);

        if (msat0T0 == 0.0)
        {
            msat0Msk[i] = 0.0;
            return;
        }

        if (Temp == 0.0)
        {
            msat0Msk[i] = msat0T0 / msat0Mul;
            return;
        }

        double Tc = TcMul * getMaskUnity(TcMsk, i);

        if (Temp > Tc)
        {
            msat0Msk[i] = 0.0;
            return;
        }

        double S  = SMul * getMaskUnity(SMsk, i);

        double preS = (S > INFINITESPINLIMIT) ? 1.0 : S / (S + 1.0);
        double pre = 3.0 * preS * (Tc / Temp);

        double dT = (Tc - Temp) / Tc;
        double lowLimit = (dT < 0.0004) ? -0.1 : 0.01;
        double hiLimit  = (dT < 0.0004) ?  0.5 : 1.1;
        double msat0 = findroot_Ridders(&pModel, S, pre, lowLimit, hiLimit);

        msat0Msk[i] = msat0T0 * fabs(msat0) / (msat0Mul);
    }
}

__export__ void brillouinAsync(double* msat0,
                               double* msat0T0,
                               double* T,
                               double* Tc,
                               double* S,
                               const double msat0Mul,
                               const double msat0T0Mul,
                               const double TcMul,
                               const double SMul,
                               int Npart,
                               CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    brillouinKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (msat0,
            msat0T0,
            T,
            Tc,
            S,
            msat0Mul,
            msat0T0Mul,
            TcMul,
            SMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
