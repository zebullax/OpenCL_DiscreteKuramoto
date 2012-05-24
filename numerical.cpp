#define PI 3.141592f
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "numerical.h"
#include "simuParameters.h"

float RK4(float h,float (*deriv)(float,float),float x0,float t0)
{
	float k1=0,k2=0,k3=0,k4=0;
	float result=0;

	k1=h*(*deriv)(t0,x0);
	k2=h*(*deriv)(t0+h/2,x0+k1/2);
	k3=h*(*deriv)(t0+h/2,x0+k2/2);
	k4=h*(*deriv)(t0+h,x0+k3);
	result = x0+k1/6+k2/3+k3/3+k4/6;

	return result;
}

/* Park Miller RNG */
#define a 16807         /* multiplier */
#define m 2147483647L   /* 2**31 - 1 */
static long randomnum = 1;
long nextlongrand(long seed)
{
      unsigned long lo, hi;

      lo = a * (long)(seed & 0xFFFF);
      hi = a * (long)((unsigned long)seed >> 16);
      lo += (hi & 0x7FFF) << 16;
      if (lo > m)
      {
            lo &= m;
            ++lo;
      }
      lo += hi >> 15;
      if (lo > m)
      {
            lo &= m;
            ++lo;
      }
      return (long)lo;
}
long longrand(void)                     /* return next random long */
{
      randomnum = nextlongrand(randomnum);
      return randomnum;
}
//return an uniformly distrib nb between 0 and 1 
float random()
{
	long randTerm = longrand();
	return ((float)(randTerm/m));
}

//Gives 2 normally distrib number, mean 0 , stddev in param)
void BoxMullerTransform(float& z1,float& z2,float stdddev)
{
	float u1 =(random());		
	float u2 =(random());
	while(u1<=0.)
		u1 =(random());		
	while(u2<=0.)
		u2 =(random());		
	z1=stdddev*(sqrt(-2*log(u1))*cos((float)(2*PI*u2)));
	z2=stdddev*(sqrt(-2*log(u2))*sin((float)(2*PI*u1)));
}


//Fill the buffer with normally distributed noize
void MakeSomeNoise(float* noiseBuffer, int length, float noiseStrength, float stdDev)
{
	float w1,w2;
	for (int i=0;i<length;++i)
	{
		if (i%2==0)
		{
			BoxMullerTransform(w1,w2,stdDev);
			noiseBuffer[i]=noiseStrength*w1;
		}
		else
			noiseBuffer[i]=noiseStrength*w2;		
	}
}

/*************************************************************************** 
	Kuramoto model specific numerical functions 
***************************************************************************/

void InitOscillators(float* frequencies, float* angles, int nbOscillo, float stdDev)
{
	//Frequencies are normally distributed
	//Angles are random
	srand ((unsigned int)time(NULL));
	float w1,w2;
	for (int i=0;i<nbOscillo;++i)
	{	
		if (i%2==0)
		{
			BoxMullerTransform(w1,w2,stdDev);
			frequencies[i]=abs(w1)*TIMESTEP;
		}
		else
		{
			frequencies[i]=abs(w2)*TIMESTEP;
		}					
		angles[i]=fmod((float)(rand()),(float)(2*PI));
	}
}

void ComputeOrderParameters(float* angles, float& r,float& psi,int nbOscillo)
{
	float cosSum=0,sinSum=0,meanAngle=0;
	for (int i=0;i<nbOscillo;++i)
	{		
		cosSum+=cos(angles[i]);
		sinSum+=sin(angles[i]);
		meanAngle+=angles[i];
	}
	cosSum/=nbOscillo;
	sinSum/=nbOscillo;
	r=sqrt(cosSum*cosSum + sinSum*sinSum);
	psi=meanAngle/nbOscillo;
}
