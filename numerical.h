#ifndef NUMERICAL_H
#define NUMERICAL_H

/***************************************************************************
	General purpose numerical functions
***************************************************************************/

float RK4(float h,float (*deriv)(float,float),float x0,float t0);
void BoxMullerTransform(float& z1,float& z2,float stdddev);
float random();
void MakeSomeNoise(float* noiseBuffer, int length, float noiseStrength, float stdDev);
void InitOscillators(float* frequencies, float* angles, int nbOscillo, float stdDev);
void ComputeOrderParameters(float* angles, float& r,float& psi,int nbOscillo);
void MakeSomeNoise(float* noiseBuffer, int length, float noiseStrength, float stdDev);

#endif
