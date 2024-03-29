//
//	Kernel updates all the oscillators phase values.
//
//	Oscillators: array of oscillators angles
//	Frequencies: Oscillators nat freqs
//	Noises: Array of normally distributed gaussian noise
//	R & PSI : field parameters
__kernel void updateAngle(__global float4* oscillators,__global float4* frequencies, __global float4* noises, __const float TIMESTEP,__const float CURRENTCOUPLINGSTRENGTH,__const float R, __const float PSI)
{
	int id = get_global_id(0);	
	
	float4 updatedAngle=oscillators[id];	
	float4 frequency=frequencies[id];
	float4 randomNoiseTerm=noises[id];	
	
	float couplingStrength=CURRENTCOUPLINGSTRENGTH,r=R,psi=PSI;

	updatedAngle=fma(TIMESTEP,(randomNoiseTerm+frequency+(couplingStrength*r*sin(psi-updatedAngle))),updatedAngle);
	updatedAngle=fmod(updatedAngle,(float)(2*M_PI_F));
	oscillators[id]=updatedAngle;
}