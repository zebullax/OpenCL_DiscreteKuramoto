//
//	Kernel updates all the oscillators phase values.
//
//	Oscillators: array of oscillators angles
//	Frequencies: Oscillators nat freqs
//	Noises: Array of normally distributed gaussian noise
//	R & PSI : field parameters
__kernel void updateAngle(__global float* oscillators,__global float* frequencies, __global float* noises, __const float TIMESTEP,__const float CURRENTCOUPLINGSTRENGTH,__const float R, __const float PSI)
{
	int id = get_global_id(0);	
	float updatedAngle=oscillators[id];
	float frequency=frequencies[id];
	float randomNoiseTerm=noises[id];
	
	updatedAngle=fma(TIMESTEP,(randomNoiseTerm+frequency+(CURRENTCOUPLINGSTRENGTH*R*sin(PSI-updatedAngle))),updatedAngle);
	updatedAngle=fmod(updatedAngle,(float)(2*M_PI_F));
}