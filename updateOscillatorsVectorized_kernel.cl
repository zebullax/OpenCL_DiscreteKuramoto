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
	
	float4 updatedAngle = vload4(id*4,oscillators);
	float4 frequency = vload4(id*4,frequencies);
	float4 randomNoiseTerm = vload4(id*4,noises);
	
	float couplingStrength=CURRENTCOUPLINGSTRENGTH,r=R,psi=PSI;

	updatedAngle=fma(TIMESTEP,(randomNoiseTerm+frequency+(couplingStrength*r*sin(psi-updatedAngle))),updatedAngle);
	updatedAngle=fmod(updatedAngle,(float)(2*M_PI_F));
	vstore4(updatedAngle,4*id,oscillators);
	
}