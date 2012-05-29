/*
	Simulation of Kuramoto model
	Final coherency values is averaged over a predefined nb of runs
*/

#pragma warning(disable: 4290) 

#define _CRT_SECURE_NO_WARNINGS
#define __CL_ENABLE_EXCEPTIONS
#define __NO_STD_STRING

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <iostream>
#include <fstream>

#include <CL/cl.hpp>
//C:\Program Files\AMD APP\lib\x86
#include "simuParameters.h"
#include "outputData.h"
#include "numerical.h"

void main()
{
	/* OPENCL Structs **********/
	const char* PlatformName = "Advanced Micro Devices, Inc."; //ATI HD 4850
	cl_int err=0;	
	cl::Context context;
	std::vector<cl::Platform> platforms; //platform = openCl implem , program = kernel container
	std::vector<cl::Device> devices;
	cl::Buffer anglesBuffer;
	cl::Buffer frequenciesBuffer;
	cl::Buffer noisesBuffer;
	cl::CommandQueue queue;
	cl::NDRange offset(0);
	cl::NDRange global_size(SCALED_NB_OSCILLOS);
	cl::NDRange local_size(1);
	cl::Event profilingEvent;
	/* Simu data */
	float currentCouplingStrength=COUPLING_MIN;
	float r=0,psi=0;
	int idx=0;
	float averagedR=0;
	//since we make multiple run for averaging the final coherency value , we dont wanna dump all trials , just one... this flag does just that
	bool dumpedAllAngles=false;
	double time =.0;
	/*we could just go static alloc on those var as well..*/
	float *angles=0,*frequencies=0,*whiteNoises;
	float* finalCoherencyValue=0; //final r for each step of the coupling strength range
	/* 
	quite a big tab..
	the other way around would be to dump in file as soon as data are available BUT I/O cost would be a bitch...
	in between would be fine
	*/
	float** anglesToDump=0;
	// Output Filenames 
	char angleDistribFilename[35];
	char angleDumpFilename[35];	
	
	try
	{
		angles = new float[NBOSCILLO];
		frequencies= new float[NBOSCILLO];
		whiteNoises = new float[NBOSCILLO];
		finalCoherencyValue = new float[(int)(1+(COUPLING_MAX-COUPLING_MIN)/COUPLING_STEP)];

		/********************************
			...OPEN CL	init... 
		*********************************/
		//list platforms and get the one we want
		cl::Platform::get(&platforms);
		std::vector<cl::Platform>::iterator iter;
		for (iter=platforms.begin();iter!=platforms.end();++iter)
		{
			if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),PlatformName)) //Looking for my AMD.
				break;
		}
		//create the context on that platform and select our gpu device
		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(*iter)(), 0};
		context = cl::Context(CL_DEVICE_TYPE_GPU, cps); 
		devices = context.getInfo<CL_CONTEXT_DEVICES>();		
		//now bind the command queue to the device
#ifdef PROFILING
		queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE); 
#else
		queue = cl::CommandQueue(context, devices[0], 0,); 
#endif		
		//Create the buffer containing the phases, frequencies, and noises
		anglesBuffer=cl::Buffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float)*NBOSCILLO,angles);
		frequenciesBuffer=cl::Buffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*NBOSCILLO,frequencies);
		noisesBuffer=cl::Buffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*NBOSCILLO,whiteNoises);
		/* BUILD THE PROGRAM & MAKE THE KERNEL */		
        std::ifstream sourceFile("updateOscillators_kernel.cl");
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),sourceCode.length()+1));         
        // Make program of the source code in the context
		cl::Program program = cl::Program(context, source); 
        // Build program for these specific devices
		program.build(devices,"-cl-fast-relaxed-math -cl-mad-enable"); 
		// Make the kernel
        cl::Kernel kernel(program, "updateAngle");
		/*********************************/		
		/*********************************/		
#ifdef DUMPALLANGLES
		//Init the 2d huge dump array
		anglesToDump = new float* [NB_OF_TIMESTEPS];
		for (int i=0;i<NB_OF_TIMESTEPS;++i)
			anglesToDump[i]=new float[NBOSCILLO];
#endif

		while(currentCouplingStrength<=COUPLING_MAX)
		{
			averagedR=0;
			dumpedAllAngles=false; 
			for(int currentRun=0;currentRun<NBOFRUNSFORAVERAGING;++currentRun)
			{
				time=.0;
#ifdef DEBUGCS
			std::cout<<currentRun<<" - Coupling strength:"<<currentCouplingStrength<<endl;		    
#endif
				/* oscillators init values*/
				InitOscillators(frequencies,angles,NBOSCILLO,STDDEV);
				MakeSomeNoise(whiteNoises,NBOSCILLO,NOISE_STRENGTH,STDDEV);
				r=psi=0;
				/* Enqueue args for kernels and copy buffers */
				queue.enqueueWriteBuffer(anglesBuffer,CL_TRUE,0,sizeof(float)*NBOSCILLO,angles,NULL,NULL);
				queue.enqueueWriteBuffer(frequenciesBuffer,CL_TRUE,0,sizeof(float)*NBOSCILLO,frequencies,NULL,NULL);
				queue.enqueueWriteBuffer(noisesBuffer,CL_TRUE,0,sizeof(float)*NBOSCILLO,whiteNoises,NULL,NULL);
				/* Set arg for our kernel */
				kernel.setArg(0,anglesBuffer);
				kernel.setArg(1,frequenciesBuffer);
				kernel.setArg(2,noisesBuffer);
				kernel.setArg(3,TIMESTEP);
				kernel.setArg(4,currentCouplingStrength);
				kernel.setArg(5,r);
				kernel.setArg(6,psi);

				/* SIMU LOOP ON TIMESTEP */
				for (int k=0;k<NB_OF_TIMESTEPS;++k)
				{
#ifdef DEBUGTS
					std::cout<<"Timestep:"<<k<<endl;
#endif
					//todo : can we compute r and psi on gpus ?
					ComputeOrderParameters(angles,r,psi,NBOSCILLO);
					queue.enqueueNDRangeKernel(kernel,offset,global_size,local_size,NULL,&profilingEvent);
					queue.finish();
#ifdef PROFILING
					cl_ulong start=profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
					cl_ulong end=profilingEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
					time += 1.e-9 * (end-start);					
#endif

#ifdef DUMPALLANGLES
					if(!dumpedAllAngles)
					{
						 queue.enqueueReadBuffer(anglesBuffer, CL_TRUE, 0, NBOSCILLO * sizeof(float), anglesToDump[k]);
					}		
#endif
				}
#ifdef PROFILING
				std::cout << "Average time for kernel to execute " << time/NB_OF_TIMESTEPS << endl;
#endif
				averagedR+=r;
#ifdef DUMPALLANGLES
				/* For debug purpose , we dump here all the timesteps phase value */
				if(!dumpedAllAngles)
				{
					makeFileName(angleDumpFilename,currentCouplingStrength,"dumpAllAngles_",".dat");
					output2DArrayToFile(angleDumpFilename,anglesToDump,NB_OF_TIMESTEPS,NBOSCILLO+1,false);
					dumpedAllAngles=true;
				}
#endif
			}
			/*
				Output file : phase dristrib
			*/
			queue.enqueueReadBuffer(anglesBuffer, CL_TRUE, 0, NBOSCILLO * sizeof(float), angles);
			for (int i=0;i<NBOSCILLO;++i)
				angles[i] = abs(angles[i]-psi);
			makeFileName(angleDistribFilename,currentCouplingStrength,"phase_",".dat");
			outputVectorToFile(angleDistribFilename,angles,NBOSCILLO,false,true);
			
			//Store Final r and update coupling strength to next value
			finalCoherencyValue[idx]=averagedR/NBOFRUNSFORAVERAGING;
			currentCouplingStrength+=COUPLING_STEP;
			dumpedAllAngles=false;
			++idx;
		}
		/*
			Output file : coherency
		*/	
		outputVectorToFile("coherency.csv",finalCoherencyValue,(int)(1+(COUPLING_MAX-COUPLING_MIN)/COUPLING_STEP),false,true);	

		/********CLEANUP***********/
		delete []angles;angles=0;
		delete []frequencies;frequencies=0;	
		delete []finalCoherencyValue;finalCoherencyValue=0;
#ifdef DUMPALLANGLES
		for (int i = 0; i < NB_OF_TIMESTEPS; i++)
		{
			delete[]anglesToDump[i];
		}
		delete[]anglesToDump;anglesToDump=0;
#endif
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
		if (angles!=0) delete []angles;
		if (frequencies!=0) delete []frequencies;
		if (finalCoherencyValue!=0) delete []finalCoherencyValue;
#ifdef DUMPALLANGLES
		for (int i = 0; i < NB_OF_TIMESTEPS; i++)
		{
			delete[]anglesToDump[i];
		}
		delete[]anglesToDump;anglesToDump=0;
#endif
	}
	catch(...)
	{
		if (angles!=0) delete []angles;
		if (frequencies!=0) delete []frequencies;
		if (finalCoherencyValue!=0) delete []finalCoherencyValue;
#ifdef DUMPALLANGLES
		for (int i = 0; i < NB_OF_TIMESTEPS; i++)
		{
			delete[]anglesToDump[i];
		}
		delete[]anglesToDump;anglesToDump=0;
#endif
	}
}
