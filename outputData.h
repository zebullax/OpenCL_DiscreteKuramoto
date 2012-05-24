#ifndef OUTPUTDATA
#define OUTPUTDATA
#include <iostream>
#include <fstream>

using namespace std;
#define SEP " "

template <class T>
void outputVectorToFile(char* name,T* X, int length,bool overWrite,bool transpose)
{
	ofstream outfile(name,(overWrite?ofstream::app:ofstream::out));
	outputVectorToStream(outfile,X,length,transpose);
	outfile.close();
}

template <class T>
void outputVectorToStream(ostream& target,T* X, int length,bool transpose)
{
    for(int j=0;j<length;++j)
    {
		if(transpose)
			target<<X[j]<<endl;
		else
			target<<X[j]<<SEP;      
    }
    target<<endl;
}

//2d Version... (tranpose is not available here)
template <class T>
void output2DArrayToFile(char* name,T** X, int nbRows,int nbCols,bool overWrite)
{
	ofstream outfile(name,(overWrite?ofstream::app:ofstream::out));
	for(int i=0;i<nbRows;++i)
	  {
	    outputVectorToStream(outfile,X[i],nbCols,false);
	  }
	outfile.close();
}

void makeFileName(char* buffer, float couplingStrength,const char* prefix, const char* suffix)
{
	char temp[10];
	strcpy(buffer,prefix);
	sprintf(temp,"%3.2f",couplingStrength);
	strcat(buffer,temp);
	strcat(buffer,suffix);
}
#endif
