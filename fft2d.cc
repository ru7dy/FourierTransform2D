// Distributed two-dimensional Discrete FFT transform
// YOUR NAME HERE
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;

const double PI = 3.1415926;
void Transform1D(Complex* h, int w, Complex* H);

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
  int numtasks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << "Number of tasks are " << numtasks << ", my rank is " << rank << endl;
  // master node 0 read the input
  
  int height = image.GetHeight();
  int width = image.GetWidth();
  int len = height * width;
  int local_num = width * height / numtasks;
  Complex* input = image.GetImageData();

  // param for MPI_Scatterv and MPI_Gatherv
  Complex* after1d = new Complex[len];
  int sendcount[numtasks];
  int displ[numtasks];
  for(int i = 0; i < numtasks; i++) {
    sendcount[i] = local_num;
    displ[i] = local_num * i;
  }
  double recvbuf[local_num]; // for slave nodes to receive from node, do 1d fft and send back
  double sendbuf[len]; // for master node to transform, send and receive from slaves

  // since all nodes read the input, do the first 1d fft locally
  Complex* after1d_local = new Complex[local_num];
  int startRow = height * rank / numtasks;
  int global_offset; // for input
  int local_offset; // for local output
  for(int i = 0; i < height / numtasks; i++) {
    global_offset = width * (startRow + i);
    local_offset = width * i;
    Transform1D(input + global_offset, width, after1d_local + local_offset);
  }

  // master node gathers data, real part
  for(int i = 0; i < local_num; i++) {
    recvbuf[i] = after1d_local[i].real;
  }
  MPI_Gatherv(recvbuf, local_num, MPI_DOUBLE, sendbuf, sendcount, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for(int i = 0; i < len; i++) {
    after1d[i].real = sendbuf[i];
  }

  // master node gathers data, imag part
  for(int i = 0; i < local_num; i++) {
    recvbuf[i] = after1d_local[i].imag;
  }
  MPI_Gatherv(recvbuf, local_num, MPI_DOUBLE, sendbuf, sendcount, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for(int i = 0; i < len; i++) {
    after1d[i].imag = sendbuf[i];
  }

  if(rank == 0)
  {
    cout<<"Generating Image File MyAfter1d.txt"<<endl;
    image.SaveImageData("MyAfter1d.txt", after1d, width, height);
  }

  delete[] after1d;
  delete[] after1d_local;
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  double coef = 2 * PI / w;

  for(int n = 0; n < w; n++) {
    for(int k = 0; k < w; k++) {
      Complex Wnk(cos(coef * n * k), sin(coef * n * k));
      H[n] = Wnk * h[k] + H[n];
    }
  }
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  int rc; 
  rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
    // printf ("Error starting MPI program. Terminating.\n");
    cerr << "Error starting MPI program. Terminating.\n" << endl;
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  Transform2D(fn.c_str()); // Perform the transform.

  MPI_Finalize();
}  
  

  
