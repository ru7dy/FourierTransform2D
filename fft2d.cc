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

void distribute(int numtasks, int rank, int total_len, int local_num, Complex* matrix);
void transpose(Complex* in, Complex* out, int w, int h);
void centralize(int numtasks, int rank, int total_len, int local_num, Complex* matrix);
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
  int total_len = height * width;
  int local_num = width * height / numtasks;
  Complex* input = image.GetImageData();

  
  Complex* after1d = new Complex[total_len];
  // Complex* after1d_local = new Complex[local_num];

  // since all nodes read the input, do the first 1d fft locally
  // Complex* after1d_local = new Complex[local_num];
  int startRow = height * rank / numtasks;
  int global_offset = 0;
  int local_offset = 0;
  for(int i = 0; i < height / numtasks; i++) {
    global_offset = width * (startRow + i);
    local_offset = width * i;
    Transform1D(input + global_offset, width, after1d + local_offset);
  }

  // slaves send data after 1d fft back to master
  centralize(numtasks, rank, total_len, local_num, after1d);

  if(rank == 0)
  {
    cout<<"Generating Image File MyAfter1d.txt"<<endl;
    image.SaveImageData("MyAfter1d.txt", after1d, width, height);
  }

  // transpose the matrix on master
  Complex* transposed1d = new Complex[total_len];
  if(rank == 0) {
    transpose(after1d, transposed1d, width, height);
  }

  // master sends the transposed data to slaves
  distribute(numtasks, rank, total_len, local_num, transposed1d);

  // slaves do 1d fft, which results in 2d fft
  Complex* after2d = new Complex[total_len];
  for(int i = 0; i < height / numtasks; i++) {
    global_offset = width * (startRow + i);
    local_offset = width * i;
    Transform1D(transposed1d + global_offset, width, after2d + local_offset);
  }

  // slaves send 2d fft results back to master
  centralize(numtasks, rank, total_len, local_num, after2d);

  // master transposes 2d fft result and save
  Complex* transposed2d = new Complex[total_len];
  if(rank == 0) {
    transpose(after2d, transposed2d, width, height);
  }

  if(rank == 0)
  {
    cout<<"Generating Image File MyAfter2d.txt"<<endl;
    image.SaveImageData("MyAfter2d.txt", transposed2d, width, height);
  }



  // delete[] after1d;
  // delete[] after1d_local;
}

void centralize(int numtasks, int rank, int total_len, int local_num, Complex* matrix) {
  if(rank != 0) { // slaves send back to master
    MPI_Request request;
    MPI_Isend(matrix, local_num * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
  }
  else { // master receives all the data from slaves
    for(int i = 1; i < numtasks; i++) {
      MPI_Status status;
      MPI_Recv(matrix + i * local_num, local_num * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
    }
  }
}

void distribute(int numtasks, int rank, int total_len, int local_num, Complex* matrix) {
  if(rank != 0) {
    MPI_Status status;
    MPI_Recv(matrix, total_len * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }
  else {
    for(int i = 1; i < numtasks; i++) {
      MPI_Request request;
      MPI_Isend(matrix, total_len * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &request);
    }
  }
}

void transpose(Complex* in, Complex* out, int w, int h) {
  int p = 0;
  for(int i = 0; i < h; i++) 
    for(int j = 0; j < w; j++) 
      out[p++] = in[i + j * w];
}

void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.
  double coef = 2 * M_PI / w;

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
  

  
